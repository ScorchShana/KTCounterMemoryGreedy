#ifndef LOW_FREQUENCY_PARSER_HEADER
#define LOW_FREQUENCY_PARSER_HEADER

#include "kmer.h"
#include "RingMemoryPool.h"
#include "SpinBackoff.h"
#include "SplitMix.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

class LowFrequencyParser
{
    const size_t BATCH_CAPACITY = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
    const std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>>& classifier_task_queues;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> local_prefix_owners;
    std::array<unsigned char, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE> grouped;
    std::vector<size_t> owner_counts;
    std::vector<size_t> owner_sums;
    std::vector<content_type> owner_contents;
    SpinBackoff<64, 128, 160> enqueue_backoff;
    SplitMix64 rng;
    inline static std::atomic<uint64_t> rng_seed{0};

public:
    LowFrequencyParser(
        const std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>>& queues,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_queue)
        : classifier_task_queues(queues), global_classifier_task_queue(global_queue),
          local_prefix_owners(prefix_owners), owner_counts(queues.size()), owner_sums(queues.size()),
          owner_contents(queues.size())
    {
        SplitMix64::seed(rng_seed.fetch_add(1));
    }

    size_t packed_bytes() const noexcept { return kmer_bytes(); }

    void parse_and_push(SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>& reads,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>& parsed)
    {
        const size_t capacity = parsed.blockSize() / kmer_bytes();
        const size_t width = kmer_bytes();
        SpinBackoff<> backoff;
        content_type content;
        while (true)
        {
            bool available = reads.consumer_try_dequeue(content);
            if (!available && reads.producer_finished()) {
                available = reads.consumer_try_dequeue(content);
                if (!available) break;
            }
            if (!available) {
                backoff.backoff();
                continue;
            }
            backoff.decay();
            size_t pos = 0;
            while (pos < content.length)
            {
                const size_t count = std::min<size_t>(capacity, content.length - pos);
                group_and_push(reinterpret_cast<const unsigned char*>(content.data) + pos * width, count, parsed);
                pos += count;
            }
            reads.consumer_enqueue(content.data);
        }
        flush_owner_blocks();
    }

private:
    size_t get_classifier_owner(const unsigned char* key) const noexcept
    {
        return local_prefix_owners[key[0]];
    }

    void enqueue_content_to_classifier(size_t owner)
    {
        const auto& content = owner_contents[owner];
        const bool reverse_enqueue = classifier_task_queues[owner]->size() >= CLASSIFIER_TASK_QUEUE_HALF_WATERMARK;
        bool submitted;
        if (reverse_enqueue)
            submitted = global_classifier_task_queue->try_enqueue(content) || classifier_task_queues[owner]->try_enqueue(content);
        else
            submitted = classifier_task_queues[owner]->try_enqueue(content) || global_classifier_task_queue->try_enqueue(content);
        if (submitted) {
            enqueue_backoff.double_decay();
        } else {
            do {
                enqueue_backoff.backoff();
                if (reverse_enqueue)
                    submitted = global_classifier_task_queue->try_enqueue(content) || classifier_task_queues[owner]->try_enqueue(content);
                else
                    submitted = classifier_task_queues[owner]->try_enqueue(content) || global_classifier_task_queue->try_enqueue(content);
            } while (!submitted);
            enqueue_backoff.decay();
        }
        owner_contents[owner] = {nullptr, 0};
    }

    void flush_owner_blocks()
    {
        for (size_t owner = 0; owner < owner_contents.size(); ++owner)
            if (owner_contents[owner].data != nullptr) enqueue_content_to_classifier(owner);
    }

    void group_and_push(const unsigned char* decoded, size_t count,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>& parsed)
    {
        std::fill(owner_counts.begin(), owner_counts.end(), 0);
        for (size_t i = 0; i < count; ++i) ++owner_counts[get_classifier_owner((decoded + (i) * kmer_bytes()))];
        owner_sums[0] = 0;
        for (size_t owner = 1; owner < owner_sums.size(); ++owner)
            owner_sums[owner] = owner_sums[owner - 1] + owner_counts[owner - 1];
        for (size_t i = 0; i < count; ++i)
            std::memcpy(grouped.data() + owner_sums[get_classifier_owner(decoded + i * kmer_bytes())]++ * kmer_bytes(), decoded + i * kmer_bytes(), kmer_bytes());

        const size_t capacity = parsed.blockSize() / kmer_bytes();
        const size_t owner_start = rng() % owner_contents.size();
        for (size_t i = 0; i < owner_contents.size(); ++i) {
            const size_t owner = (i + owner_start) % owner_contents.size();
            size_t remaining = owner_counts[owner];
            size_t offset = owner_sums[owner] - remaining;
            auto& content = owner_contents[owner];
            while (remaining > 0) {
                if (content.data == nullptr) {
                    char* block;
                    if (!parsed.producer_try_dequeue(block)) {
                        // Publish partial blocks before waiting so consumers can return memory.
                        flush_owner_blocks();
                        parsed.producer_dequeue(block);
                    }
                    content = {block, 0};
                }
                const size_t copied = std::min<size_t>(capacity - content.length, remaining);
                std::memcpy(content.data + content.length * kmer_bytes(), grouped.data() + offset * kmer_bytes(),
                    copied * kmer_bytes());
                content.length += copied;
                offset += copied;
                remaining -= copied;
                if (content.length == capacity) enqueue_content_to_classifier(owner);
            }
        }
    }
};

#endif
