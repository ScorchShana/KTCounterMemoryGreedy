#ifndef LOW_FREQUENCY_PARSER_HEADER
#define LOW_FREQUENCY_PARSER_HEADER

#include "kmer.h"
#include "MsdPrefixCodec.h"
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
        content_type content{};
        try {
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
                    const auto input = std::span(reinterpret_cast<const uint8_t*>(content.data) + pos, content.length - pos);
                    const auto group = msd_prefix::parse_group_header(input, width);
                    if (input.size() < group.encoded_bytes) throw std::runtime_error("truncated pending parser group");
                    const auto* common = input.data() + msd_prefix::GROUP_HEADER_BYTES;
                    const auto* suffix = common + group.prefix_bytes;
                    if (group.prefix_bytes) {
                        restore_to_owner(common, suffix, group.prefix_bytes, group.records, parsed);
                    } else {
                        for (size_t done = 0; done < group.records;) {
                            const size_t count = std::min(capacity, size_t(group.records) - done);
                            group_and_push(suffix + done * width, count, parsed);
                            done += count;
                        }
                    }
                    pos += group.encoded_bytes;
                }
                reads.consumer_enqueue(content.data);
                content = {};
            }
            flush_owner_blocks();
        } catch (...) {
            // These blocks are privately held, so each return queue has room.
            // Avoid failure-aware blocking operations while already unwinding.
            if (content.data)
                while (!reads.consumer_try_enqueue(content.data)) std::this_thread::yield();
            for (auto& tail : owner_contents) if (tail.data) {
                while (!parsed.consumer_try_enqueue(tail.data)) std::this_thread::yield();
                tail = {};
            }
            throw;
        }
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

    void restore_to_owner(const uint8_t* prefix, const uint8_t* suffix, size_t prefix_bytes,
        size_t count, RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>& parsed)
    {
        const size_t width = kmer_bytes();
        const size_t suffix_bytes = width - prefix_bytes;
        const size_t owner = get_classifier_owner(prefix);
        auto& content = owner_contents[owner];
        while (count) {
            if (!content.data) {
                char* block;
                if (!parsed.producer_try_dequeue(block)) {
                    flush_owner_blocks();
                    parsed.producer_dequeue(block);
                }
                content = {block, 0};
            }
            const size_t take = std::min(count, parsed.blockSize() / width - content.length);
            auto* target = reinterpret_cast<uint8_t*>(content.data) + content.length * width;
            if (prefix_bytes == 1) {
                const uint8_t common = prefix[0];
                for (size_t i = 0; i < take; ++i) {
                    target[0] = common;
                    std::memcpy(target + 1, suffix, suffix_bytes);
                    target += width;
                    suffix += suffix_bytes;
                }
            } else {
                for (size_t i = 0; i < take; ++i) {
                    std::memcpy(target, prefix, prefix_bytes);
                    std::memcpy(target + prefix_bytes, suffix, suffix_bytes);
                    target += width;
                    suffix += suffix_bytes;
                }
            }
            content.length += take;
            count -= take;
            if (content.length == parsed.blockSize() / width) enqueue_content_to_classifier(owner);
        }
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
