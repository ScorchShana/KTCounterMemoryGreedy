#ifndef FINAL_DRAIN_WRITER_HEADER
#define FINAL_DRAIN_WRITER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "SpinBackoff.h"

#include <cstring>
#include <algorithm>

template<uint32_t N>
class FinalDrainWriter
{
private:
    uint64_t local_sorted_kmer_count = 0;
    const uint32_t k;
    const uint64_t tail_bits;
    const uint64_t tail_bytes;
    const uint64_t full_words;
    const uint64_t kmer_bytes;
    const uint32_t total_bytes;
    const uint64_t mask;

    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY>* pool_;
    char* current_block_;
    uint64_t current_offset_;

    SpinBackoff<> dequeue_backoff;
    SpinBackoff<> enqueue_backoff;

public:

#ifdef TEST_MODE
    uint64_t producer_enqueue_spin_time{ 0 };
    uint64_t producer_dequeue_spin_time{ 0 };
#endif

    FinalDrainWriter(const FinalDrainWriter&) = delete;
    FinalDrainWriter& operator=(const FinalDrainWriter&) = delete;
    FinalDrainWriter(FinalDrainWriter&&) = delete;
    FinalDrainWriter& operator=(FinalDrainWriter&&) = delete;

    explicit FinalDrainWriter(uint32_t in_k,
        RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY>* pool) :
        local_sorted_kmer_count(0),
        k(in_k), tail_bits(2 * (k % BASES_PER_U64T)), tail_bytes((tail_bits + 7) / 8), full_words(k / BASES_PER_U64T),
        kmer_bytes(full_words * sizeof(uint64_t) + tail_bytes),
        total_bytes(static_cast<uint32_t>(kmer_bytes + count_max_bytes)),
        mask((~uint64_t{ 0 }) << (64 - tail_bits)),
        pool_(pool), current_block_(nullptr), current_offset_(0)
    {
        pool_->producer_dequeue(current_block_);
    }

    void close()
    {
        if (current_block_ != nullptr)
        {
            pool_->producer_enqueue({ current_block_, current_offset_ });
            current_block_ = nullptr;
        }
        sorted_kmer_count.fetch_add(local_sorted_kmer_count, std::memory_order_relaxed);
        local_sorted_kmer_count = 0;
    }

    ~FinalDrainWriter()
    {
        close();
    }

    void write_kmer_record(const uint64_t* kmer_data, const uint32_t count)
    {
        local_sorted_kmer_count++;
        if (current_offset_ + total_bytes > FINAL_DRAIN_RING_POOL_BLOCK_SIZE) [[unlikely]]
            flush_block();

        std::memcpy(current_block_ + current_offset_, kmer_data,
            full_words * sizeof(uint64_t));
        current_offset_ += full_words * sizeof(uint64_t);

        uint64_t tail_data = kmer_data[full_words] & mask;
        std::memcpy(current_block_ + current_offset_,
            reinterpret_cast<const char*>(&tail_data) + (8 - tail_bytes), tail_bytes);
        current_offset_ += tail_bytes;

        std::memcpy(current_block_ + current_offset_, &count, count_max_bytes);
        current_offset_ += count_max_bytes;
    }

    void write_map_record(const concurrent_node<N>* nodes, const uint32_t count)
    {
        local_sorted_kmer_count += count;

        const uint32_t first_to_write = std::min<uint32_t>(count,
            (FINAL_DRAIN_RING_POOL_BLOCK_SIZE - current_offset_) / total_bytes);
        for (uint32_t i = 0; i < first_to_write; i++)
        {
            const auto& node = nodes[i];
            const uint32_t rec_count = node.count.load(std::memory_order_relaxed);

            if (rec_count + 1 < filter_min || rec_count > filter_max) [[unlikely]]
            {
                continue;
            }

            std::memcpy(current_block_ + current_offset_, node.k_mer.data.data(),
                full_words * sizeof(uint64_t));
            current_offset_ += full_words * sizeof(uint64_t);

            uint64_t tail_data = node.k_mer.data[full_words] & mask;
            std::memcpy(current_block_ + current_offset_,
                reinterpret_cast<const char*>(&tail_data) + (8 - tail_bytes), tail_bytes);
            current_offset_ += tail_bytes;

            std::memcpy(current_block_ + current_offset_, &rec_count, count_max_bytes);
            current_offset_ += count_max_bytes;
        }

        if (first_to_write < count) [[unlikely]]
        {
            flush_block();

            for (uint32_t i = first_to_write; i < count; i++)
            {
                const auto& node = nodes[i];
                const uint32_t rec_count = node.count.load(std::memory_order_relaxed);

                if (rec_count + 1 < filter_min || rec_count > filter_max) [[unlikely]]
                {
                    continue;
                }

                std::memcpy(current_block_ + current_offset_, node.k_mer.data.data(),
                    full_words * sizeof(uint64_t));
                current_offset_ += full_words * sizeof(uint64_t);

                uint64_t tail_data = node.k_mer.data[full_words] & mask;
                std::memcpy(current_block_ + current_offset_,
                    reinterpret_cast<const char*>(&tail_data) + (8 - tail_bytes), tail_bytes);
                current_offset_ += tail_bytes;

                std::memcpy(current_block_ + current_offset_, &rec_count, count_max_bytes);
                current_offset_ += count_max_bytes;
            }
        }
    }

private:
    void flush_block()
    {
        if (pool_->producer_try_enqueue({ current_block_, current_offset_ }))
        {
            enqueue_backoff.double_decay();
        }
        else
        {
#ifdef TEST_MODE
            producer_enqueue_spin_time++;
#endif
            enqueue_backoff.backoff();
            while (!pool_->producer_try_enqueue({ current_block_, current_offset_ }))
            {
                enqueue_backoff.backoff();
#ifdef TEST_MODE
                producer_enqueue_spin_time++;
#endif
            }
            enqueue_backoff.decay();
        }

        if (pool_->producer_try_dequeue(current_block_))
        {
            dequeue_backoff.double_decay();
        }
        else
        {
#ifdef TEST_MODE
            producer_dequeue_spin_time++;
#endif
            dequeue_backoff.backoff();
            while (!pool_->producer_try_dequeue(current_block_))
            {
                dequeue_backoff.backoff();
#ifdef TEST_MODE
                producer_dequeue_spin_time++;
#endif
            }
            dequeue_backoff.decay();
        }

        current_offset_ = 0;
    }
};

#endif
