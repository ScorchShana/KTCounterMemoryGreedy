#ifndef FINAL_DRAIN_WRITER_HEADER
#define FINAL_DRAIN_WRITER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "SpinBackoff.h"

#include <cstring>
#include <algorithm>

class FinalDrainWriter
{
private:
    uint64_t local_sorted_kmer_count = 0;
    const size_t kmer_bytes = ::kmer_bytes();
    uint32_t record_bytes_ = static_cast<uint32_t>(kmer_bytes + count_max_bytes);
    uint8_t prefix_bytes_ = 0;
    uint32_t group_records_ = 0;
    uint64_t group_offset_ = 0;

    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY>* pool_;
    char* current_block_;
    uint64_t current_offset_;

    SpinBackoff<> dequeue_backoff;
    SpinBackoff<> enqueue_backoff;

public:
    uint64_t local_filtered_below_min_count = 0;
    uint64_t local_filtered_above_max_count = 0;

#ifdef TEST_MODE
    uint64_t producer_enqueue_spin_time{ 0 };
    uint64_t producer_dequeue_spin_time{ 0 };
#endif

    FinalDrainWriter(const FinalDrainWriter&) = delete;
    FinalDrainWriter& operator=(const FinalDrainWriter&) = delete;
    FinalDrainWriter(FinalDrainWriter&&) = delete;
    FinalDrainWriter& operator=(FinalDrainWriter&&) = delete;

    explicit FinalDrainWriter(RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY>* pool) :
        local_sorted_kmer_count(0),
        pool_(pool), current_block_(nullptr), current_offset_(0)
    {
        pool_->producer_dequeue(current_block_);
    }

    void close()
    {
        if (current_block_ != nullptr)
        {
            finish_group();
            pool_->producer_enqueue({ current_block_, current_offset_ });
            current_block_ = nullptr;
        }
        sorted_kmer_count.fetch_add(local_sorted_kmer_count, std::memory_order_relaxed);
        local_sorted_kmer_count = 0;
        if (local_filtered_below_min_count != 0)
        {
            filtered_below_min_count.fetch_add(local_filtered_below_min_count, std::memory_order_relaxed);
            local_filtered_below_min_count = 0;
        }
        if (local_filtered_above_max_count != 0)
        {
            filtered_above_max_count.fetch_add(local_filtered_above_max_count, std::memory_order_relaxed);
            local_filtered_above_max_count = 0;
        }
    }

    ~FinalDrainWriter()
    {
        if (!RunState::active || !RunState::active->failed()) close();
    }

    // Tree supplies P < key width. Prefix bytes come from the first emitted key;
    // the disk header matches low.bin: uint8 P, native uint32 N, P key bytes.
    void begin_node(uint8_t prefix_bytes)
    {
        finish_group();
        prefix_bytes_ = prefix_bytes;
        record_bytes_ = static_cast<uint32_t>(kmer_bytes - prefix_bytes_ + count_max_bytes);
    }

    void write_kmer_record(const unsigned char* kmer_data, const uint32_t count)
    {
        local_sorted_kmer_count++;
        if (current_offset_ + record_bytes_ > FINAL_DRAIN_RING_POOL_BLOCK_SIZE) [[unlikely]]
            flush_block();

        if (group_records_ == 0) [[unlikely]]
        {
            if (current_offset_ + 5 + prefix_bytes_ + record_bytes_ > FINAL_DRAIN_RING_POOL_BLOCK_SIZE)
                flush_block();
            group_offset_ = current_offset_;
            current_block_[current_offset_] = static_cast<char>(prefix_bytes_);
            current_offset_ += 5; // N is filled before the block is published.
            std::memcpy(current_block_ + current_offset_, kmer_data, prefix_bytes_);
            current_offset_ += prefix_bytes_;
        }

        std::memcpy(current_block_ + current_offset_, kmer_data + prefix_bytes_, kmer_bytes - prefix_bytes_);
        current_offset_ += kmer_bytes - prefix_bytes_;

        std::memcpy(current_block_ + current_offset_, &count, count_max_bytes);
        current_offset_ += count_max_bytes;
        ++group_records_;
    }

private:
    void finish_group()
    {
        if (group_records_ != 0)
        {
            std::memcpy(current_block_ + group_offset_ + 1, &group_records_, sizeof(group_records_));
            group_records_ = 0;
        }
    }

    void flush_block()
    {
        finish_group();
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
