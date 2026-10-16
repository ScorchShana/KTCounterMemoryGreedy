#pragma once

#include "ConcurrentMemoryPool.h"
#include "kmer.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

// Same byte layout as ExportWriter: native full words, then the high bytes
// of the last word. Storage is compact; computational keys remain aligned.
template <uint32_t N>
class PackedKmerCodec
{
    size_t full_words_;
    size_t tail_bytes_;
    uint64_t tail_mask_;

public:
    explicit PackedKmerCodec(uint32_t k) noexcept
        : full_words_(k / BASES_PER_U64T),
          tail_bytes_(((k % BASES_PER_U64T) * 2 + 7) / 8),
          tail_mask_(k % BASES_PER_U64T == 0 ? 0 :
              ~uint64_t{0} << (64 - 2 * (k % BASES_PER_U64T)))
    {}

    size_t packed_bytes() const noexcept { return full_words_ * sizeof(uint64_t) + tail_bytes_; }

    void pack(const kmer<N>& key, uint8_t* output) const noexcept
    {
        const size_t full_bytes = full_words_ * sizeof(uint64_t);
        std::memcpy(output, key.data.data(), full_bytes);
        if (tail_bytes_ != 0) {
            const uint64_t tail = key.data[full_words_] & tail_mask_;
            std::memcpy(output + full_bytes,
                reinterpret_cast<const uint8_t*>(&tail) + sizeof(tail) - tail_bytes_, tail_bytes_);
        }
    }

    void unpack(const uint8_t* input, kmer<N>& key) const noexcept
    {
        key.reset();
        const size_t full_bytes = full_words_ * sizeof(uint64_t);
        std::memcpy(key.data.data(), input, full_bytes);
        if (tail_bytes_ != 0) {
            std::memcpy(reinterpret_cast<uint8_t*>(&key.data[full_words_]) + sizeof(uint64_t) - tail_bytes_,
                input + full_bytes, tail_bytes_);
            key.data[full_words_] &= tail_mask_;
        }
    }

    uint32_t root_prefix(const uint8_t* input) const noexcept
    {
        uint64_t first = 0;
        if (full_words_ != 0) {
            std::memcpy(&first, input, sizeof(first));
        } else {
            std::memcpy(reinterpret_cast<uint8_t*>(&first) + sizeof(first) - tail_bytes_, input, tail_bytes_);
            first &= tail_mask_;
        }
        return static_cast<uint32_t>(first >> (64 - 2 * ROOT_BASES));
    }
};

// Object and packed slots share one pool allocation. Like other allocate_large
// clients, lifetime ends with the pool: never delete/free or deallocate this ring.
template <uint32_t N>
class PackedKmerRing
{
    PackedKmerCodec<N> codec_;
    uint8_t* storage_;
    size_t capacity_;
    size_t allocated_bytes_;
    size_t write_pos_ = 0;
    size_t size_ = 0;

    PackedKmerRing(uint32_t k, uint8_t* storage, size_t capacity, size_t allocated_bytes) noexcept
        : codec_(k), storage_(storage), capacity_(capacity), allocated_bytes_(allocated_bytes)
    {}

public:
    PackedKmerRing(const PackedKmerRing&) = delete;
    PackedKmerRing& operator=(const PackedKmerRing&) = delete;

    static PackedKmerRing* create(ConcurrentMemoryPool* pool, uint32_t k, size_t max_bytes)
    {
        const size_t usable_bytes = max_bytes / BLOCK_SIZE * BLOCK_SIZE;
        constexpr size_t header_bytes = align_up(sizeof(PackedKmerRing), CACHE_LINE_SIZE);
        const size_t packed_bytes = (size_t{k} + 3) / 4;
        if (k == 0 || k > N * BASES_PER_U64T || usable_bytes < header_bytes ||
            usable_bytes - header_bytes < packed_bytes) {
            std::cerr << "PackedKmerRing budget or k-mer length is invalid\n";
            std::exit(-1);
        }
        const size_t capacity = (usable_bytes - header_bytes) / packed_bytes;
        const size_t bytes = header_bytes + capacity * packed_bytes;
        auto* memory = static_cast<uint8_t*>(pool->allocate_large(bytes));
        return ::new (memory) PackedKmerRing(k, memory + header_bytes, capacity, align_up(bytes, BLOCK_SIZE));
    }

    const PackedKmerCodec<N>& codec() const noexcept { return codec_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t size() const noexcept { return size_; }
    size_t allocated_bytes() const noexcept { return allocated_bytes_; }

    // Output is untouched until the ring is full, and must not alias key/storage.
    bool push(const kmer<N>& key, uint8_t* evicted_packed) noexcept
    {
        const size_t width = codec_.packed_bytes();
        uint8_t* slot = storage_ + write_pos_ * width;
        const bool full = size_ == capacity_;
        if (full) std::memcpy(evicted_packed, slot, width);
        else ++size_;
        codec_.pack(key, slot);
        if (++write_pos_ == capacity_) write_pos_ = 0;
        return full;
    }

    bool pop_oldest(uint8_t* output) noexcept
    {
        if (size_ == 0) return false;
        const size_t oldest = write_pos_ >= size_ ? write_pos_ - size_ : capacity_ - (size_ - write_pos_);
        std::memcpy(output, storage_ + oldest * codec_.packed_bytes(), codec_.packed_bytes());
        --size_;
        return true;
    }
};
