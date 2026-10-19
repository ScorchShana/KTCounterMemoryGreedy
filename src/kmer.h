#ifndef KMER_HEADER
#define KMER_HEADER

#include "definition.h"
#include <cstring>
#include "SpinLock.h"

// Operations on borrowed, MSB-first bytes. No per-record objects or ownership.
class kmer
{
    const size_t bytes_ = kmer_bytes();
    std::array<uint8_t, MAX_DEPTH> prefix_byte_{};
    std::array<uint8_t, MAX_DEPTH> prefix_shift_{};
    std::array<uint8_t, MAX_DEPTH> prefix_mask_{};

public:
    kmer() noexcept
    {
        for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth) {
            const uint32_t start = ROOT_BASES + depth * NODE_BASES;
            prefix_byte_[depth] = start < k_length ? start / 4 : 0;
            prefix_shift_[depth] = 4 - (start % 4) * 2;
            prefix_mask_[depth] = start < k_length ? 0x0f : 0;
        }
    }
    size_t bytes() const noexcept { return bytes_; }
    int compare(const unsigned char* a, const unsigned char* b) const noexcept
    { return std::memcmp(a, b, bytes_); }
    bool equal(const unsigned char* a, const unsigned char* b) const noexcept
    { return compare(a, b) == 0; }
    static uint32_t root_prefix(const unsigned char* key) noexcept { return key[0]; }
    uint32_t node_prefix(const unsigned char* key, uint32_t depth) const noexcept
    { return (key[prefix_byte_[depth]] >> prefix_shift_[depth]) & prefix_mask_[depth]; }
};

struct kmer_block
{
    uint32_t count = 0;
    std::array<unsigned char, KMER_BLOCK_SIZE - sizeof(uint32_t)> k_mers;
};
static_assert(sizeof(kmer_block) == KMER_BLOCK_SIZE);
class ConcurrentOpenAddressHashMap;
struct node
{
    // cache line 0-2 for write
    // SpinLock buffer_lock;
    SpinLock buffer_lock;
    std::atomic<int> writer_count{ 0 }; // 当前正在写入的线程数
    std::atomic<ConcurrentOpenAddressHashMap*> hash_map{ nullptr };
    kmer_block* active_block = nullptr;
    uint64_t count = 0; // counter for used block
    std::array<kmer_block*, MAX_KMER_BLOCK_NUM> kmer_blocks{};
    // char padding[3 * CACHE_LINE_SIZE - 3 * sizeof(uint64_t) - sizeof(std::array<kmer_block *, MAX_KMER_BLOCK_NUM>)];

    // cache line 3 for read
    alignas(CACHE_LINE_SIZE) std::atomic<node*> children_ptr = nullptr;
};

// const int node_size = sizeof(node<1>);


#endif
