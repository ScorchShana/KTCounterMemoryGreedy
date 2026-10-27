#ifndef KMER_HEADER
#define KMER_HEADER

#include "definition.h"
#include <cstring>
#include <algorithm>
#include <cstddef>
#include "SpinLock.h"

// Operations on borrowed, MSB-first bytes. No per-record objects or ownership.
class kmer
{
    const size_t bytes_ = kmer_bytes();
    std::array<uint8_t, MAX_DEPTH> stored_bytes_{};
    std::array<uint8_t, MAX_DEPTH> removed_bytes_{};
    std::array<uint8_t, MAX_DEPTH> prefix_shift_{};
    std::array<uint8_t, MAX_DEPTH> prefix_mask_{};

public:
    kmer() noexcept
    {
        for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth) {
            const uint32_t start = ROOT_BASES + depth * NODE_BASES;
            removed_bytes_[depth] = std::min(bytes_, size_t{1} + depth / 2);
            stored_bytes_[depth] = std::max(size_t{1}, bytes_ - removed_bytes_[depth]);
            prefix_shift_[depth] = depth % 2 == 0 ? 4 : 0;
            prefix_mask_[depth] = start < k_length ? 0x0f : 0;
        }
    }
    size_t bytes() const noexcept { return bytes_; }
    size_t bytes(uint32_t depth) const noexcept { return stored_bytes_[depth]; }
    size_t removed_bytes(uint32_t depth) const noexcept { return removed_bytes_[depth]; }
    int compare(const unsigned char* a, const unsigned char* b) const noexcept
    { return std::memcmp(a, b, bytes_); }
    bool equal(const unsigned char* a, const unsigned char* b) const noexcept
    { return ::__memcmpeq(a, b, bytes_) == 0; }
    int compare(const unsigned char* a, const unsigned char* b, uint32_t depth) const noexcept
    { return std::memcmp(a, b, bytes(depth)); }
    bool equal(const unsigned char* a, const unsigned char* b, uint32_t depth) const noexcept
    { return ::__memcmpeq(a, b, bytes(depth)) == 0; }
    static uint32_t root_prefix(const unsigned char* key) noexcept { return key[0]; }
    uint32_t node_prefix(const unsigned char* key, uint32_t depth) const noexcept
    { return (key[0] >> prefix_shift_[depth]) & prefix_mask_[depth]; }

    // Only the full-key Tree boundary calls this; same-layer transfers are memcpy.
    void write_root_batch(unsigned char* dest, const unsigned char* full, size_t count) const noexcept
    {
        if (bytes_ == 1) {
            std::memset(dest, 0, count);
        } else {
            const size_t width = bytes(0);
            for (size_t i = 0; i < count; ++i)
                std::memcpy(dest + i * width, full + i * bytes_ + 1, width);
        }
    }
    void write_child(unsigned char* dest, const unsigned char* parent, uint32_t depth) const noexcept
    {
        if (removed_bytes_[depth + 1] == bytes_) {
            dest[0] = 0;
        } else {
            const size_t cut = removed_bytes_[depth + 1] - removed_bytes_[depth];
            std::memcpy(dest, parent + cut, bytes(depth + 1));
        }
    }
    void restore_full_key(unsigned char* full, const unsigned char* suffix,
        uint32_t prefix_code, uint32_t depth) const noexcept
    {
        const size_t cut = removed_bytes_[depth];
        const uint32_t route_bits = 2 * (ROOT_BASES + depth * NODE_BASES);
        for (size_t i = 0; i < cut; ++i)
            full[i] = static_cast<unsigned char>(prefix_code >> (route_bits - 8 * (i + 1)));
        if (cut < bytes_) std::memcpy(full + cut, suffix, bytes_ - cut);
    }
};

struct kmer_block
{
    uint32_t count = 0;
    std::array<unsigned char, KMER_BLOCK_SIZE - sizeof(uint32_t)> k_mers;
};
static_assert(sizeof(kmer_block) == KMER_BLOCK_SIZE);
template<typename CountType = uint32_t>
class ConcurrentOpenAddressHashMap;
template<typename CountType>
struct node
{
    // cache line 0-2 for write
    // SpinLock buffer_lock;
    SpinLock buffer_lock;
    std::atomic<int> writer_count{ 0 }; // 当前正在写入的线程数
    std::atomic<ConcurrentOpenAddressHashMap<CountType>*> hash_map{ nullptr };
    kmer_block* active_block = nullptr;
    uint64_t count = 0; // counter for used block
    std::array<kmer_block*, MAX_KMER_BLOCK_NUM> kmer_blocks{};
    uint32_t prefix_code = 0; // Immutable route prefix, initialized before publication.
    // char padding[3 * CACHE_LINE_SIZE - 3 * sizeof(uint64_t) - sizeof(std::array<kmer_block *, MAX_KMER_BLOCK_NUM>)];

    // cache line 3 for read
    alignas(CACHE_LINE_SIZE) std::atomic<node*> children_ptr = nullptr;
};

static_assert(2 * (ROOT_BASES + (MAX_DEPTH - 1) * NODE_BASES) <= 32);
static_assert(sizeof(node<uint8_t>) == 256 && alignof(node<uint8_t>) == 64);
static_assert(sizeof(node<uint16_t>) == 256 && alignof(node<uint16_t>) == 64);
static_assert(sizeof(node<uint32_t>) == 256 && alignof(node<uint32_t>) == 64);
static_assert(sizeof(node<uint8_t>[16]) == KMER_BLOCK_SIZE);
static_assert(sizeof(node<uint16_t>[16]) == KMER_BLOCK_SIZE);
static_assert(sizeof(node<uint32_t>[16]) == KMER_BLOCK_SIZE);

// const int node_size = sizeof(node<1>);


#endif
