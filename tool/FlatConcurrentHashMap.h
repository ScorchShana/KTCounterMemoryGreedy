#ifndef FLAT_CONCURRENT_HASH_MAP_HEADER
#define FLAT_CONCURRENT_HASH_MAP_HEADER

#include "../src/definition.h"
#include "../src/kmer.h"
#include "../include/xxh3.h"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <iostream>
#include <sys/mman.h>
#include <thread>
#include <vector>


#if defined(__AVX2__)
#define FLAT_HASHMAP_AVX2 1
#define FLAT_HASHMAP_GROUP_SIZE 32
#elif defined(__SSE4_2__)
#define FLAT_HASHMAP_SSE 1
#define FLAT_HASHMAP_GROUP_SIZE 16
#else
#define FLAT_HASHMAP_SCALAR 1
#define FLAT_HASHMAP_GROUP_SIZE 8
#endif

#if defined(FLAT_HASHMAP_AVX2) || defined(FLAT_HASHMAP_SSE)
#include <immintrin.h>
#endif

#if defined(HISTOGRAM_COMPACT_KEYS) && !HISTOGRAM_COMPACT_KEYS
#error "Non-compact histogram keys are no longer supported; remove HISTOGRAM_COMPACT_KEYS=0"
#endif

template <uint32_t N>
class FlatConcurrentHashMap
{
public:
    static constexpr double LOAD_FACTOR = 0.65;
    static constexpr uint64_t GROUP_SIZE = FLAT_HASHMAP_GROUP_SIZE;

    using Entry = ExportRecord<N>;

    struct PreparedLookup
    {
        uint64_t base = 0;
        uint8_t fp = 0;
    };

    [[nodiscard]] static uint64_t required_mmap_bytes(uint64_t expected_unique_key, uint32_t k_len = 32 * N)
    {
        return calculate_layout(expected_unique_key, k_len).mmap_bytes;
    }

    explicit FlatConcurrentHashMap(uint64_t expected_unique, uint32_t in_n_thread = 1, uint32_t k_len = 32 * N)
        : FlatConcurrentHashMap(calculate_layout(expected_unique, k_len), in_n_thread)
    {
    }

    ~FlatConcurrentHashMap() noexcept
    {
        if (ctrl_ != nullptr)
        {
            munmap(ctrl_, ctrl_size_);
            ctrl_ = nullptr;
        }
        if (packed_keys_ != nullptr)
        {
            munmap(packed_keys_, kmer_size_);
            packed_keys_ = nullptr;
        }
        if (count_ != nullptr)
        {
            munmap(count_, count_size_);
            count_ = nullptr;
        }
    }

    FlatConcurrentHashMap(const FlatConcurrentHashMap&) = delete;
    FlatConcurrentHashMap& operator=(const FlatConcurrentHashMap&) = delete;
    FlatConcurrentHashMap(FlatConcurrentHashMap&&) = delete;
    FlatConcurrentHashMap& operator=(FlatConcurrentHashMap&&) = delete;

    bool insert_unique(const kmer<N>& key, uint32_t count) noexcept
    {
        if (key_bytes_ == sizeof(key))
            return insert_unique(reinterpret_cast<const char*>(&key), count);
        char packed[sizeof(key)];
        pack_key(packed, reinterpret_cast<const char*>(&key));
        return insert_unique(packed, count);
    }

    // Packed keys are hashed, stored and compared without expansion.
    bool insert_unique(const char* packed_key, uint32_t count) noexcept
    {
        if (sealed_.load(std::memory_order_acquire)) [[unlikely]]
        {
            return false;
        }

        const uint64_t hash = XXH3_64bits(packed_key, key_bytes_);
        const uint8_t fp = fingerprint(hash);
        uint64_t base = initial_slot(hash);

        for (uint64_t probe = 0; probe < group_count_; ++probe)
        {
            for (uint64_t offset = 0; offset < GROUP_SIZE; ++offset)
            {
                const uint64_t slot = wrap_slot(base + offset);
                std::atomic_ref<uint8_t> ctrl_ref(ctrl_[slot]);
                if (ctrl_ref.load(std::memory_order_acquire) != 0)
                {
                    cpu_relax();
                    continue;
                }
                uint8_t expected = 0;
                if (ctrl_ref.compare_exchange_strong(
                    expected,
                    fp,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
                {
                    std::memcpy(packed_keys_ + slot * key_bytes_, packed_key, key_bytes_);
                    count_[slot] = count;

                    // size_.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                cpu_relax();
            }
            base = next_group_base(base);
        }

        return false;
    }

    void seal() noexcept
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (uint64_t i = 0; i < GROUP_SIZE - 1; ++i)
        {
            ctrl_[static_cast<size_t>(capacity_ + i)] = ctrl_[static_cast<size_t>(i)];
        }
        sealed_.store(true, std::memory_order_release);
    }

    bool find(const kmer<N>& key, uint32_t& out_count) const noexcept
    {
        const PreparedLookup lookup = prepare_lookup(key);
        return find_prepared(key, lookup, out_count);
    }

    [[nodiscard]] PreparedLookup prepare_lookup(const kmer<N>& key) const noexcept
    {
        if (key_bytes_ == sizeof(key))
            return prepare_lookup(reinterpret_cast<const char*>(&key));
        char packed[sizeof(key)];
        pack_key(packed, reinterpret_cast<const char*>(&key));
        return prepare_lookup(packed);
    }

    [[nodiscard]] PreparedLookup prepare_lookup(const char* packed_key) const noexcept
    {
        const uint64_t hash = XXH3_64bits(packed_key, key_bytes_);
        return PreparedLookup{ initial_slot(hash), fingerprint(hash) };
    }

    void prefetch(const PreparedLookup& lookup) const noexcept
    {
        constexpr int CTRL_PREFETCH_LOCALITY = 0;
        constexpr int KMER_PREFETCH_LOCALITY = 0;
        constexpr int COUNT_PREFETCH_LOCALITY = 0;
#if defined(__GNUC__) || defined(__clang__)
        const uint8_t* ctrl = ctrl_ + lookup.base;
        const uint32_t* count = count_ + lookup.base;
        __builtin_prefetch(ctrl, 0, CTRL_PREFETCH_LOCALITY);
        // __builtin_prefetch(ctrl + GROUP_SIZE - 1, 0, CTRL_PREFETCH_LOCALITY);
        __builtin_prefetch(packed_keys_ + lookup.base * key_bytes_, 0, KMER_PREFETCH_LOCALITY);
        __builtin_prefetch(count, 0, COUNT_PREFETCH_LOCALITY);

#else
        (void)lookup;
#endif
    }

    bool find_prepared(const kmer<N>& key, const PreparedLookup& lookup, uint32_t& out_count) const noexcept
    {
        uint64_t slot;
        return find_prepared_slot(key, lookup, out_count, slot);
    }

    bool find_prepared(const char* packed_key, const PreparedLookup& lookup, uint32_t& out_count) const noexcept
    {
        uint64_t slot;
        return find_prepared_slot(packed_key, lookup, out_count, slot);
    }

    bool find_prepared_slot(const kmer<N>& key, const PreparedLookup& lookup,
                            uint32_t& out_count, uint64_t& out_slot) const noexcept
    {
        if (key_bytes_ == sizeof(key))
            return find_prepared_slot(reinterpret_cast<const char*>(&key), lookup, out_count, out_slot);
        char packed[sizeof(key)];
        pack_key(packed, reinterpret_cast<const char*>(&key));
        return find_prepared_slot(packed, lookup, out_count, out_slot);
    }

    bool find_prepared_slot(const char* packed_key, const PreparedLookup& lookup,
                            uint32_t& out_count, uint64_t& out_slot) const noexcept
    {
        const uint8_t fp = lookup.fp;
        uint64_t base = lookup.base;

        for (uint64_t probe = 0; probe < group_count_; ++probe)
        {
            const auto masks = match_and_empty(base, fp);
            uint32_t match_mask = masks.first;
            const uint32_t empty_mask = masks.second;

            while (match_mask != 0)
            {
                const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(match_mask));
                const uint64_t slot = wrap_slot(base + bit);
                if (std::memcmp(packed_keys_ + slot * key_bytes_, packed_key, key_bytes_) == 0) [[likely]]
                {
                    out_count = count_[slot];
                    out_slot = slot;
                    return true;
                }
                match_mask &= match_mask - 1;
            }

            if (empty_mask != 0)
            {
                return false;
            }
            base = next_group_base(base);
        }

        return false;
    }

    bool contains(const kmer<N>& key) const noexcept
    {
        uint32_t ignored = 0;
        return find(key, ignored);
    }

    uint64_t size() const noexcept
    {
        return size_.load(std::memory_order_relaxed);
    }

    uint64_t capacity() const noexcept
    {
        return capacity_;
    }

    template <typename Func>
    void for_each_entry(Func&& func) const
    {
        for (uint64_t i = 0; i < capacity_; ++i)
        {
            if (ctrl_[i] != 0)
            {
                // Legacy callbacks borrow a reconstructed key only for this call.
                kmer<N> key{};
                const char* stored = packed_keys_ + i * key_bytes_;
                const uint32_t full_bytes = key_bytes_ & ~7U;
                const uint32_t tail_bytes = key_bytes_ & 7U;
                std::memcpy(key.data.data(), stored, full_bytes);
                if (tail_bytes)
                    std::memcpy(reinterpret_cast<char*>(key.data.data()) + full_bytes + 8 - tail_bytes,
                                stored + full_bytes, tail_bytes);
                func(key, count_[i]);
            }
        }
    }

    uint32_t* mutable_count_at(uint64_t slot)
    {
        return &count_[slot];
    }

    const uint32_t* count_at(uint64_t slot) const
    {
        return &count_[slot];
    }

private:
    // Legacy callers supply the full-width representation.
    void pack_key(char* stored, const char* key) const noexcept
    {
        const uint32_t full_bytes = key_bytes_ & ~7U;
        const uint32_t tail_bytes = key_bytes_ & 7U;
        if (full_bytes) std::memcpy(stored, key, full_bytes);
        if (tail_bytes)
            std::memcpy(stored + full_bytes, key + full_bytes + 8 - tail_bytes, tail_bytes);
    }

    static constexpr uint64_t HUGE_PAGE_BYTES = 2ULL * 1024ULL * 1024ULL;
    static_assert(std::is_trivially_destructible_v<Entry>, "Entry must be trivially destructible");

    struct Layout
    {
        uint64_t capacity;
        uint32_t key_bytes;
        uint64_t ctrl_bytes;
        uint64_t kmer_bytes;
        uint64_t count_bytes;
        uint64_t mmap_bytes;
    };

    explicit FlatConcurrentHashMap(const Layout& layout, uint32_t in_n_thread)
        : capacity_(layout.capacity),
        group_count_(capacity_ / GROUP_SIZE),
        inv_(((__uint128_t)1 << 64) / capacity_),
        mmap_size_(layout.mmap_bytes),
        ctrl_size_(layout.ctrl_bytes),
        key_bytes_(layout.key_bytes),
        kmer_size_(layout.kmer_bytes),
        count_size_(layout.count_bytes),
        n_thread(in_n_thread)
    {
        std::tuple<void*, void*, void*> ptrs;
        ptrs = mmap_hash_map_memory(ctrl_size_, kmer_size_, count_size_);
        ctrl_ = static_cast<uint8_t*>(std::get<0>(ptrs));
        packed_keys_ = static_cast<char*>(std::get<1>(ptrs));
        count_ = static_cast<uint32_t*>(std::get<2>(ptrs));
    }

    static uint64_t round_up(uint64_t value, uint64_t alignment)
    {
        const uint64_t remainder = value % alignment;
        if (remainder && value > std::numeric_limits<uint64_t>::max() - (alignment - remainder))
            throw std::length_error("hash table size overflow");
        return remainder == 0 ? value : value + alignment - remainder;
    }

    static uint64_t calculate_capacity(uint64_t expected_unique)
    {
        if (expected_unique == 0)
        {
            return GROUP_SIZE;
        }

        const long double raw =
            static_cast<long double>(expected_unique) / static_cast<long double>(LOAD_FACTOR);
        const uint64_t maximum = std::numeric_limits<uint64_t>::max() / GROUP_SIZE * GROUP_SIZE;
        if (raw > static_cast<long double>(maximum)) throw std::length_error("hash table capacity overflow");
        uint64_t requested = static_cast<uint64_t>(raw);
        if (static_cast<long double>(requested) < raw)
        {
            ++requested;
        }
        if (requested < GROUP_SIZE)
        {
            requested = GROUP_SIZE;
        }
        return round_up(requested, GROUP_SIZE);
    }

    static Layout calculate_layout(uint64_t expected_unique_key, uint32_t k_len)
    {
        if (!k_len || k_len > 128 || k_len > 32 * N)
            throw std::invalid_argument("invalid hash table k_len");
        const uint32_t key_bytes = (k_len + 3) / 4;
        const uint64_t capacity = calculate_capacity(expected_unique_key);
        const uint64_t maximum = std::numeric_limits<size_t>::max();
        if (capacity > maximum / key_bytes || capacity > maximum / sizeof(uint32_t) ||
            capacity > maximum - (GROUP_SIZE - 1))
            throw std::length_error("hash table size overflow");

        const uint64_t ctrl_bytes = round_up(capacity + GROUP_SIZE - 1, 2LL * 1024 * 1024);

        const uint64_t kmer_bytes = round_up(capacity * key_bytes, 2ULL * 1024 * 1024);

        const uint64_t count_bytes = round_up(capacity * sizeof(uint32_t), 2ULL * 1024 * 1024);

        if (ctrl_bytes > maximum || kmer_bytes > maximum - ctrl_bytes ||
            count_bytes > maximum - ctrl_bytes - kmer_bytes)
            throw std::length_error("hash table size overflow");
        const uint64_t mmap_bytes = ctrl_bytes + kmer_bytes + count_bytes;

        return Layout{ capacity, key_bytes, ctrl_bytes, kmer_bytes, count_bytes, mmap_bytes };
    }

    static void* map_memory(uint64_t mmap_bytes)
    {
        if (mmap_bytes == 0 || mmap_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) [[unlikely]]
        {
            throw std::bad_alloc();
        }

        const size_t length = static_cast<size_t>(mmap_bytes);

#ifdef TEST_MODE
        std::cout << "attempting to mmap " << length << " bytes with huge pages" << std::endl;
#endif

        void* base = mmap(nullptr,
            length,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1,
            0);

        if (base != MAP_FAILED)
        {
            return base;
        }

#ifdef TEST_MODE
        std::cout << "mmap with huge pages failed, falling back to regular pages" << std::endl;
#endif

        base = mmap(nullptr,
            length,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0);

        if (base == MAP_FAILED)
        {
            throw std::bad_alloc();
        }

        madvise(base, length, MADV_HUGEPAGE);
        return base;
    }

    std::tuple<void*, void*, void*> mmap_hash_map_memory(uint64_t ctrl_bytes, uint64_t kmer_bytes, uint64_t count_bytes)
    {

        constexpr size_t PAGE_SIZE_2MB = 2 * 1024 * 1024;

        std::vector<std::pair<void*, uint64_t>> ctrl_range;
        std::vector<std::pair<void*, uint64_t>> kmer_range;
        std::vector<std::pair<void*, uint64_t>> count_range;

        uint8_t* ctrl_ptr = nullptr;
        char* packed_ptr = nullptr;
        uint32_t* count_ptr = nullptr;
        std::vector<std::thread> threads;
        try
        {
            ctrl_ptr = static_cast<uint8_t*>(map_memory(ctrl_bytes));
            packed_ptr = static_cast<char*>(map_memory(kmer_bytes));
            count_ptr = static_cast<uint32_t*>(map_memory(count_bytes));

            uint64_t ctrl_total_pages = ctrl_bytes / PAGE_SIZE_2MB;
            uint64_t kmer_total_pages = kmer_bytes / PAGE_SIZE_2MB;
            uint64_t count_total_pages = count_bytes / PAGE_SIZE_2MB;

            for (uint32_t i = 0;i < n_thread;i++)
            {
                size_t ctrl_start_page = (i * ctrl_total_pages) / n_thread;
                size_t ctrl_end_page = (i + 1) * ctrl_total_pages / n_thread;
                size_t ctrl_my_pages = ctrl_end_page - ctrl_start_page;           // 该线程负责的 2 MB 页数
                char* ctrl_my_start = (char*)ctrl_ptr + ctrl_start_page * PAGE_SIZE_2MB; // 起始地址
                ctrl_range.emplace_back(ctrl_my_start, ctrl_my_pages);

                size_t kmer_start_page = (i * kmer_total_pages) / n_thread;
                size_t kmer_end_page = (i + 1) * kmer_total_pages / n_thread;
                size_t kmer_my_pages = kmer_end_page - kmer_start_page;           // 该线程负责的 2 MB 页数
                char* kmer_my_start = (char*)packed_ptr + kmer_start_page * PAGE_SIZE_2MB; // 起始地址
                kmer_range.emplace_back(kmer_my_start, kmer_my_pages);

                size_t count_start_page = (i * count_total_pages) / n_thread;
                size_t count_end_page = (i + 1) * count_total_pages / n_thread;
                size_t count_my_pages = count_end_page - count_start_page;           // 该线程负责的 2 MB 页数
                char* count_my_start = (char*)count_ptr + count_start_page * PAGE_SIZE_2MB; // 起始地址
                count_range.emplace_back(count_my_start, count_my_pages);
            }

            threads.reserve(n_thread);
            for (uint32_t i = 0;i < n_thread;i++)
            {
                threads.emplace_back([&, i]()
                    {
                        for (size_t j = 0; j < ctrl_range[i].second; ++j)
                        {
                            volatile char* page_addr = ((volatile char*)ctrl_range[i].first) + j * PAGE_SIZE_2MB;
                            *page_addr = 0;
                        }
                        for (size_t j = 0; j < kmer_range[i].second; ++j)
                        {
                            volatile char* page_addr = ((volatile char*)kmer_range[i].first) + j * PAGE_SIZE_2MB;
                            *page_addr = 0;
                        }
                        for (size_t j = 0; j < count_range[i].second; ++j)
                        {
                            volatile char* page_addr = ((volatile char*)count_range[i].first) + j * PAGE_SIZE_2MB;
                            *page_addr = 0;
                        }
                    });
            }

            for (auto& t : threads)
            {
                t.join();
            }

            return std::make_tuple(ctrl_ptr, packed_ptr, count_ptr);
        }
        catch (...)
        {
            // Range vectors and mappings must outlive already-started workers.
            for (auto& thread : threads)
                if (thread.joinable()) thread.join();
            if (ctrl_ptr) munmap(ctrl_ptr, ctrl_bytes);
            if (packed_ptr) munmap(packed_ptr, kmer_bytes);
            if (count_ptr) munmap(count_ptr, count_bytes);
            throw;
        }
    }

    uint64_t fast_mod(const uint64_t value) const noexcept
    {
        uint64_t q = (uint64_t)((__uint128_t)inv_ * value >> 64);
        uint64_t r = value - q * capacity_;
        r = (r >= capacity_) ? (r - capacity_) : r;
        return r;
        // return value % capacity_;
    }

    static uint8_t fingerprint(uint64_t hash) noexcept
    {
        return static_cast<uint8_t>((hash & 0x7FULL) | 0x80U);
    }

    uint64_t initial_slot(uint64_t hash) const noexcept
    {
        return fast_mod(hash >> 7);
    }

    uint64_t wrap_slot(uint64_t slot) const noexcept
    {
        return slot >= capacity_ ? slot - capacity_ : slot;
    }

    uint64_t next_group_base(uint64_t base) const noexcept
    {
        base += GROUP_SIZE;
        return base >= capacity_ ? base - capacity_ : base;
    }

    std::pair<uint32_t, uint32_t> match_and_empty(uint64_t base, uint8_t fp) const noexcept
    {
        const uint8_t* ctrl = ctrl_ + base;

#if defined(FLAT_HASHMAP_AVX2)
        const __m256i ctrl_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ctrl));
        const __m256i fp_vec = _mm256_set1_epi8(static_cast<char>(fp));
        const __m256i zero_vec = _mm256_setzero_si256();
        const uint32_t match_mask =
            static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(ctrl_vec, fp_vec)));
        const uint32_t empty_mask =
            static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(ctrl_vec, zero_vec)));
        return { match_mask, empty_mask };
#elif defined(FLAT_HASHMAP_SSE)
        const __m128i ctrl_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ctrl));
        const __m128i fp_vec = _mm_set1_epi8(static_cast<char>(fp));
        const __m128i zero_vec = _mm_setzero_si128();
        const uint32_t match_mask =
            static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl_vec, fp_vec)));
        const uint32_t empty_mask =
            static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl_vec, zero_vec)));
        return { match_mask, empty_mask };
#else
        uint32_t match_mask = 0;
        uint32_t empty_mask = 0;
        for (uint64_t i = 0; i < GROUP_SIZE; ++i)
        {
            const uint8_t c = ctrl[i];
            if (c == fp)
            {
                match_mask |= (1U << i);
            }
            if (c == 0)
            {
                empty_mask |= (1U << i);
            }
        }
        return { match_mask, empty_mask };
#endif
    }

    const uint64_t capacity_;
    const uint64_t group_count_;
    const uint64_t inv_;
    uint64_t mmap_size_ = 0;
    uint8_t* ctrl_ = nullptr;
    uint64_t ctrl_size_ = 0;
    const uint32_t key_bytes_;
    char* packed_keys_ = nullptr;
    uint64_t kmer_size_ = 0;
    uint32_t* count_ = nullptr;
    uint64_t count_size_ = 0;
    uint32_t n_thread;
    alignas(64) std::atomic<uint64_t> size_{ 0 };
    std::atomic<bool> sealed_{ false };
};

#endif
