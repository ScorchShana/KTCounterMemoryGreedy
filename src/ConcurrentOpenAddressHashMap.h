#ifndef CONCURRENT_OPEN_ADDRESS_HASH_MAP_HEADER
#define CONCURRENT_OPEN_ADDRESS_HASH_MAP_HEADER

#include "definition.h"
#include "ConcurrentMemoryPool.h"
#include "kmer.h"
#include "../include/rapidhash.h"

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <limits>
#include <stdexcept>
#include <type_traits>

#ifdef TEST_MODE
#include <utility>
#endif

template<typename CountType>
class ConcurrentOpenAddressHashMap
{
    friend struct KmerTreeAllocationTestAccess;
    static_assert(std::is_same_v<CountType, uint8_t> ||
        std::is_same_v<CountType, uint16_t> || std::is_same_v<CountType, uint32_t>);
    static_assert(std::atomic<CountType>::is_always_lock_free);


    struct alignas(CACHE_LINE_SIZE) SizeCounter
    {
        std::atomic<int64_t> value{ 0 };
    };

    static constexpr size_t align_up(const size_t value, const size_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    constexpr static uint8_t EMPTY = 0x00;
    constexpr static uint8_t INSERTING = 0x01;
    constexpr static double LOAD_FACTOR = 0.8;

    const size_t bytes_;
    const CountType limit_;
    SizeCounter size;
    uint64_t segment_id;
    uint64_t capacity;
    std::atomic<uint8_t>* ctrls;
    unsigned char* keys;
    std::atomic<CountType>* counts;
    std::atomic<bool> sealed;
    std::atomic<bool> building_next;
    std::atomic<ConcurrentOpenAddressHashMap*> next_map;

    inline static thread_local std::size_t map_num_in_block = 0;
    inline static thread_local char* map_mem_block = nullptr;
    inline static thread_local ConcurrentMemoryPool* pool = nullptr;

public:

#ifdef TEST_MODE
    inline static thread_local std::map<uint32_t, uint32_t> segment_histogram;
    inline static thread_local uint64_t segment_probe = 0;
    inline static thread_local uint64_t segment_probe_time = 0;
#endif 

    enum class InsertResult
    {
        INSERTED,
        UPDATED,
        FULL
    };

#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
    enum class DebugEvent
    {
        BEFORE_SEALED_CHECK,
        AFTER_SEALED_OBSERVED,
        AFTER_OPEN_CHECK_BEFORE_CTRL_CAS,
        AFTER_SIZE_RESERVED_BEFORE_PUBLISH,
        BEFORE_COUNT_FETCH_ADD,
        AFTER_COUNT_FETCH_ADD
    };

    using DebugHook = void (*)(DebugEvent,
        const ConcurrentOpenAddressHashMap*,
        const unsigned char*);

    inline static std::atomic<DebugHook> debug_hook{ nullptr };

    static void set_debug_hook(DebugHook hook)
    {
        debug_hook.store(hook, std::memory_order_release);
    }

    static void notify_debug_hook(DebugEvent event,
        const ConcurrentOpenAddressHashMap* map,
        const unsigned char* key)
    {
        DebugHook hook = debug_hook.load(std::memory_order_acquire);
        if (hook != nullptr)
        {
            hook(event, map, key);
        }
    }
#endif

    ConcurrentOpenAddressHashMap(uint64_t capacity, uint32_t limit = count_max, size_t key_bytes = kmer_bytes())
        : bytes_(checked_key_bytes(key_bytes)), limit_(checked_limit(limit)), segment_id(0), capacity(capacity), sealed(false), building_next(false), next_map(nullptr)
    {
        size.value.store(0, std::memory_order_relaxed);
        char* mem_area = reinterpret_cast<char*>(pool->allocate_large(get_mem_size(capacity, bytes_)));
        ctrls = reinterpret_cast<std::atomic<uint8_t>*>(mem_area);
        keys = reinterpret_cast<unsigned char*>(ctrls + capacity);
        counts = reinterpret_cast<std::atomic<CountType>*>(mem_area + counts_offset(capacity, bytes_));
        for (uint64_t i = 0; i < capacity; ++i)
        {
            new (ctrls + i) std::atomic<uint8_t>(EMPTY);
        }
    }

private:
    static size_t checked_key_bytes(size_t bytes)
    {
        if (bytes == 0) throw std::invalid_argument("key width must be positive");
        return bytes;
    }

    static CountType checked_limit(uint32_t limit)
    {
        if (limit == 0 || limit > std::numeric_limits<CountType>::max())
            throw std::invalid_argument("count limit is outside the counter type range");
        return static_cast<CountType>(limit);
    }

public:
    static size_t counts_offset(uint64_t capacity, size_t key_bytes = kmer_bytes()) noexcept {
        return align_up(capacity * (sizeof(std::atomic<uint8_t>) + key_bytes),
            alignof(std::atomic<CountType>));
    }
    static uint64_t get_mem_size(uint64_t capacity, size_t key_bytes = kmer_bytes()) noexcept {
        return counts_offset(capacity, key_bytes) + capacity * sizeof(std::atomic<CountType>);
    }
    static uint64_t hash_key(const unsigned char* key, size_t key_bytes = kmer_bytes()) {
        return rapidhash(key, key_bytes);
    }

    size_t key_bytes() const noexcept { return bytes_; }

    // Extract fingerprint from hash (high 7 bits, most significant bit is 1)
    static uint8_t fingerprint(const uint64_t h)
    {
        uint8_t fp = static_cast<uint8_t>(h >> 57);
        fp |= 0x80; // Set the most significant bit to 1
        return fp;
    }

    uint64_t get_capacity() const
    {
        return capacity;
    }

    ConcurrentOpenAddressHashMap* get_next_map() const
    {
        return next_map.load(std::memory_order_acquire);
    }

    // 遍历当前 map 及所有 next_map segment 中的条目。
    // Visitor 签名为 void(const unsigned char* key, uint32_t count)。
    // 调用方需保证所有写入已结束（如 final drain 阶段），期间不能并发修改。
    template <typename Visitor>
    void for_each_entry(Visitor&& visitor) const
    {
        const ConcurrentOpenAddressHashMap* map = this;
        while (map != nullptr)
        {
            for (uint64_t i = 0; i < map->capacity; ++i)
            {
                const uint8_t ctrl = map->ctrls[i].load(std::memory_order_acquire);
                if ((ctrl & 0x80U) != 0) [[likely]]
                {
                    visitor(map->keys + i * map->bytes_, static_cast<uint32_t>(map->counts[i].load(std::memory_order_relaxed)));
                }
            }
            map = map->next_map.load(std::memory_order_acquire);
        }
    }

#ifdef TEST_MODE
    uint32_t get_segment_count()
    {
        const ConcurrentOpenAddressHashMap* map = this;
        uint32_t cnt = 0;
        while (map != nullptr)
        {
            cnt++;
            map = map->next_map.load(std::memory_order_acquire);
        }
        return cnt;
    }

    void count_to_histogram()
    {
        const ConcurrentOpenAddressHashMap* map = this;
        uint32_t cnt = 0;
        while (map != nullptr)
        {
            cnt++;
            map = map->next_map.load(std::memory_order_acquire);
        }
        segment_histogram[cnt]++;
    }

    std::pair<uint64_t, uint64_t> get_kmer_infos()
    {
        uint64_t singleton_kmers = 0;
        uint64_t unique_kmers = 0;

        const ConcurrentOpenAddressHashMap* map = this;
        while (map != nullptr)
        {
            for (uint64_t i = 0; i < map->capacity; ++i)
            {
                const uint8_t ctrl = map->ctrls[i].load(std::memory_order_acquire);
                if ((ctrl & 0x80U) != 0) [[likely]]
                {
                    ++unique_kmers;
                    singleton_kmers += (map->counts[i].load(std::memory_order_relaxed) > 1) ? 0 : 1;
                }
            }
            map = map->next_map.load(std::memory_order_acquire);
        }
        return std::make_pair(singleton_kmers, unique_kmers);
    }

    static std::map<uint32_t, uint32_t> get_segment_histogram()
    {
        return segment_histogram;
    }
#endif

    static void set_memory_pool(ConcurrentMemoryPool* memory_pool)
    {
        pool = memory_pool;
        map_num_in_block = 0;
        map_mem_block = reinterpret_cast<char*>(pool->allocate());
    }

    static char* get_map_metadata_mem()
    {
        constexpr size_t MAP_SIZE = align_up(sizeof(ConcurrentOpenAddressHashMap), CACHE_LINE_SIZE);
        constexpr size_t MAP_NUM_PER_BLOCK = KMER_BLOCK_SIZE / MAP_SIZE;
        if (map_num_in_block >= MAP_NUM_PER_BLOCK)
        {
            map_mem_block = reinterpret_cast<char*>(pool->allocate());
            map_num_in_block = 0;
        }
        char* mem_area = map_mem_block + map_num_in_block * MAP_SIZE;
        ++map_num_in_block;
        return mem_area;
    }

    InsertResult try_increment(
        uint64_t h,
        const unsigned char* key,
        const uint8_t fp,
        const CountType value,
        uint64_t& local_count)
    {
        const int64_t cur_max_size = static_cast<int64_t>(capacity * LOAD_FACTOR);
        const uint64_t mod = capacity - 1;
        uint64_t index = h & mod;
        uint64_t offset = 0;

        for (;;) {
            uint8_t ctrl = ctrls[index].load(std::memory_order_acquire);
            if (ctrl == EMPTY)
            {
                // Try to insert
#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
                notify_debug_hook(DebugEvent::BEFORE_SEALED_CHECK, this, key);
#endif
                if (sealed.load(std::memory_order_acquire))
                {
#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
                    notify_debug_hook(DebugEvent::AFTER_SEALED_OBSERVED, this, key);
#endif
                    ctrl = ctrls[index].load(std::memory_order_acquire);
                    if (ctrl == EMPTY)
                    {
                        return InsertResult::FULL; // Indicate that the map is full
                    }
                    else
                    {
                        cpu_relax();
                        continue; // Another thread inserted, retry
                    }
                }

#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
                notify_debug_hook(DebugEvent::AFTER_OPEN_CHECK_BEFORE_CTRL_CAS, this, key);
#endif

                uint8_t expected = EMPTY;
                if (ctrls[index].compare_exchange_strong(expected,
                    INSERTING,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed))
                {

                    if (sealed.load(std::memory_order_acquire)) [[unlikely]]
                    {
                        ctrls[index].store(EMPTY, std::memory_order_release); // Revert control to EMPTY
                        return InsertResult::FULL; // Indicate that the map is full
                    }
                    // Successfully marked as inserting
                    const uint64_t old_size = size.value.fetch_add(1, std::memory_order_acq_rel);

                    if (old_size >= static_cast<uint64_t>(cur_max_size)) [[unlikely]]
                    {
                        size.value.fetch_sub(1, std::memory_order_acq_rel); // Revert size increment
                        ctrls[index].store(EMPTY, std::memory_order_release); // Revert control to EMPTY
                        sealed.store(true, std::memory_order_release);
                        return InsertResult::FULL; // Indicate that the map is full
                    }
                    else
                    {
#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
                        notify_debug_hook(DebugEvent::AFTER_SIZE_RESERVED_BEFORE_PUBLISH, this, key);
#endif
                        std::memcpy(keys + index * bytes_, key, bytes_);
                        const CountType corrected_value = std::min(value, limit_);
                        new (counts + index) std::atomic<CountType>(corrected_value);
                        ctrls[index].store(fp, std::memory_order_release); // Set the fingerprint
                        ++local_count;
                        return InsertResult::INSERTED; // Return true if we need to build next map
                    }
                }
                else
                {
                    cpu_relax(); // Another thread is inserting, pause and retry
                    continue;
                }
            }
            else if (ctrl == fp)
            {
                if (::__memcmpeq(keys + index * bytes_, key, bytes_) == 0) [[likely]]
                {
                    const CountType cur = counts[index].load(std::memory_order_relaxed);
                    if (cur < limit_)
                    {
#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
                        notify_debug_hook(DebugEvent::BEFORE_COUNT_FETCH_ADD, this, key);
#endif
                        const CountType prev = counts[index].fetch_add(value, std::memory_order_relaxed);
#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
                        notify_debug_hook(DebugEvent::AFTER_COUNT_FETCH_ADD, this, key);
#endif
                        if (value > limit_ - std::min(prev, limit_)) [[unlikely]]
                        {
                            counts[index].store(limit_, std::memory_order_relaxed);
                        }
                    }
                    return InsertResult::UPDATED;
                }
            }
            else if (ctrl == INSERTING)
            {
                cpu_relax(); // Another thread is inserting, pause and retry
                continue;
            }

            // probe to next index
            ++offset;
            offset &= mod;
            index = (index + offset) & mod;
        }
    }

    void increment(const unsigned char* key, const CountType value, uint64_t& local_count) {
        const uint64_t h = hash_key(key, bytes_);
        const uint8_t fp = fingerprint(h);

        ConcurrentOpenAddressHashMap* map_ptr = this;

#ifdef TEST_MODE
        ++segment_probe_time;
#endif

        for (;;)
        {
#ifdef TEST_MODE
            ++segment_probe;
#endif
            InsertResult res = map_ptr->try_increment(h, key, fp, value, local_count);
            if (res == InsertResult::FULL)
            {
                map_ptr = map_ptr->ensure_next_map();
            }
            else
            {
                return;
            }
        }
    }

#ifdef CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING
    const std::atomic<CountType>* debug_count_storage() const noexcept { return counts; }

    template <typename Visitor>
    void debug_visit_entries(Visitor&& visitor) const
    {
        const ConcurrentOpenAddressHashMap* map = this;
        uint64_t segment_index = 0;
        while (map != nullptr)
        {
            for (uint64_t i = 0; i < map->capacity; ++i)
            {
                const uint8_t ctrl = map->ctrls[i].load(std::memory_order_acquire);
                if ((ctrl & 0x80U) != 0)
                {
                    visitor(segment_index,
                        map->keys + i * map->bytes_,
                        static_cast<uint32_t>(map->counts[i].load(std::memory_order_relaxed)));
                }
            }
            map = map->next_map.load(std::memory_order_acquire);
            ++segment_index;
        }
    }

    template <typename Visitor>
    void debug_visit_segments(Visitor&& visitor) const
    {
        const ConcurrentOpenAddressHashMap* map = this;
        uint64_t segment_index = 0;
        while (map != nullptr)
        {
            visitor(segment_index,
                map->size.value.load(std::memory_order_acquire),
                map->sealed.load(std::memory_order_acquire),
                map->capacity);
            map = map->next_map.load(std::memory_order_acquire);
            ++segment_index;
        }
    }

    uint64_t debug_inserting_slot_count() const
    {
        uint64_t inserting_count = 0;
        const ConcurrentOpenAddressHashMap* map = this;
        while (map != nullptr)
        {
            for (uint64_t i = 0; i < map->capacity; ++i)
            {
                if (map->ctrls[i].load(std::memory_order_acquire) == INSERTING)
                {
                    ++inserting_count;
                }
            }
            map = map->next_map.load(std::memory_order_acquire);
        }
        return inserting_count;
    }
#endif

    ConcurrentOpenAddressHashMap* ensure_next_map()
    {
        SpinBackoff<> backoff;
        for (;;) {
            pool->run_state().check();
            auto* next = next_map.load(std::memory_order_acquire);
            if (next) return next;
            bool expected = false;
            if (building_next.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
                try {
                    next = next_map.load(std::memory_order_acquire);
                    if (!next) {
                        const uint64_t next_id = segment_id + 1;
                        const uint64_t factor = next_id > SEGMENT_WATERMARK ? 4 : 2;
                        const uint64_t next_capacity = capacity > concurrent_hash_map_max_capacity / factor
                            ? concurrent_hash_map_max_capacity : capacity * factor;
                        void* memory = get_map_metadata_mem();
                        next = new (memory) ConcurrentOpenAddressHashMap(next_capacity, limit_, bytes_);
                        next->segment_id = next_id;
                        next_map.store(next, std::memory_order_release);
                    }
                    building_next.store(false, std::memory_order_release);
                    return next;
                } catch (...) {
                    building_next.store(false, std::memory_order_release);
                    throw;
                }
            }
            backoff.backoff();
        }
    }

};

#endif
