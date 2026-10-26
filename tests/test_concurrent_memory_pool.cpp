#include "first_pool_fixture.h"
#include "../src/ConcurrentMemoryPool.h"
#include <algorithm>
#include <barrier>
#include <future>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>

struct ConcurrentMemoryPoolTestAccess
{
    static Arena& arena(ConcurrentMemoryPool& pool, int i) { return pool.arenas_[i]; }
    static void reclaim(ConcurrentMemoryPool& pool)
    {
        auto& arena = pool.arenas_[0];
        std::lock_guard lock(arena.mutex);
        arena.scan_reclaim_goal_pages.store(1, std::memory_order_release);
        pool.update_pressure(arena);
    }
    static void single(ConcurrentMemoryPool& pool)
    {
        pool.num_arenas_ = 1;
        pool.cpus_per_node_[0] = pool.total_cpus_ = 1;
    }
};
static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
int main()
{
    try {
        k_length = 33;
        ConcurrentMemoryPool pool(64ULL << 20);
        ConcurrentMemoryPoolTestAccess::single(pool);
        pool.init_arenas();
        auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool, 0);
        for (bool reclaim : {true, false}) {
            if (reclaim) ConcurrentMemoryPoolTestAccess::reclaim(pool);
            else { pool.clear_scan_reclaim_goals(); pool.poll_first_progress(0); }
            std::unique_lock held(arena.mutex);
            std::promise<bool> returned;
            std::promise<void> started;
            auto result = returned.get_future();
            std::thread attempt([&] {
                const unsigned char record[9]{0x6d};
                started.set_value();
                returned.set_value(pool.cache_first_record(0, record));
            });
            started.get_future().wait();
            const bool immediate = result.wait_for(std::chrono::milliseconds(reclaim ? 200 : 20)) == std::future_status::ready;
            held.unlock();
            attempt.join();
            require(reclaim ? immediate && !result.get() : !immediate && result.get(),
                "FIRST admission ignored Reclaim filtering or mistook lock contention for exhaustion");
        }
        pool.set_spill_service(true);
        pool.request_first_drain();
        ConcurrentMemoryPool::FirstSegmentView source;
        require(pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Spill, source)
            == ConcurrentMemoryPool::FirstClaimStatus::Data && pool.complete_first(source),
            "blocking FIRST admission fixture did not publish and complete");
        pool.set_spill_service(false);
        // A busy slow-path mutex must never make a try allocation wait.
        for (bool central : {false, true}) {
            std::unique_lock held(arena.mutex);
            std::promise<void*> returned;
            auto result = returned.get_future();
            std::thread attempt([&pool, central, &returned] {
                returned.set_value(central ? pool.try_allocate() : pool.try_allocate_large(BLOCK_SIZE));
            });
            const bool immediate = result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
            held.unlock();
            attempt.join();
            require(immediate && result.get() == nullptr, "try allocation waited on a busy mutex");
        }
        {
            ConcurrentMemoryPool other(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::single(other);
            other.init_arenas();
            std::promise<void> ready, go;
            std::promise<void*> returned;
            auto gate = go.get_future();
            auto result = returned.get_future();
            std::thread switching([&] {
                void* page = pool.allocate();
                pool.deallocate(page); // Leave pages in the old pool's TLS cache.
                ready.set_value();
                gate.wait();
                returned.set_value(other.try_allocate());
            });
            ready.get_future().wait();
            std::unique_lock held(arena.mutex);
            go.set_value();
            const bool immediate = result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
            held.unlock();
            switching.join();
            require(immediate && result.get() == nullptr, "try allocation waited while switching TLS pools");
        }
        const size_t capacity = arena.block_count() * BLOCK_SIZE;
        require(reinterpret_cast<uintptr_t>(arena.start_addr) % HUGE_PAGE_SIZE == 0 &&
            reinterpret_cast<uintptr_t>(arena.end_addr) % HUGE_PAGE_SIZE == 0, "Arena alignment");
        std::vector<void*> tree(1000);
        std::vector<FirstSegmentRef> first;
        std::barrier start(2);
        std::thread worker([&] {
            start.arrive_and_wait();
            for (auto& ptr : tree) { ptr = pool.allocate(); require(ptr, "small allocation failed early"); }
        });
        start.arrive_and_wait();
        std::vector<unsigned char> records(FIRST_SEGMENT_SIZE);
        while (cache_full_first(pool, 0, records.data())) {
            const uint64_t slot = arena.first_begin / FIRST_SEGMENT_SIZE;
            first.push_back({0, slot, arena.segments[slot].tag.load() >> 3});
        }
        worker.join();
        std::set<void*> unique(tree.begin(), tree.end());
        require(unique.size() == tree.size(), "duplicate Tree pages");
        for (const auto& item : first) {
            const auto* segment = &arena.segments[item.segment_index];
            require(segment && segment->data >= static_cast<char*>(arena.start_addr) + arena.first_begin, "missing FIRST segment");
            for (void* ptr : tree)
                require(static_cast<char*>(ptr) + BLOCK_SIZE <= segment->data, "Tree overlaps FIRST");
        }
        require(pool.try_allocate_large(SIZE_MAX) == nullptr, "large overflow accepted");
        require(pool.try_allocate_large(capacity + BLOCK_SIZE) == nullptr, "impossible contiguous request accepted");
        // Consume the last partial refill, including the final single page.
        size_t consumed = tree.size();
        while (pool.try_allocate()) ++consumed;
        require(consumed * BLOCK_SIZE + first.size() * HUGE_PAGE_SIZE == capacity,
            "refill lost pages when fewer than a batch remained");
        require(pool.try_allocate() == nullptr, "exhaustion must be nonblocking");
        pool.deallocate(tree.back());
        require(pool.try_allocate() == tree.back(), "free-list allocation blocked by low gap");
        std::cout << "ConcurrentMemoryPool tests passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
