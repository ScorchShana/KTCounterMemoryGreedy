#include "first_pool_fixture.h"
#include <array>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& p) { p.num_arenas_ = 1; p.cpus_per_node_[0] = p.total_cpus_ = 1; }
    static Arena& arena(ConcurrentMemoryPool& p) { return p.arenas_[0]; }
};
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using Consumer = ConcurrentMemoryPool::FirstConsumer;
using Status = ConcurrentMemoryPool::FirstClaimStatus;
using View = ConcurrentMemoryPool::FirstSegmentView;

static void check_concurrent_publishers()
{
    ConcurrentMemoryPool p(64ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    p.set_spill_service(true);
    auto* descriptors = ConcurrentMemoryPoolTestAccess::arena(p).segments.get();
    std::array<std::future<void>, 4> publishers;
    for (size_t owner = 0; owner < publishers.size(); ++owner) publishers[owner] = std::async(std::launch::async, [&, owner] {
        std::vector<unsigned char> bytes(FIRST_SEGMENT_SIZE);
        for (size_t n = 0; n < 16; ++n) {
            std::memset(bytes.data(), owner * 16 + n, bytes.size());
            while (!cache_full_first(p, 0, bytes.data())) std::this_thread::yield();
        }
    });
    std::array<bool, 64> seen{};
    for (size_t n = 0; n < seen.size();) {
        p.request_first_drain();
        View view;
        if (p.try_claim_first(0, Consumer::Spill, view) != Status::Data) { std::this_thread::yield(); continue; }
        const auto id = view.records[0];
        require(id < seen.size() && !seen[id], "concurrent publication duplicated a segment");
        seen[id] = true;
        for (size_t i = 0; i < view.record_count * kmer_bytes(); ++i)
            require(view.records[i] == id, "consumer read an incomplete Filling segment");
        require(p.complete_first(view), "publisher source completion rejected");
        ++n;
    }
    for (auto& worker : publishers) worker.get();
    require(p.first_empty() && descriptors == ConcurrentMemoryPoolTestAccess::arena(p).segments.get(), "metadata grew or source storage leaked");
}

int main()
{
    try {
        for (const auto [k, records, tail] : std::array<std::array<size_t, 3>, 5>{{
            {1, 2097152, 0}, {31, 262144, 0}, {33, 233016, 8}, {65, 123361, 15}, {128, 65536, 0}}}) {
            k_length = k;
            ConcurrentMemoryPool p(32ULL << 20);
            ConcurrentMemoryPoolTestAccess::single(p);
            p.init_arenas();
            p.set_spill_service(true);
            require(p.first_segment_record_capacity() == records && FIRST_SEGMENT_SIZE - records * kmer_bytes() == tail, "capacity or padding changed");
            std::vector<unsigned char> bytes(FIRST_SEGMENT_SIZE, 0x73);
            View stale;
            for (unsigned round = 0; round < 3; ++round) {
                p.poll_first_progress(0);
                require(cache_full_first(p, 0, bytes.data()), "normal FIRST publication failed");
                p.request_first_drain();
                View view;
                require(p.try_claim_first(0, Consumer::Spill, view) == Status::Data && view.record_count == records, "valid segment not claimable");
                require(!p.complete_first(stale), "old generation completed a new source");
                require(std::memcmp(view.records, bytes.data(), records * kmer_bytes()) == 0, "source content changed");
                require(!cache_full_first(p, 0, bytes.data()), "FIRST grew during reclaim");
                require(p.complete_first(view), "source completion failed");
                require(p.gap_bytes(0) == (32ULL << 20), "complete segment not refunded");
                stale = view;
            }
            p.freeze_first();
            View none;
            require(p.try_claim_first(0, Consumer::Scan, none) == Status::Done, "frozen empty pool did not finish");
            require(!cache_full_first(p, 0, bytes.data()), "frozen FIRST grew");
        }
        k_length = 33;
        check_concurrent_publishers();
        std::cout << "FirstFullSegments tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
