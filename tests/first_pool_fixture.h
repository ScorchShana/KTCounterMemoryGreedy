#ifndef FIRST_POOL_FIXTURE_HEADER
#define FIRST_POOL_FIXTURE_HEADER
#include "../src/ConcurrentMemoryPool.h"

// Drive the production TLS interface; no second full-segment pool API in tests.
inline bool cache_full_first(ConcurrentMemoryPool& pool, int arena, const unsigned char* records)
{
    const size_t width = kmer_bytes();
    for (size_t i = 0; i < pool.first_segment_record_capacity(); ++i)
        if (!pool.cache_first_record(arena, records + i * width)) return false;
    return true;
}
#endif
