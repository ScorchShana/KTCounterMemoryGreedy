#ifndef EXPORT_RECORD_RADIX_SORT_HEADER
#define EXPORT_RECORD_RADIX_SORT_HEADER

#include "../src/kmer.h"
#include "../src/definition.h"

#include <cstring>
#include <cstdint>
#include <array>
#include <type_traits>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

constexpr uint32_t RADIX_BITS = 8;
constexpr uint32_t HISTOGRAM_SIZE = 1U << RADIX_BITS;
constexpr uint32_t RADIX_MASK = HISTOGRAM_SIZE - 1;
constexpr uint32_t PARALLEL_NUM = 4;
constexpr uint32_t HISTOGRAM_NUM = PARALLEL_NUM;
constexpr uint32_t IGNORE_PREFIX_BITS = 2 * (ROOT_BASES + (MAX_DEPTH - 1) * NODE_BASES);
constexpr uint32_t UINT64_BITS = 64;

template <uint32_t N>
static uint32_t extract_histogram_pos(const kmer<N>& k_mer, const uint32_t& kmer_index,
    const uint32_t& bit_offset)
{
    // no across uint64_t boundary
    const uint32_t shift_bits = k_mer.data[kmer_index] >> bit_offset;
    return shift_bits & RADIX_MASK;
}

template <uint32_t N, typename UINT_TYPE = uint32_t>
static void generate_histogram(ExportRecord<N>* records, const uint64_t length,
    const uint32_t end_pos, std::array<UINT_TYPE*, HISTOGRAM_NUM>& histograms)
{
    const uint32_t kmer_index = end_pos / UINT64_BITS;
    const uint32_t bit_offset = UINT64_BITS - 1 - (end_pos % UINT64_BITS);

    for (uint32_t i = 0; i < HISTOGRAM_NUM; i++)
    {
        std::memset(histograms[i], 0, HISTOGRAM_SIZE * sizeof(UINT_TYPE));
    }

    for (uint64_t i = 0; i + 3 < length; i += 4)
    {
#pragma GCC unroll 4
        for (uint32_t j = 0; j < HISTOGRAM_NUM; j++)
        {
            const uint32_t histogram_pos = extract_histogram_pos(records[i + j].key, kmer_index, bit_offset);
            histograms[j][histogram_pos]++;
        }
    }

    for (uint64_t i = (length / 4) * 4; i < length; i++)
    {
        const uint32_t histogram_pos = extract_histogram_pos(records[i].key, kmer_index, bit_offset);
        histograms[0][histogram_pos]++;
    }


#if defined(__AVX2__)
    if constexpr (std::is_same_v<UINT_TYPE, uint32_t>)
    {
        for (uint32_t i = 0; i < HISTOGRAM_SIZE; i += 16)
        {
            __m256i va0 = _mm256_loadu_si256((__m256i*) & histograms[0][i]);
            __m256i vb0 = _mm256_loadu_si256((__m256i*) & histograms[1][i]);
            __m256i vc0 = _mm256_loadu_si256((__m256i*) & histograms[2][i]);
            __m256i vd0 = _mm256_loadu_si256((__m256i*) & histograms[3][i]);

            __m256i va1 = _mm256_loadu_si256((__m256i*) & histograms[0][i + 8]);
            __m256i vb1 = _mm256_loadu_si256((__m256i*) & histograms[1][i + 8]);
            __m256i vc1 = _mm256_loadu_si256((__m256i*) & histograms[2][i + 8]);
            __m256i vd1 = _mm256_loadu_si256((__m256i*) & histograms[3][i + 8]);

            // 两组并行树形加法
            __m256i vs0_0 = _mm256_add_epi32(va0, vb0);
            __m256i vs0_1 = _mm256_add_epi32(vc0, vd0);
            __m256i vres0 = _mm256_add_epi32(vs0_0, vs0_1);

            __m256i vs1_0 = _mm256_add_epi32(va1, vb1);
            __m256i vs1_1 = _mm256_add_epi32(vc1, vd1);
            __m256i vres1 = _mm256_add_epi32(vs1_0, vs1_1);

            _mm256_storeu_si256((__m256i*) & histograms[1][i], vres0);
            _mm256_storeu_si256((__m256i*) & histograms[1][i + 8], vres1);
        }
    }
    else
    {
        for (uint32_t i = 0; i < HISTOGRAM_SIZE; i += 8)
        {
            __m256i va0 = _mm256_loadu_si256((__m256i*) & histograms[0][i]);
            __m256i vb0 = _mm256_loadu_si256((__m256i*) & histograms[1][i]);
            __m256i vc0 = _mm256_loadu_si256((__m256i*) & histograms[2][i]);
            __m256i vd0 = _mm256_loadu_si256((__m256i*) & histograms[3][i]);

            __m256i va1 = _mm256_loadu_si256((__m256i*) & histograms[0][i + 4]);
            __m256i vb1 = _mm256_loadu_si256((__m256i*) & histograms[1][i + 4]);
            __m256i vc1 = _mm256_loadu_si256((__m256i*) & histograms[2][i + 4]);
            __m256i vd1 = _mm256_loadu_si256((__m256i*) & histograms[3][i + 4]);

            // 两组并行树形加法
            __m256i vs0_0 = _mm256_add_epi64(va0, vb0);
            __m256i vs0_1 = _mm256_add_epi64(vc0, vd0);
            __m256i vres0 = _mm256_add_epi64(vs0_0, vs0_1);

            __m256i vs1_0 = _mm256_add_epi64(va1, vb1);
            __m256i vs1_1 = _mm256_add_epi64(vc1, vd1);
            __m256i vres1 = _mm256_add_epi64(vs1_0, vs1_1);

            _mm256_storeu_si256((__m256i*) & histograms[1][i], vres0);
            _mm256_storeu_si256((__m256i*) & histograms[1][i + 4], vres1);
        }
    }
#elif defined(__SSE2__)
    if constexpr (std::is_same_v<UINT_TYPE, uint32_t>)
    {
        for (uint32_t i = 0; i < HISTOGRAM_SIZE; i += 8)
        {
            __m128i va0 = _mm_loadu_si128((__m128i*) & histograms[0][i]);
            __m128i vb0 = _mm_loadu_si128((__m128i*) & histograms[1][i]);
            __m128i vc0 = _mm_loadu_si128((__m128i*) & histograms[2][i]);
            __m128i vd0 = _mm_loadu_si128((__m128i*) & histograms[3][i]);

            __m128i va1 = _mm_loadu_si128((__m128i*) & histograms[0][i + 4]);
            __m128i vb1 = _mm_loadu_si128((__m128i*) & histograms[1][i + 4]);
            __m128i vc1 = _mm_loadu_si128((__m128i*) & histograms[2][i + 4]);
            __m128i vd1 = _mm_loadu_si128((__m128i*) & histograms[3][i + 4]);

            // 两组并行树形加法
            __m128i vs0_0 = _mm_add_epi32(va0, vb0);
            __m128i vs0_1 = _mm_add_epi32(vc0, vd0);
            __m128i vres0 = _mm_add_epi32(vs0_0, vs0_1);

            __m128i vs1_0 = _mm_add_epi32(va1, vb1);
            __m128i vs1_1 = _mm_add_epi32(vc1, vd1);
            __m128i vres1 = _mm_add_epi32(vs1_0, vs1_1);

            _mm_storeu_si128((__m128i*) & histograms[1][i], vres0);
            _mm_storeu_si128((__m128i*) & histograms[1][i + 4], vres1);
        }
    }
    else
    {
        for (uint32_t i = 0; i < HISTOGRAM_SIZE; i += 4)
        {
            __m128i va0 = _mm_loadu_si128((__m128i*) & histograms[0][i]);
            __m128i vb0 = _mm_loadu_si128((__m128i*) & histograms[1][i]);
            __m128i vc0 = _mm_loadu_si128((__m128i*) & histograms[2][i]);
            __m128i vd0 = _mm_loadu_si128((__m128i*) & histograms[3][i]);

            __m128i va1 = _mm_loadu_si128((__m128i*) & histograms[0][i + 2]);
            __m128i vb1 = _mm_loadu_si128((__m128i*) & histograms[1][i + 2]);
            __m128i vc1 = _mm_loadu_si128((__m128i*) & histograms[2][i + 2]);
            __m128i vd1 = _mm_loadu_si128((__m128i*) & histograms[3][i + 2]);

            // 两组并行树形加法
            __m128i vs0_0 = _mm_add_epi64(va0, vb0);
            __m128i vs0_1 = _mm_add_epi64(vc0, vd0);
            __m128i vres0 = _mm_add_epi64(vs0_0, vs0_1);

            __m128i vs1_0 = _mm_add_epi64(va1, vb1);
            __m128i vs1_1 = _mm_add_epi64(vc1, vd1);
            __m128i vres1 = _mm_add_epi64(vs1_0, vs1_1);

            _mm_storeu_si128((__m128i*) & histograms[1][i], vres0);
            _mm_storeu_si128((__m128i*) & histograms[1][i + 2], vres1);
        }
    }
#else

    for (uint32_t j = 0; j < HISTOGRAM_SIZE; j++)
    {
        const UINT_TYPE sum1 = histograms[0][j] + histograms[1][j];
        const UINT_TYPE sum2 = histograms[2][j] + histograms[3][j];
        histograms[1][j] = sum1 + sum2;
    }

#endif

    UINT_TYPE sum = 0;
    for (uint32_t j = 0; j < HISTOGRAM_SIZE; j++)
    {
        histograms[0][j] = sum;
        sum += histograms[1][j];
    }

}

template<uint32_t N, typename UINT_TYPE = uint32_t>
static void scatter_records(ExportRecord<N>* records, ExportRecord<N>* temp_records,
    const uint64_t length, const uint32_t end_pos, std::array<UINT_TYPE*, HISTOGRAM_NUM>& histograms)
{
    const uint32_t kmer_index = end_pos / UINT64_BITS;
    const uint32_t bit_offset = UINT64_BITS - 1 - (end_pos % UINT64_BITS);

    for (uint64_t i = 0; i + 3 < length; i += 4)
    {
#pragma GCC unroll 4
        for (uint32_t j = 0; j < PARALLEL_NUM; j++)
        {
            const uint32_t histogram_pos = extract_histogram_pos(records[i + j].key, kmer_index, bit_offset);
            const UINT_TYPE count = histograms[0][histogram_pos];
            temp_records[count].key = records[i + j].key;
            temp_records[count].count = records[i + j].count;
            histograms[0][histogram_pos]++;
        }
    }

    for (uint64_t i = (length / 4) * 4; i < length; i++)
    {
        const uint32_t histogram_pos = extract_histogram_pos(records[i].key, kmer_index, bit_offset);
        const UINT_TYPE count = histograms[0][histogram_pos]++;
        temp_records[count].key = records[i].key;
        temp_records[count].count = records[i].count;
    }
}

template <uint32_t N, typename UINT_TYPE = uint32_t>
static ExportRecord<N>* radix_sort(ExportRecord<N>* records, ExportRecord<N>* temp_records, const uint64_t length, uint32_t total_bits)
{


    std::array<UINT_TYPE*, HISTOGRAM_NUM> histograms;
    alignas(64) UINT_TYPE histogram_storage[HISTOGRAM_NUM * HISTOGRAM_SIZE];

    for (uint32_t i = 0; i < HISTOGRAM_NUM; i++)
    {
        histograms[i] = &histogram_storage[i * HISTOGRAM_SIZE];
    }

    uint32_t current_different_bits = total_bits - IGNORE_PREFIX_BITS;
    uint32_t end_pos = total_bits - 1;

    while (current_different_bits > 0)
    {
        uint32_t bits_to_process = std::min(end_pos + 1 - end_pos / UINT64_BITS * UINT64_BITS, RADIX_BITS);
        bits_to_process = std::min(bits_to_process, current_different_bits);
        current_different_bits -= bits_to_process;

        generate_histogram<N, UINT_TYPE>(records, length, end_pos, histograms);
        scatter_records<N, UINT_TYPE>(records, temp_records, length, end_pos, histograms);

        end_pos -= bits_to_process;
        std::swap(records, temp_records);
    }

    return records;
}

template <uint32_t N>
ExportRecord<N>* export_record_radix_sort(ExportRecord<N>* records, ExportRecord<N>* temp_records, const uint64_t length, const uint32_t k_len)
{
    const uint32_t total_bits = k_len * 2;
    if (length > UINT32_MAX)
    {
        return radix_sort<N, uint64_t>(records, temp_records, length, total_bits);
    }
    else
    {
        return radix_sort<N, uint32_t>(records, temp_records, length, total_bits);

    }
}

#endif // EXPORT_RECORD_RADIX_SORT_HEADER