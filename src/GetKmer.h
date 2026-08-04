#ifndef GET_KMER_HEADER
#define GET_KMER_HEADER

#include <cstdint>

#include "kmer.h"

template <uint32_t N>
class GetKmer
{

public:
    kmer<N> seq_kmer;
    kmer<N> rev_kmer;
    kmer<N> canonical_kmer;

    uint64_t rev_insert_shift; // 反向 k-mer 插入时需要左移的位数

    explicit GetKmer(const uint32_t in_k) : k(in_k), kmer_last_index((k - 1) / 32), have_read(0), remainder(k % BASES_PER_U64T)
    {

        for (int i = 0; i < 256; i++)
        {
            char_to_2bit[i] = 255;
        }

        char_to_2bit['A'] = 0;
        char_to_2bit['a'] = 0;
        char_to_2bit['C'] = 1;
        char_to_2bit['c'] = 1;
        char_to_2bit['G'] = 2;
        char_to_2bit['g'] = 2;
        char_to_2bit['T'] = 3;
        char_to_2bit['t'] = 3;
        char_to_2bit['U'] = 3;
        char_to_2bit['u'] = 3;

        if (remainder != 0ULL)
        {
            back_mask = ((1ULL << (2 * remainder)) - 1ULL) << (64 - 2 * remainder);
            rev_insert_shift = 2 * (BASES_PER_U64T - remainder);
        }
        else
        {
            back_mask = 0xffffffffffffffffULL;
            rev_insert_shift = 0;
        }
        clear();
    }

    inline void clear()
    {
        seq_kmer.reset();
        rev_kmer.reset();
        have_read = 0;
    }

    bool get_next_one(const unsigned char base)
    {

        have_read++;

        if (char_to_2bit[base] > 3)
        {
            clear();
            return false;
        }

        const uint64_t base_2bit = static_cast<uint64_t>(char_to_2bit[base]);

        seq_kmer.template shift_right_static<1>();
        seq_kmer.data[0] |= (base_2bit << (BASES_PER_U64T * 2 - 2));
        seq_kmer.data[kmer_last_index] &= back_mask;

        rev_kmer.template shift_left_static<1>();

        rev_kmer.data[kmer_last_index] |= (base_2bit ^ 0b11) << rev_insert_shift;

        canonicalize();

        return have_read >= k;
    }

    // 批量插入 count 个连续碱基
#if defined(__AVX2__)
    uint32_t batch_insert(uint64_t packed_codes, uint32_t count, kmer<N>* output) noexcept
    {

        uint32_t out_cnt = 0;

        for (uint32_t kk = 0; kk < count; ++kk)
        {
            // Parser 里 packed = (packed << 2) | code，所以最早的 base 在高位，最新的 base 在低位。
            // 这里必须按“旧 -> 新”的顺序取出。
            const uint64_t code =
                (packed_codes >> (2 * (count - 1 - kk))) & 0x3ULL;

            seq_kmer.template shift_right_static<1>();
            seq_kmer.data[0] |= code << 62;
            seq_kmer.data[kmer_last_index] &= back_mask;

            rev_kmer.template shift_left_static<1>();
            rev_kmer.data[kmer_last_index] |= (code ^ 0b11ULL) << rev_insert_shift;

            if (++have_read >= k) [[likely]]
            {
                uint64_t mask = -(seq_kmer < rev_kmer);
                kmer<N>& out = output[out_cnt++];

                if constexpr (N == 1)
                {
                    out.data[0] = (seq_kmer.data[0] & mask) | (rev_kmer.data[0] & ~mask);
                }
                else if constexpr (N == 2)
                {
                    out.data[0] = (seq_kmer.data[0] & mask) | (rev_kmer.data[0] & ~mask);
                    out.data[1] = (seq_kmer.data[1] & mask) | (rev_kmer.data[1] & ~mask);
                }
                else
                {
                    for (uint32_t j = 0; j < N; ++j)
                    {
                        out.data[j] = (seq_kmer.data[j] & mask) | (rev_kmer.data[j] & ~mask);
                    }
                }
            }
        }

        return out_cnt;
    }

#elif defined(__SSE4_2__)
    uint32_t batch_insert(uint32_t packed_codes, uint32_t count, kmer<N>* output) noexcept
    {

        uint32_t out_cnt = 0;

        for (uint32_t kk = 0; kk < count; ++kk)
        {
            // Parser 里 packed = (packed << 2) | code，所以最早的 base 在高位，最新的 base 在低位。
            // 这里必须按“旧 -> 新”的顺序取出。
            const uint64_t code =
                (packed_codes >> (2 * (count - 1 - kk))) & 0x3ULL;

            seq_kmer.template shift_right_static<1>();
            seq_kmer.data[0] |= code << 62;
            seq_kmer.data[kmer_last_index] &= back_mask;

            rev_kmer.template shift_left_static<1>();
            rev_kmer.data[kmer_last_index] |= (code ^ 0b11ULL) << rev_insert_shift;

            if (++have_read >= k) [[likely]]
            {
                uint64_t mask = -(seq_kmer < rev_kmer);
                kmer<N>& out = output[out_cnt++];

                if constexpr (N == 1)
                {
                    out.data[0] = (seq_kmer.data[0] & mask) | (rev_kmer.data[0] & ~mask);
                }
                else if constexpr (N == 2)
                {
                    out.data[0] = (seq_kmer.data[0] & mask) | (rev_kmer.data[0] & ~mask);
                    out.data[1] = (seq_kmer.data[1] & mask) | (rev_kmer.data[1] & ~mask);
                }
                else
                {
                    for (uint32_t j = 0; j < N; ++j)
                    {
                        out.data[j] = (seq_kmer.data[j] & mask) | (rev_kmer.data[j] & ~mask);
                    }
                }
            }
        }

        return out_cnt;
    }
#else
#endif

    void canonicalize() noexcept
    {
        uint64_t mask = -(seq_kmer < rev_kmer); // 无分支掩码

        if constexpr (N == 1)
        {
            canonical_kmer.data[0] = (seq_kmer.data[0] & mask) | (rev_kmer.data[0] & (~mask));
        }
        else if constexpr (N == 2)
        {
            canonical_kmer.data[0] = (seq_kmer.data[0] & mask) | (rev_kmer.data[0] & (~mask));
            canonical_kmer.data[1] = (seq_kmer.data[1] & mask) | (rev_kmer.data[1] & (~mask));
        }
        else if constexpr (N == 4)
        {
            canonical_kmer.data[0] = (seq_kmer.data[0] & mask) | (rev_kmer.data[0] & (~mask));
            canonical_kmer.data[1] = (seq_kmer.data[1] & mask) | (rev_kmer.data[1] & (~mask));
            canonical_kmer.data[2] = (seq_kmer.data[2] & mask) | (rev_kmer.data[2] & (~mask));
            canonical_kmer.data[3] = (seq_kmer.data[3] & mask) | (rev_kmer.data[3] & (~mask));
        }
        else {
            for (uint32_t i = 0; i < N; ++i)
            {
                canonical_kmer.data[i] = (seq_kmer.data[i] & mask) | (rev_kmer.data[i] & (~mask));
            }
        }
    }

private:
    const uint32_t k;
    const uint32_t kmer_last_index;
    uint64_t back_mask;
    uint64_t have_read = 0;
    const uint64_t remainder; // 最后一个 uint64_t 中剩余的碱基数（如果 k 不是 BASES_PER_U64T 的倍数）
    std::array<uint8_t, 256> char_to_2bit;
};

#endif