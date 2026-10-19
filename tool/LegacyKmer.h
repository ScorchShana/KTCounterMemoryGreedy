#ifndef LEGACY_KMER_HEADER
#define LEGACY_KMER_HEADER

#include <cstring>
#include <cstdint>
#include <array>
#include <compare>

#include "../src/definition.h"

template <uint32_t N>
struct legacy_kmer
{
    std::array<uint64_t, N> data;
    // 前面的碱基放在data[0]，并且优先填满data[0]，前面碱基放在高位，后面碱基放在低位（即左对齐）。例如 k=31 时，62 位有效数据在 data[0] 的高 62 位，剩余 2 位为 0；k=32 时，64 位有效数据占满 data[0]，data[1] 全为 0；k=33 时，data[0] 全为有效数据，data[1] 的高 2 位为有效数据，剩余 62 位为 0；以此类推。

    legacy_kmer() = default;
    legacy_kmer &operator=(const legacy_kmer<N> &a) = default;
    legacy_kmer(const legacy_kmer<N> &a) = default;

    auto operator<=>(const legacy_kmer &other) const = default;

    bool operator==(const legacy_kmer &other) const noexcept
    {
        return std::memcmp(data.data(), other.data.data(), sizeof(data)) == 0;
    }

    void reset()
    {
        data.fill(0);
    }

    template <uint32_t LEN>
    uint64_t get_prefix()
    {
        constexpr uint32_t shift_bit = 64 - (LEN * 2);
        return data[0] >> shift_bit;
    }

    void shift_left(const uint32_t n_bases)
    {

        const uint32_t shift_bits = n_bases * 2;
        const uint32_t reverse_bits = 8 * sizeof(uint64_t) - shift_bits;

        // 编译器会自动识别 "(a << n) | (b >> (64-n))" 模式并将其编译为 "SHLD" (Double Precision Shift Left) 指令
        if constexpr (N > 1)
        {
            for (uint32_t i = 0; i < N - 1; ++i)
            {
                data[i] = (data[i] << shift_bits) | (data[i + 1] >> reverse_bits);
            }
        }

        // 3. 处理最后一个 uint64_t
        data[N - 1] <<= shift_bits;
    }

    template <uint32_t SHIFT_BASES>
    void shift_left_static()
    {
        static_assert(SHIFT_BASES > 0 && SHIFT_BASES < 32, "Shift amount out of range");

        constexpr uint32_t shift = SHIFT_BASES * 2;
        constexpr uint32_t rev_shift = 64 - shift;

        if constexpr (N > 1)
        {
            for (uint32_t i = 0; i < N - 1; ++i)
            {
                data[i] = (data[i] << shift) | (data[i + 1] >> rev_shift);
            }
        }
        data[N - 1] <<= shift;
    }

    void shift_right(uint32_t n_bases)
    {

        const uint32_t shift_bits = n_bases * 2;
        const uint32_t reverse_bits = 64 - shift_bits;

        // 倒序遍历
        // 必须从最后一个元素开始处理，因为我们需要读取 "前一个" 元素的高位。
        // 如果正序遍历，data[i-1] 会在 data[i] 使用前被修改，导致数据错误。
        if constexpr (N > 1)
        {

            // 使用 int 而不是 uint32_t 避免 i=0 时减 1 导致下溢 (虽然判断了 i>0，但在循环逻辑中 int 更直观)
            for (int i = N - 1; i > 0; --i)
            {
                // 当前块右移 | 前一块的低位左移过来 (补到高位)
                // 编译器会将其优化为 SHRD 指令
                data[i] = (data[i] >> shift_bits) | (data[i - 1] << reverse_bits);
            }
        }

        // 3. 处理第一个元素 (高位补 0)
        data[0] >>= shift_bits;
    }

    template <uint32_t SHIFT_BASES>
    void shift_right_static()
    {
        static_assert(SHIFT_BASES > 0 && SHIFT_BASES < 32, "Shift amount out of range (must be < 64 bits)");

        constexpr uint32_t shift = SHIFT_BASES * 2;
        constexpr uint32_t rev = 64 - shift;

        if constexpr (N > 1)
        {
            for (int i = N - 1; i > 0; --i)
            {
                data[i] = (data[i] >> shift) | (data[i - 1] << rev);
            }
        }
        data[0] >>= shift;
    }
};

// Legacy tool records retain the original integer layout.
template <uint32_t N>
struct ExportRecord { legacy_kmer<N> key; uint32_t count; };
#endif
