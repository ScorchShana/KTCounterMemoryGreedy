#ifndef HASH_FUNCTION_HEADER
#define HASH_FUNCTION_HEADER

#include <cstdint>

#include "kmer.h"

// 将一个 uint64 值内部混合，消除低位/高位的简单模式
static inline uint64_t mix64(uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

// 最终混合：MurmurHash3 的 64-bit finalizer
static inline uint64_t finalize(uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

uint64_t mix_hash(const uint64_t h1,const uint64_t h2)
{
    const uint64_t MixFactor = 0x9ddfea08eb382d69ULL;
    uint64_t a = (h1 ^ h2) * MixFactor;
    a ^= (a >> 47);
    uint64_t b = (h2 ^ a) * MixFactor;
    b ^= (b >> 47);
    b *= MixFactor;
    return b;
}

/**
 * 计算 1~4 个 uint64 元素的哈希值
 * @param arr  数组指针
 * @param len  元素个数，必须为 1..4
 * @return 64 位哈希值
 */
template <uint32_t N>
uint64_t hash_func(const kmer<N> &arr)
{
    // 初始值：用黄金比例乘以长度，保证不同长度起点不同
    uint64_t h = 0x9e3779b97f4a7c15ULL * N;
    // 再混入一个与长度相关的常数，进一步强化区分
    h += 0x85ebca77c2b2ae63ULL; // prime5 from xxHash64

    for (int i = 0; i < N; ++i)
    {
        uint64_t x = mix64(arr.data[i]); // 1. 元素内部雪崩
        h ^= x;                          // 2. 异或混入
        h = (h << 27) | (h >> 37);       // 3. 旋转 27 位
        h = h * 0x9e3779b97f4a7c15ULL +  // 4. 乘加（prime1, prime2）
            0x165667b19e3779f9ULL;
    }

    return finalize(h);
}

template <uint32_t N>
void double_hash_func(const kmer<N> &arr, uint64_t &h1_out, uint64_t &h2_out)
{
    // 两个不同的初始值（不同黄金比例倍数 + 偏移）
    uint64_t h1 = 0x9e3779b97f4a7c15ULL * N + 0x85ebca77c2b2ae63ULL;
    uint64_t h2 = 0x9e3779b97f4a7c15ULL * (N ^ 0x9e3779b9) + 0x9e3779b97f4a7c15ULL;

    for (int i = 0; i < N; ++i)
    {
        uint64_t x = mix64(arr.data[i]);

        // h1 路径：旋转 + 乘加
        h1 ^= x;
        h1 = (h1 << 27) | (h1 >> 37);
        h1 = h1 * 0x9e3779b97f4a7c15ULL + 0x165667b19e3779f9ULL;

        // h2 路径：使用不同的旋转量和常数，保证独立性
        h2 ^= x;
        h2 = (h2 << 31) | (h2 >> 33); // 不同旋转量（31）
        h2 = h2 * 0xbf58476d1ce4e5b9ULL + 0x94d049bb133111ebULL;
    }

    h1_out = finalize(h1);
    h2_out = finalize(h2) | 1; // 确保步长为奇数（对 2 的幂表）
}

/*
template <uint32_t N>
inline uint64_t hash_func(const kmer<N> &key)
{
    uint64_t h = key.data[0];
    for (uint32_t i = 1; i < N; ++i)
    {
        h += key.data[i];
    }
    return h;
}
*/

/*
template <uint32_t N>
inline uint64_t hash_func(const kmer<N> &key)
{
    uint64_t h = XXH3_64bits(key.data.data(), sizeof(uint64_t) * N);
    return h;
}
*/
/*
inline uint64_t mum64(uint64_t A, uint64_t B)
{
    __uint128_t r = static_cast<__uint128_t>(A) * static_cast<__uint128_t>(B | 1ULL);
    return static_cast<uint64_t>(r ^ (r >> 64));
}

template <uint32_t N>
inline uint64_t hash_func(const kmer<N> &key)
{
    if constexpr (N == 1)
    {
        uint64_t h = key.data[0];
        h += 0x9e3779b97f4a7c15ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        return h ^ (h >> 31);
    }
    if constexpr (N == 2)
    {
        uint64_t h = mum64(key.data[0] ^ 0x9e3779b97f4a7c15ULL, key.data[1] ^ 0x9e3779b97f4a7c15ULL);
        return h;
    }
    else
    {
        uint64_t h1 = mum64(key.data[0], key.data[1]);
        uint64_t h2 = mum64(key.data[2], key.data[3]);
        uint64_t h = mum64(h1 ^ 0x9e3779b97f4a7c15ULL, h2);
        return h;
    }
}
*/
#endif