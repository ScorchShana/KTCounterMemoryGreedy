#pragma once
#include <cstdint>

class SplitMix64 {
    static thread_local uint64_t x;
public:
    uint64_t operator()() const noexcept {
        x += 0x9e3779b97f4a7c15;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
    static void seed(uint64_t s) noexcept { x = s; }
};
inline thread_local uint64_t SplitMix64::x;