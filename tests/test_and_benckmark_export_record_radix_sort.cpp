// test_radix_sort.cpp
// 测试与基准测试 export_record_radix_sort 函数，并与 std::sort 比较
// 编译: g++ -std=c++20 -O3 -mavx2 -I../src test_radix_sort.cpp -o test_radix_sort

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <unordered_set>
#include <set>

#include "../sort/ExportRecordRadixSort.h"


enum Distribution : uint8_t { SEQUENTIAL, RANDOM, REVERSED };

// -------------------- 辅助函数：限制 kmer 长度（保留高 2*k 位，低位清零，保证前20位为0） --------------------
template <uint32_t N>
void limit_kmer(kmer<N>& km, uint32_t k) {
    constexpr uint32_t total_bits = N * 64;
    const uint32_t keep_bits = 2 * k;          // 需要保留的位数
    if (keep_bits >= total_bits) return;       // 全部保留

    uint32_t remaining = keep_bits;
    for (uint32_t i = 0; i < N; ++i) {
        if (remaining >= 64) {
            remaining -= 64;                   // 整个 uint64_t 全部保留
            continue;
        }
        else {
            // 保留本单元的高 remaining 位，低位清零
            if (remaining == 0) {
                km.data[i] = 0;
            }
            else {
                km.data[i] &= (~0ULL) << (64 - remaining);
            }
            // 后面的单元全部清零
            for (uint32_t j = i + 1; j < N; ++j) {
                km.data[j] = 0;
            }
            break;
        }
    }
}

// -------------------- 生成唯一测试记录 --------------------
template <uint32_t N>
void generate_unique_records(std::vector<ExportRecord<N>>& records,
    size_t num,
    uint32_t k) {
    records.clear();
    records.reserve(num);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, ~0ULL);

    // 用 std::set 去重（kmer 支持 operator<）
    std::set<kmer<N>> unique_set;

    // 生成足够多的随机 kmer 直至达到数量 num
    while (unique_set.size() < num) {
        kmer<N> km;
        for (uint32_t i = 0; i < N; ++i) {
            km.data[i] = dist(gen);
        }
        km.data[0] &= (1ULL << 44) - 1ULL;
        limit_kmer(km, k);          // 限制长度为 k
        unique_set.insert(km);
    }

    // 转换为 ExportRecord 数组
    for (const auto& km : unique_set) {
        ExportRecord<N> rec;
        rec.key = km;
        rec.count = static_cast<uint32_t>(records.size() % 65536); // 任意计数
        records.push_back(rec);
    }
    // 注意：set 已有序，但后续我们会打乱或反转，所以初始顺序不影响
}

// -------------------- 验证两个 vector 是否相等（按 key 和 count） --------------------
template <uint32_t N>
bool verify(const std::vector<ExportRecord<N>>& a,
    const std::vector<ExportRecord<N>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].key != b[i].key || a[i].count != b[i].count) {
            return false;
        }
    }
    return true;
}

// -------------------- 运行一次测试（给定 N, k, 记录数） --------------------
template <uint32_t N>
void run_test(uint32_t k, size_t num_records) {
    // 检查 k 是否合法（2*k <= N*64）
    if (2 * k > N * 64) {
        std::cerr << "Skipping N=" << N << ", k=" << k << " (too large)\n";
        return;
    }

    std::cout << "\n=== N = " << N << ", k = " << k
        << ", records = " << num_records << " ===\n";

    // 生成唯一的记录（已有序，由 set 保证升序）
    std::vector<ExportRecord<N>> base;
    generate_unique_records(base, num_records, k);

    // 三种分布：顺序（已有序）、反转、随机
    std::vector<Distribution> dists = { SEQUENTIAL,RANDOM, REVERSED };
    const char* dist_names[] = { "Sequential",  "Random" ,"Reversed" };

    for (auto dist : dists) {
        // 拷贝一份
        auto rec_std = base;

        // 根据分布调整顺序
        switch (dist) {
        case SEQUENTIAL:
            // 已有序，无需操作
            break;
        case REVERSED:
            std::reverse(rec_std.begin(), rec_std.end());
            //std::reverse(rec_radix.begin(), rec_radix.end());
            break;
        case RANDOM: {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(rec_std.begin(), rec_std.end(), g);
            //std::shuffle(rec_radix.begin(), rec_radix.end(), g);
            break;
        }
        default:
            break;
        }

        // 临时缓冲区用于基数排序
        std::vector<ExportRecord<N>> temp(num_records);
        auto rec_radix = rec_std; // 基数排序的输入与 std::sort 一样

        // ---------- std::sort ----------
        auto start = std::chrono::high_resolution_clock::now();
        std::sort(rec_std.begin(), rec_std.end(),
            [](const ExportRecord<N>& a, const ExportRecord<N>& b) {
                return a.key < b.key;
            });
        auto end = std::chrono::high_resolution_clock::now();
        auto time_std = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // ---------- radix sort ----------
        start = std::chrono::high_resolution_clock::now();
        auto res_ptr = export_record_radix_sort(rec_radix.data(), temp.data(), num_records, k);
        end = std::chrono::high_resolution_clock::now();
        auto time_radix = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // 验证结果
        bool ok = false;
        if (res_ptr == rec_radix.data())
        {
            // 结果已经在 rec_radix 中
            ok = verify(rec_std, rec_radix);
            std::cout << "Radix sort result is in original buffer.\n";
        }
        else
        {
            ok = verify(rec_std, temp);
            std::cout << "Radix sort result is in temp buffer.\n";
        }

        std::cout << "Distribution: " << dist_names[static_cast<int>(dist)]
            << " | std::sort: " << std::setw(9) << time_std << " ns"
            << " | radix_sort: " << std::setw(9) << time_radix << " ns"
            << " | speedup: " << std::fixed << std::setprecision(2)
            << (time_radix > 0 ? static_cast<double>(time_std) / time_radix : 0.0)
            << "x | " << (ok ? "PASS" : "FAIL") << std::endl;

        if (!ok)
        {
            std::cerr << "Test failed for distribution: " << dist_names[static_cast<int>(dist)] << std::endl;
            exit(-1);
        }
    }
}

// -------------------- 主函数 --------------------
int main() {
    // 固定记录数量（可根据需要调整）
    std::vector<size_t> num_records;
    num_records.push_back(32ULL * 1024);
    num_records.push_back(64ULL * 1024);
    num_records.push_back(128ULL * 1024);
    num_records.push_back(256ULL * 1024);
    num_records.push_back(1024ULL * 1024);

    // 定义测试参数：每个 N 对应的 k 列表
    std::vector<uint32_t> k_values_N1 = { 28, 31 };          // N=1 最大 32
    std::vector<uint32_t> k_values_N2 = { 41, 45, 51, 63 }; // N=2 最大 64
    //std::vector<uint32_t> k_values_N4 = { 81, 91, 101, 111, 127 };

    // 分别运行

    for (uint64_t num : num_records) {
        for (auto k : k_values_N1) {
            run_test<1>(k, num);
        }
        for (auto k : k_values_N2) {
            run_test<2>(k, num);
        }
        // for (auto k : k_values_N4) {
        //     run_test<4>(k, num);
        // }
    }

    return 0;
}