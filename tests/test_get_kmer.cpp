#include "../src/GetKmer.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    std::vector<unsigned char> reference_records(const std::vector<unsigned char>& codes, uint32_t k)
    {
        const size_t width = (k + 3) / 4;
        std::vector<unsigned char> result;
        for (size_t end = k; end <= codes.size(); ++end) {
            std::vector<unsigned char> reversed(k), complement(k);
            for (uint32_t i = 0; i < k; ++i) {
                reversed[i] = codes[end - 1 - i];
                complement[i] = 3 - codes[end - k + i];
            }
            const auto& key = reversed < complement ? reversed : complement;
            const size_t offset = result.size();
            result.resize(offset + width, 0);
            for (uint32_t i = 0; i < k; ++i)
                result[offset + i / 4] |= key[i] << (6 - 2 * (i % 4));
        }
        return result;
    }

    // Catch incorrect tail stores, output strides, and warmup across batch boundaries.
    void check_batches(const std::vector<unsigned char>& codes, uint32_t k, uint32_t chunk)
    {
        GetKmer parser;
        const size_t width = (k + 3) / 4;
        std::vector<unsigned char> actual;
        size_t emitted = 0;
        for (size_t pos = 0; pos < codes.size();) {
            const uint32_t count = std::min<size_t>(chunk, codes.size() - pos);
            std::vector<unsigned char> output(count * width + 2, 0xa5);
            if (parser.batch_insert(codes.data() + pos, 0, output.data() + 1) != 0)
                throw std::runtime_error("empty batch emitted a record");
            const uint32_t written = parser.batch_insert(codes.data() + pos, count, output.data() + 1);
            pos += count;
            const size_t expected_total = pos >= k ? pos - k + 1 : 0;
            if (written != expected_total - emitted)
                throw std::runtime_error("batch record count differs");
            emitted = expected_total;
            if (output.front() != 0xa5)
                throw std::runtime_error("batch overwrote leading guard");
            for (size_t i = 1 + written * width; i < output.size(); ++i)
                if (output[i] != 0xa5) throw std::runtime_error("batch wrote past its records");
            actual.insert(actual.end(), output.begin() + 1, output.begin() + 1 + written * width);
        }
        unsigned char guard = 0xa5;
        if (parser.batch_insert(codes.data(), 0, &guard) != 0 || guard != 0xa5)
            throw std::runtime_error("empty batch changed output after warmup");
        if (actual != reference_records(codes, k))
            throw std::runtime_error("batch record bytes differ");
    }

    // Catch changed ASCII mapping, invalid-base resets, and explicit clear behavior.
    void check_characters(uint32_t k)
    {
        GetKmer parser;
        const std::string alphabet = "AaCcGgTtUu";
        std::string input;
        for (uint32_t i = 0; i < k + 37; ++i) input += alphabet[(i * 7) % alphabet.size()];
        input += "N\n\r?";
        input += static_cast<char>(0xff);
        for (uint32_t i = 0; i < k + 37; ++i) input += alphabet[(i * 3) % alphabet.size()];
        std::vector<unsigned char> run;
        const size_t width = (k + 3) / 4;
        for (size_t i = 0; i < input.size(); ++i) {
            if (i == k + 3) {
                parser.clear();
                run.clear();
            }
            const size_t code = alphabet.find(input[i]);
            if (code == std::string::npos) run.clear();
            else run.push_back(std::min<size_t>(code / 2, 3));
            const bool expected = run.size() >= k;
            std::vector<unsigned char> output(width + 2, 0xa5);
            if (parser.get_next_one(static_cast<unsigned char>(input[i]), output.data() + 1) != expected)
                throw std::runtime_error("character record count differs");
            if (expected) {
                const std::vector<unsigned char> window(run.end() - k, run.end());
                const auto record = reference_records(window, k);
                if (!std::equal(record.begin(), record.end(), output.begin() + 1))
                    throw std::runtime_error("character record bytes differ");
            } else {
                for (size_t j = 1; j <= width; ++j)
                    if (output[j] != 0xa5) throw std::runtime_error("warmup changed output");
            }
            if (output.front() != 0xa5 || output.back() != 0xa5)
                throw std::runtime_error("character overwrote a guard");
        }
    }
}

int main()
{
    uint32_t current_k = 0;
    try {
        std::mt19937 rng(8831);
        for (current_k = 1; current_k <= 128; ++current_k) {
            k_length = current_k;
            std::vector<unsigned char> codes(current_k + 67);
            for (auto& code : codes) code = rng() % 4;
            for (uint32_t chunk : {static_cast<uint32_t>(codes.size()), 1u, 7u, 16u, 32u,
                                   std::max(1u, current_k - 1), current_k})
                check_batches(codes, current_k, chunk);
            // Equal leading words force canonical comparison beyond the first word.
            std::fill(codes.begin(), codes.end(), 0);
            for (size_t i = 0; i < codes.size(); i += 2) codes[i] = 3;
            check_batches(codes, current_k, 16);
            check_characters(current_k);
        }
        std::cout << "GetKmer tests passed for k=1..128\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "k=" << current_k << ": " << error.what() << '\n';
        return 1;
    }
}
