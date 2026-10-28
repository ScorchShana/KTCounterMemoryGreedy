#ifndef LOW_MSD_TEST_HELPERS_HEADER
#define LOW_MSD_TEST_HELPERS_HEADER
#include "../src/MsdPrefixCodec.h"
#include <filesystem>
#include <fstream>
#include <vector>

namespace low_msd_test {
using Key = std::vector<uint8_t>;
// Raw MSD groups make fixture bytes independent of the production encoder.
inline void write_raw(std::ostream& stream, std::span<const uint8_t> records, size_t width)
{
    if (!width || width > 32 || records.size() % width) throw std::runtime_error("invalid fixture records");
    while (!records.empty()) {
        const uint32_t count = static_cast<uint32_t>(std::min(msd_prefix::batch_records(width), records.size() / width));
        stream.put(0);
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
        const size_t bytes = size_t(count) * width;
        stream.write(reinterpret_cast<const char*>(records.data()), bytes);
        records = records.subspan(bytes);
    }
}

// Deliberately read independently of writer AIO boundaries. Only tests use this
// adapter; production tool readers keep their existing interfaces and format.
inline std::vector<Key> read(std::istream& input, size_t width, size_t slice = 65536)
{
    if (!slice || !width || width > 32) throw std::runtime_error("invalid test decoder dimensions");
    std::vector<char> chunk(slice);
    std::vector<uint8_t> pending, decoded(msd_prefix::BUFFER_BYTES);
    std::vector<Key> keys;
    while (input.read(chunk.data(), chunk.size()) || input.gcount()) {
        pending.insert(pending.end(), chunk.data(), chunk.data() + input.gcount());
        size_t pos = 0;
        while (pending.size() - pos >= msd_prefix::GROUP_HEADER_BYTES) {
            const size_t prefix = pending[pos];
            uint32_t count;
            std::memcpy(&count, pending.data() + pos + 1, sizeof(count));
            if (prefix >= width || !count || count > msd_prefix::batch_records(width))
                throw std::runtime_error("invalid test MSD group");
            const size_t bytes = 5 + prefix + size_t(count) * (width - prefix);
            if (pending.size() - pos < bytes) break;
            const auto group = msd_prefix::decode_group(std::span(pending).subspan(pos), width, decoded);
            for (size_t i = 0; i < group.records; ++i)
                keys.emplace_back(decoded.begin() + i * width, decoded.begin() + (i + 1) * width);
            pos += group.encoded_bytes;
        }
        pending.erase(pending.begin(), pending.begin() + pos);
    }
    if (!input.eof() || !pending.empty()) throw std::runtime_error("incomplete test MSD stream");
    return keys;
}
inline std::vector<Key> read(const std::filesystem::path& path, size_t width, size_t slice = 65536)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open test low output");
    return read(input, width, slice);
}
} // namespace low_msd_test
#endif
