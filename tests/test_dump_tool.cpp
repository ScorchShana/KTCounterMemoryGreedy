#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
struct Record { std::string key; uint32_t count; };
fs::path dir;
std::string tool;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}
void infos(unsigned k, uint32_t maximum) {
    std::ofstream f(dir / "infos.bin", std::ios::binary);
    uint32_t values[] = {k, maximum};
    f.write(reinterpret_cast<char*>(values), sizeof(values));
}
unsigned width(uint32_t maximum) {
    return maximum <= 255 ? 1 : maximum <= 65535 ? 2 : maximum <= 16777215 ? 3 : 4;
}
// Independent fixture encoder: reverse the supplied TEXT, then pack four bases
// per byte, most significant pair first. No production headers are included.
std::vector<uint8_t> packed(const std::string& key) {
    std::vector<uint8_t> bytes((key.size() + 3) / 4);
    for (size_t i = 0; i < key.size(); ++i) {
        const auto base = std::string("ACGT").find(key[key.size() - 1 - i]);
        bytes[i / 4] |= uint8_t(base << (6 - 2 * (i % 4)));
    }
    return bytes;
}
void group(std::ofstream& f, const std::vector<Record>& records, unsigned cb, unsigned p = 0) {
    require(!records.empty(), "empty fixture group");
    auto first = packed(records.front().key);
    require(p < first.size(), "fixture prefix");
    f.put(char(p));
    uint32_t n = records.size();
    f.write(reinterpret_cast<char*>(&n), 4);
    f.write(reinterpret_cast<char*>(first.data()), p);
    for (const auto& r : records) {
        auto b = packed(r.key);
        require(std::equal(first.begin(), first.begin() + p, b.begin()), "fixture shared prefix");
        f.write(reinterpret_cast<char*>(b.data() + p), b.size() - p);
        for (unsigned j = 0; j < cb; ++j) f.put(char(r.count >> (8 * j)));
    }
    require(bool(f), "fixture write failed");
}
void fixture(unsigned k, uint32_t maximum, const std::vector<Record>& high,
             const std::vector<Record>& low = {}, unsigned p = 0) {
    infos(k, maximum);
    std::ofstream h(dir / "high.bin", std::ios::binary), l(dir / "low.bin", std::ios::binary);
    if (!high.empty()) group(h, high, width(maximum), p);
    if (!low.empty()) group(l, low, 0);
}
std::string numbered(unsigned k, uint64_t n) {
    std::string key(k, 'A');
    for (size_t i = 0; i < k && n; ++i, n >>= 2) key[i] = "ACGT"[n & 3];
    return key;
}
int run(bool precise = true, bool sorted = false, const std::string& memory = "0.08",
        unsigned workers = 2, std::optional<uint32_t> min = 1,
        std::optional<uint32_t> max = UINT32_MAX, const fs::path& output = {}, unsigned fd_limit = 0,
        const std::vector<std::string>& extra = {}, unsigned address_limit_mib = 0,
        bool deny_threads = false) {
    std::vector<std::string> args = {tool, precise ? "precise" : "approximate", dir.string(),
        std::to_string(workers), memory, (output.empty() ? dir / "out.tsv" : output).string()};
    if (min) args.push_back(std::to_string(*min));
    if (max) args.push_back(std::to_string(*max));
    if (sorted) args.push_back("--sort");
    args.insert(args.end(), extra.begin(), extra.end());
    pid_t pid = fork();
    require(pid >= 0, "fork failed");
    if (pid == 0) {
        if (fd_limit) { rlimit lim{fd_limit, fd_limit}; if (setrlimit(RLIMIT_NOFILE, &lim)) _exit(126); }
        if (address_limit_mib) {
            rlim_t bytes = rlim_t(address_limit_mib) << 20;
            rlimit lim{bytes,bytes}; if (setrlimit(RLIMIT_AS,&lim)) _exit(126);
        }
        if (deny_threads) { rlimit lim{1,1}; if (setrlimit(RLIMIT_NPROC,&lim)) _exit(126); }
        int out = open((dir / "stdout").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int err = open((dir / "stderr").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        dup2(out, STDOUT_FILENO); dup2(err, STDERR_FILENO); close(out); close(err);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        execv(argv[0], argv.data());
        _exit(127);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(25);
    int status;
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            kill(pid, SIGKILL); waitpid(pid, &status, 0);
            throw std::runtime_error("dump hung: " + read(dir / "stderr"));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(WIFEXITED(status), "dump crashed");
    return WEXITSTATUS(status);
}
std::vector<std::string> lines(const std::string& text) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        require(end != std::string::npos, "unterminated line");
        result.push_back(text.substr(start, end - start)); start = end + 1;
    }
    return result;
}
void verify(const std::vector<Record>& h, const std::vector<Record>& l, bool precise,
            bool sorted, unsigned workers = 2, uint32_t min = 1, uint32_t max = UINT32_MAX,
            const std::string& memory = "0.08", unsigned fd_limit = 0) {
    int status = run(precise, sorted, memory, workers, min, max, {}, fd_limit);
    require(status == 0, "dump failed: " + read(dir / "stderr"));
    std::vector<std::string> expected;
    for (const auto& r : h) if (r.count >= min && r.count <= max)
        expected.push_back(r.key + "\t" + std::to_string(r.count));
    if (precise && min <= 1 && max >= 1) for (const auto& r : l) expected.push_back(r.key + "\t1");
    auto actual = lines(read(dir / "out.tsv"));
    if (sorted) require(std::is_sorted(actual.begin(), actual.end()), "not globally sorted");
    std::sort(expected.begin(), expected.end()); std::sort(actual.begin(), actual.end());
    require(actual == expected, "records differ");
    for (const auto& e : fs::directory_iterator(dir))
        require(e.path().filename().string().find("dump-sort-") != 0, "temporary directory leaked");
}
void representation() {
    // Literal asymmetric sentinel catches reversal and partial-byte errors.
    fixture(7, 255, {{"ACGTTCA", 1}, {"TGCATGC", 255}}, {{"CAAAAGT", 1}});
    verify({{"ACGTTCA", 1}, {"TGCATGC", 255}}, {{"CAAAAGT", 1}}, true, true);
    const uint32_t maxima[] = {255, 65535, 16777215, UINT32_MAX};
    for (unsigned k = 1; k <= 128; ++k) {
        uint32_t maximum = maxima[(k - 1) % 4];
        std::vector<Record> h = {{numbered(k, 0), 1}, {numbered(k, 1), maximum - 1},
                                 {numbered(k, 2), maximum}};
        std::vector<Record> l = {{numbered(k, 3), 1}};
        fixture(k, maximum, h, l, (k + 3) / 4 - 1);
        unsigned workers = k % 4 == 0 ? 12 : k % 4 == 1 ? 1 : k % 4 == 2 ? 2 : 8;
        verify(h, l, true, k % 2, workers);
        verify(h, l, false, !(k % 2), workers);
    }
}
void full_width_keys() {
    std::mt19937 rng(713919);
    for (unsigned k : {4,5,17,31,32,33,63,64,65,127,128}) {
        std::set<std::string> used;
        std::vector<Record> high,low;
        unsigned records = k < 6 ? 64 : 1152;
        while (used.size() < records) {
            std::string key(k,'A');
            for (auto& ch : key) ch = "ACGT"[rng() % 4];
            if (!used.insert(key).second) continue;
            if (used.size() % 17 == 0) low.push_back({key,1});
            else high.push_back({key,used.size()%19 == 0 ? 1U : uint32_t(rng())});
        }
        fixture(k,UINT32_MAX,high,low);
        verify(high,low,true,false,12);
        verify(high,low,true,true,8);
    }
}
void default_max_frequency() {
    const std::vector<Record> high = {{"AAAA",9999},{"CCCC",10000},{"GGGG",10001}};
    const std::vector<Record> low = {{"TTTT",1}};
    fixture(4,65535,high,low);
    for (bool precise : {false,true}) for (bool sorted : {false,true}) {
        for (auto minimum : {std::optional<uint32_t>{},std::optional<uint32_t>{1}}) {
            int status = run(precise,sorted,"0.08",2,minimum,std::nullopt);
            require(status == 0,"default max frequency invocation failed: " + read(dir/"stderr"));
            auto actual = lines(read(dir/"out.tsv"));
            std::vector<std::string> expected = {"AAAA\t9999","CCCC\t10000"};
            if (precise) expected.push_back("TTTT\t1");
            if (sorted) require(std::is_sorted(actual.begin(),actual.end()),"default output not sorted");
            std::sort(actual.begin(),actual.end());
            require(actual == expected,"default max_freq must include 10000 and exclude 10001");
        }
        verify(high,low,precise,sorted,2,1,10001);
    }
}
void output_cases() {
    std::vector<Record> h = {{"ACGTACGTACGTACGTA", 1}, {"GGTACGTACGTACGTAC", 2},
                             {"TCGTACGTACGTACGTA", 300}};
    std::vector<Record> l = {{"CCGTACGTACGTACGTA", 1}};
    for (bool sorted : {false, true}) for (bool precise : {false, true}) {
        for (const auto& sides : std::vector<std::pair<bool,bool>>{{false,false},{true,false},{false,true},{true,true}}) {
            const auto a = sides.first ? h : std::vector<Record>{};
            const auto b = sides.second ? l : std::vector<Record>{};
            fixture(17, 65535, a, b);
            verify(a,b,precise,sorted);
        }
        fixture(17, 65535, h, l);
        for (auto bounds : std::vector<std::pair<uint32_t,uint32_t>>{{1,1},{2,2},{300,300},{301,1000},{0,UINT32_MAX}})
            verify(h,l,precise,sorted,2,bounds.first,bounds.second);
    }
    fixture(17, 65535, h, l);
    fs::remove(dir / "low.bin");
    verify(h,{},false,false);
    verify(h,{},true,true,2,2);
}
void boundaries() {
    constexpr size_t window = 2ULL << 20;
    // Move each part of a group across an exact AIO window boundary.
    for (unsigned split : {1U, 4U, 6U, 11U, 14U, 18U, 23U, 25U}) {
        infos(65, UINT32_MAX);
        std::ofstream h(dir / "high.bin", std::ios::binary), l(dir / "low.bin", std::ios::binary);
        std::vector<Record> expected;
        const size_t target = window - split;
        const size_t stride = 17 + 4;
        size_t n = (target - 5) / stride - 100;
        std::vector<Record> bulk;
        for (size_t i = 0; i < n; ++i) bulk.push_back({numbered(65, i), uint32_t(i % 13 + 1)});
        group(h, bulk, 4); expected = std::move(bulk);
        size_t remaining = target - size_t(h.tellp());
        bool found = false;
        for (unsigned singles = 0; singles < 10 && !found; ++singles) {
            for (unsigned p = 0; p < 16 && !found; ++p) {
                if (remaining <= singles*26 + 5 + p) continue;
                size_t rest = remaining - singles*26;
                if ((rest - 5 - p) % (21 - p)) continue;
                for (unsigned i = 0; i < singles; ++i) {
                    auto key = numbered(65, 500000+i); key.back() = 'G';
                    Record r{key,9}; group(h,{r},4); expected.push_back(r);
                }
                size_t count = (rest - 5 - p) / (21 - p);
                std::vector<Record> fill;
                for (size_t i = 0; i < count; ++i) {
                    std::string key = numbered(65, i); key.replace(5,60,60,'T');
                    fill.push_back({key, 7});
                }
                group(h, fill, 4, p);
                expected.insert(expected.end(), fill.begin(), fill.end()); found = true;
            }
        }
        require(found && size_t(h.tellp()) == target, "boundary fixture alignment");
        std::vector<Record> tail;
        for (unsigned i = 0; i < 500; ++i) {
            auto key = numbered(65, 1000000 + i); key.replace(45,20,20,'C');
            tail.push_back({key, UINT32_MAX - i});
        }
        group(h, tail, 4, 5); expected.insert(expected.end(), tail.begin(), tail.end());
        h.close(); l.close();
        verify(expected,{},true,false,12);
        verify(expected,{},true,true,8);
    }
    // A single group spans many windows and expanded text exceeds all 8 blocks.
    std::vector<Record> high;
    for (unsigned i = 0; i < 310000; ++i) high.push_back({numbered(65,i), i % 99 + 1});
    fixture(65,255,high);
    for (unsigned workers : {1,2,8,12,12}) verify(high,{},true,false,workers);
    require(run(true,false,"0.08",12,1,UINT32_MAX,"/dev/full") != 0,
            "backpressure writer failure hung or was ignored");
    // The low side uses the same cross-window parser without count bytes.
    fixture(65,255,{},high);
    verify({},high,true,false,12);
    verify({},high,true,true,8);
    // Thousands of tiny groups exercise descriptor and task queue bounds.
    infos(17,255);
    std::ofstream h(dir / "high.bin", std::ios::binary), l(dir / "low.bin", std::ios::binary);
    high.clear();
    for (unsigned i = 0; i < 20000; ++i) {
        Record r{numbered(17,i),i%255+1}; group(h,{r},1); high.push_back(r);
    }
    h.close(); l.close(); verify(high,{},true,false,12);
}
void sort_capacity() {
    fixture(17,255,{});
    verify({},{},true,true,1,1,UINT32_MAX,"0.032");
    std::string log = read(dir / "stdout");
    size_t marker = log.find("sort capacity ");
    require(marker != std::string::npos,"missing capacity diagnostic");
    size_t capacity = std::stoull(log.substr(marker+14));
    require(capacity < 1000000,"unexpected fixture capacity");
    std::vector<Record> high;
    for (size_t i = 0; i < capacity; ++i) high.push_back({numbered(17,i),1});
    fixture(17,255,high); verify(high,{},true,true,1,1,UINT32_MAX,"0.032");
    require(read(dir / "stdout").find("sort temporary bytes 0") != std::string::npos,"exact capacity spilled");
    high.push_back({numbered(17,capacity),1});
    fixture(17,255,high); verify(high,{},true,true,1,1,UINT32_MAX,"0.032");
    require(read(dir / "stdout").find("sort temporary bytes 0") == std::string::npos,"capacity overflow did not spill");
}
void external_sort() {
    std::vector<Record> high,low;
    for (unsigned i = 0; i < 180000; ++i) high.push_back({numbered(33,i),i%65535+1});
    for (unsigned i = 180000; i < 200000; ++i) low.push_back({numbered(33,i),1});
    std::mt19937 rng(42); std::shuffle(high.begin(),high.end(),rng);
    fixture(33,65535,high,low);
    verify(high,low,true,true,2,1,UINT32_MAX,"0.08");
    require(read(dir / "stdout").find("sort temporary bytes 0") != std::string::npos,
            "in-memory sort created runs");
    auto reference = read(dir / "out.tsv");
    verify(high,low,true,true,2,1,UINT32_MAX,"0.032",18);
    require(read(dir / "stdout").find("sort merge passes 1\n") == std::string::npos,
            "fixture did not exercise multiple merge passes");
    require(read(dir / "stdout").find("sort temporary bytes 0") == std::string::npos,
            "fixture did not spill");
    require(read(dir / "out.tsv") == reference, "external/in-memory bytes differ");
    require(run(true,true,"0.032",2,1,UINT32_MAX,{},17) != 0,"insufficient fan-in accepted");
    for (const auto& e : fs::directory_iterator(dir))
        require(e.path().filename().string().find("dump-sort-") != 0,"error leaked runs");
    require(run(true,true,"0.032",2,1,UINT32_MAX,"/dev/full",18) != 0,"merge write error ignored");
    if (geteuid() != 0) {
        fs::permissions(dir,fs::perms::owner_read|fs::perms::owner_exec);
        int status = run(true,true,"0.032",2,1,UINT32_MAX,{},18);
        fs::permissions(dir,fs::perms::owner_all);
        require(status != 0 && read(dir/"stderr").find("mkdtemp") != std::string::npos,
                "temporary directory failure not propagated");
    }
    for (const auto& e : fs::directory_iterator(dir))
        require(e.path().filename().string().find("dump-sort-") != 0,"write error leaked runs");
}
void errors() {
    const std::vector<Record> h = {{"ACGTTCAACGTTCAACG",300}};
    fixture(17,65535,h);
    for (const std::string memory : {"0","-1","nan","inf","0.001","0.03125"})
        require(run(true,false,memory) != 0,"invalid budget accepted");
    require(run(true,false,"0.08",0) != 0,"zero workers accepted");
    require(run(true,false,"0.08",2,3,2) != 0,"reversed range accepted");
    require(run(true,false,"0.08",2,1,UINT32_MAX,{},0,{"--bogus"}) != 0,"unknown flag accepted");
    require(run(true,false,"0.08",2,1,UINT32_MAX,"/dev/full") != 0,"write error ignored");
    require(run(true,false,"0.08",2,1,UINT32_MAX,dir) != 0,"output directory accepted");
    // Sanitizers reserve a huge virtual address range and have their own
    // threads, so resource-failure tests run only against the normal binary.
    if (!std::getenv("ASAN_OPTIONS")) {
        require(run(true,true,"1",2,1,UINT32_MAX,{},0,{},128) != 0 &&
                read(dir/"stderr").find("bad_alloc") != std::string::npos,
                "sorting allocation failure not reported");
        if (geteuid() != 0)
            require(run(true,false,"0.08",2,1,UINT32_MAX,{},0,{},0,true) != 0 &&
                    read(dir/"stderr").find("dump_tool:") != std::string::npos,
                    "thread creation failure not reported");
    }
    auto original = read(dir / "high.bin");
    for (const char* name : {"high.bin","symbolic","hard"}) {
        fs::path path = dir / name;
        if (std::string(name) == "symbolic") fs::create_symlink(dir / "high.bin",path);
        if (std::string(name) == "hard") fs::create_hard_link(dir / "high.bin",path);
        require(run(true,false,"0.08",2,1,UINT32_MAX,path) != 0,"input overwrite accepted");
        require(read(dir / "high.bin") == original,"input corrupted");
    }
    require(run(true,true,"0.08",2,1,UINT32_MAX,dir/"infos.bin") != 0,"infos overwrite accepted");
    require(run(true,false,"0.08",2,1,UINT32_MAX,dir/"low.bin") != 0,"low overwrite accepted");
    for (unsigned cut = 1; cut < original.size(); ++cut) {
        fixture(17,65535,h);
        fs::resize_file(dir/"high.bin",cut);
        require(run() != 0,"truncation accepted at " + std::to_string(cut));
    }
    for (unsigned offset : {0U,1U}) {
        fixture(17,65535,h);
        std::fstream f(dir/"high.bin",std::ios::in|std::ios::out|std::ios::binary);
        f.seekp(offset); f.put(offset == 0 ? char(5) : char(0)); f.close();
        require(run() != 0,"invalid group accepted");
    }
    for (bool sorted : {false,true}) {
        fixture(17,65535,h);
        std::ofstream l(dir/"low.bin",std::ios::binary);
        group(l,{{"TGCATGCATGCATGCAT",1}},0,4); l.close();
        auto low = read(dir/"low.bin");
        for (size_t cut = 1; cut < low.size(); ++cut) {
            { std::ofstream f(dir/"low.bin",std::ios::binary); f.write(low.data(),cut); }
            require(run(true,sorted) != 0,"low prefix/suffix truncation accepted");
        }
    }
    for (auto values : std::vector<std::pair<unsigned,uint32_t>>{{0,1},{129,1},{17,0}}) {
        fixture(17,65535,h); infos(values.first,values.second);
        require(run() != 0,"invalid infos accepted");
    }
    fixture(17,65535,h); fs::remove(dir/"infos.bin");
    require(mkfifo((dir/"infos.bin").c_str(),0600) == 0,"fixture mkfifo failed");
    require(run() != 0,"non-regular infos accepted");
    fs::remove(dir/"infos.bin");
    fixture(17,65535,h); fs::resize_file(dir/"infos.bin",7);
    require(run() != 0,"short infos accepted");
    fixture(17,65535,h); fs::remove(dir/"high.bin");
    require(run() != 0,"missing input accepted");
    fs::create_directory(dir/"high.bin"); require(run() != 0,"non-regular input accepted");
    fs::remove(dir/"high.bin");
}
}
int main(int argc,char** argv) {
    if (argc < 2) return 2;
    tool = fs::absolute(argv[1]);
    char path[] = "/tmp/dump-tool-test-XXXXXX";
    if (!mkdtemp(path)) return 2;
    dir = path;
    try {
        default_max_frequency(); representation(); full_width_keys(); output_cases();
        if (argc < 3) { boundaries(); sort_capacity(); external_sort(); errors(); }
        fs::remove_all(dir); std::cout << "DumpToolTest passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\nfixtures: " << dir << '\n'; return 1;
    }
}
