#include "../tool/FlatConcurrentHashMap.h"
#include <array>
#include <cerrno>
#include <dlfcn.h>
#include <iostream>
#include <pthread.h>

namespace
{
    bool injecting = false;
    int fail_mapping = 0, mapping_calls = 0, fail_thread = 0, thread_calls = 0;
    std::array<void*, 3> mappings{};
    int live_mappings = 0;
}

// Interpose only in this test executable: exercise real constructor failures
// without adding fault injection or test-only methods to production classes.
extern "C" void* mmap(void* addr, size_t size, int prot, int flags, int fd, off_t offset) noexcept
{
    using Function = void* (*)(void*, size_t, int, int, int, off_t);
    static auto real_mmap = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "mmap"));
    if (injecting)
    {
        if ((flags & MAP_HUGETLB) || ++mapping_calls == fail_mapping)
        {
            errno = ENOMEM;
            return MAP_FAILED;
        }
    }
    void* result = real_mmap(addr, size, prot, flags, fd, offset);
    if (injecting && result != MAP_FAILED)
    {
        for (auto& slot : mappings)
            if (!slot) { slot = result; ++live_mappings; break; }
    }
    return result;
}

extern "C" int munmap(void* addr, size_t size) noexcept
{
    using Function = int (*)(void*, size_t);
    static auto real_munmap = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "munmap"));
    int status = real_munmap(addr, size);
    if (status == 0)
        for (auto& slot : mappings)
            if (slot == addr) { slot = nullptr; --live_mappings; break; }
    return status;
}

extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                               void* (*start)(void*), void* arg) noexcept
{
    using Function = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
    static auto real_create = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "pthread_create"));
    if (injecting && ++thread_calls == fail_thread) return EAGAIN;
    return real_create(thread, attr, start, arg);
}

int main(int argc, char* argv[])
{
    if (argc != 2) return 1;
    bool test_threads = std::string(argv[1]) == "threads";
    for (uint32_t k : {21U, 32U})
    for (int failure : {2, 3})
    {
        mapping_calls = thread_calls = 0;
        fail_mapping = test_threads ? 0 : failure;
        fail_thread = test_threads ? failure : 0;
        bool threw = false;
        injecting = true;
        try { FlatConcurrentHashMap<1> map(1, 4, k); }
        catch (const std::exception&) { threw = true; }
        injecting = false;
        if (!threw || live_mappings != 0)
        {
            std::cerr << "constructor failure " << failure << " left " << live_mappings << " mappings\n";
            return 1;
        }
    }
    // Successful construction after a failure must also release all mappings.
    fail_mapping = fail_thread = 0;
    injecting = true;
    { FlatConcurrentHashMap<1> map(1, 2, 21); }
    injecting = false;
    if (live_mappings != 0) return 1;
    std::cout << "Hash-map constructor failure cleanup passed\n";
    return 0;
}
