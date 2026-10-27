#include "../src/CountingHashMap.h"

void counting_hash_map_link_probe()
{
    k_length = 1;
    CountingHashMap map;
    const unsigned char key = 0;
    map.increment(&key);
    map.clear();
}
