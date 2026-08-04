#include "../tool/FinalDrainReader.h"
#include "../src/kmer.h"

#include <chrono>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    FinalDrainReader<2> reader(2*32);
    reader.open(argv[1]);
    ExportRecord<2> k_mers[32 * 1024 / sizeof(ExportRecord<2>)];

    auto start = std::chrono::high_resolution_clock::now();
    while (!reader.finished())
    {
        reader.read_records(reinterpret_cast<ExportRecord<2>*>(k_mers), 32 * 1024 / sizeof(ExportRecord<2>));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total records read: " << reader.get_record_amount() << std::endl;
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << " ns" << std::endl;
    return 0;
}