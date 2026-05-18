#include "dataset.h"

#include <cstdint>
#include <filesystem>
#include <fstream>

namespace {

void WriteU16LE(std::ofstream& ofs, uint16_t value) {
    unsigned char bytes[2] = {static_cast<unsigned char>(value & 0xFFU),
                              static_cast<unsigned char>((value >> 8U) & 0xFFU)};
    ofs.write(reinterpret_cast<const char*>(bytes), 2);
}

void WriteU32LE(std::ofstream& ofs, uint32_t value) {
    unsigned char bytes[4] = {static_cast<unsigned char>(value & 0xFFU),
                              static_cast<unsigned char>((value >> 8U) & 0xFFU),
                              static_cast<unsigned char>((value >> 16U) & 0xFFU),
                              static_cast<unsigned char>((value >> 24U) & 0xFFU)};
    ofs.write(reinterpret_cast<const char*>(bytes), 4);
}

void WriteI32LE(std::ofstream& ofs, int32_t value) {
    WriteU32LE(ofs, static_cast<uint32_t>(value));
}

}  // namespace

int main() {
    const std::filesystem::path root = std::filesystem::current_path() / "dataset_bad_bmp_test_tmp";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "raw");

    std::ofstream ofs(root / "raw" / "bad.bmp", std::ios::binary);
    WriteU16LE(ofs, 0x4D42);
    WriteU32LE(ofs, 54);
    WriteU16LE(ofs, 0);
    WriteU16LE(ofs, 0);
    WriteU32LE(ofs, 54);
    WriteU32LE(ofs, 40);
    WriteI32LE(ofs, 64);
    WriteI32LE(ofs, 64);
    WriteU16LE(ofs, 1);
    WriteU16LE(ofs, 24);
    WriteU32LE(ofs, 1);  // Compression is intentionally unsupported.
    WriteU32LE(ofs, 0);
    WriteI32LE(ofs, 0);
    WriteI32LE(ofs, 0);
    WriteU32LE(ofs, 0);
    WriteU32LE(ofs, 0);
    ofs.close();

    DatasetPrep::ConvertDataset(DatasetPrep::ConvertOptions{
        .output = {.output_directory = root / "out", .overwrite = true, .show_progress = false},
        .kind = DatasetPrep::DatasetKind::kTrevi,
        .raw_directory = root / "raw",
        .num_points_a = 1,
        .shuffle = false,
    });

    return 0;
}
