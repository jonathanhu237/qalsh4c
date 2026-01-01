#include "dataset.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

void WriteU32BE(std::ofstream& ofs, uint32_t value) {
    unsigned char bytes[4] = {static_cast<unsigned char>((value >> 24U) & 0xFFU),
                              static_cast<unsigned char>((value >> 16U) & 0xFFU),
                              static_cast<unsigned char>((value >> 8U) & 0xFFU),
                              static_cast<unsigned char>(value & 0xFFU)};
    ofs.write(reinterpret_cast<const char*>(bytes), 4);
}

void WriteFvecs(const std::filesystem::path& file_path, const std::vector<std::vector<float>>& vectors) {
    std::ofstream ofs(file_path, std::ios::binary);
    for (const auto& vector : vectors) {
        const uint32_t dimension = static_cast<uint32_t>(vector.size());
        ofs.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        ofs.write(reinterpret_cast<const char*>(vector.data()), static_cast<std::streamsize>(vector.size() * sizeof(float)));
    }
}

void WriteMnist(const std::filesystem::path& file_path) {
    std::ofstream ofs(file_path, std::ios::binary);
    WriteU32BE(ofs, 2051);
    WriteU32BE(ofs, 4);
    WriteU32BE(ofs, 2);
    WriteU32BE(ofs, 2);
    for (unsigned char value = 0; value < 16; ++value) {
        ofs.write(reinterpret_cast<const char*>(&value), 1);
    }
}

void WriteBmp24(const std::filesystem::path& file_path) {
    constexpr uint32_t width = 128;
    constexpr uint32_t height = 64;
    constexpr uint16_t bits = 24;
    constexpr uint32_t row_stride = ((bits * width + 31U) / 32U) * 4U;
    constexpr uint32_t pixel_size = row_stride * height;
    constexpr uint32_t pixel_offset = 14 + 40;
    constexpr uint32_t file_size = pixel_offset + pixel_size;

    std::ofstream ofs(file_path, std::ios::binary);
    WriteU16LE(ofs, 0x4D42);
    WriteU32LE(ofs, file_size);
    WriteU16LE(ofs, 0);
    WriteU16LE(ofs, 0);
    WriteU32LE(ofs, pixel_offset);
    WriteU32LE(ofs, 40);
    WriteI32LE(ofs, static_cast<int32_t>(width));
    WriteI32LE(ofs, static_cast<int32_t>(height));
    WriteU16LE(ofs, 1);
    WriteU16LE(ofs, bits);
    WriteU32LE(ofs, 0);
    WriteU32LE(ofs, pixel_size);
    WriteI32LE(ofs, 0);
    WriteI32LE(ofs, 0);
    WriteU32LE(ofs, 0);
    WriteU32LE(ofs, 0);

    std::vector<unsigned char> row(row_stride, 0);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t offset = static_cast<size_t>(x) * 3U;
            row[offset] = static_cast<unsigned char>(x % 256U);
            row[offset + 1] = static_cast<unsigned char>(y % 256U);
            row[offset + 2] = static_cast<unsigned char>((x + y) % 256U);
        }
        ofs.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
}

DatasetPrep::CommonOutputOptions OutputOptions(const std::filesystem::path& output_dir) {
    return DatasetPrep::CommonOutputOptions{
        .output_directory = output_dir,
        .ground_truth_specs = "",
        .overwrite = true,
        .show_progress = false,
    };
}

bool CheckMetadata(const std::filesystem::path& output_dir, unsigned int dimensions, unsigned int a, unsigned int b) {
    DatasetMetadata metadata;
    metadata.Load(output_dir / "metadata.json");
    return metadata.num_dimensions == dimensions && metadata.num_points_a == a && metadata.num_points_b == b &&
           !metadata.chamfer_distance_l1_ab.has_value() && !metadata.chamfer_distance_l2_ab.has_value();
}

}  // namespace

int main() {
    const std::filesystem::path root = std::filesystem::current_path() / "dataset_loader_test_tmp";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    const std::filesystem::path mnist_dir = root / "mnist-raw";
    std::filesystem::create_directories(mnist_dir);
    WriteMnist(mnist_dir / "train-images-idx3-ubyte");
    WriteMnist(mnist_dir / "t10k-images-idx3-ubyte");
    DatasetPrep::ConvertDataset(DatasetPrep::ConvertOptions{
        .output = OutputOptions(root / "mnist-out"),
        .kind = DatasetPrep::DatasetKind::kMnist,
        .raw_directory = mnist_dir,
        .num_points_a = 3,
        .shuffle = false,
    });
    if (!CheckMetadata(root / "mnist-out", 4, 3, 5)) {
        return 1;
    }

    const std::filesystem::path p53_dir = root / "p53-raw";
    std::filesystem::create_directories(p53_dir);
    {
        std::ofstream ofs(p53_dir / "K9.data");
        ofs << "9,10,11\n";
        ofs << "1,2,3,4,5,6\n";
        ofs << "?,2,3,4,5,6\n";
        ofs << "7,8,9,10,11,12\n";
    }
    DatasetPrep::ConvertDataset(DatasetPrep::ConvertOptions{
        .output = OutputOptions(root / "p53-out"),
        .kind = DatasetPrep::DatasetKind::kP53,
        .raw_directory = p53_dir,
        .num_points_a = 1,
        .shuffle = false,
    });
    if (!CheckMetadata(root / "p53-out", 4, 1, 1)) {
        return 2;
    }

    const std::filesystem::path sift_dir = root / "sift-raw";
    std::filesystem::create_directories(sift_dir);
    WriteFvecs(sift_dir / "sift_base.fvecs", {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}});
    WriteFvecs(sift_dir / "sift_learn.fvecs", {{3.0F, 4.0F}});
    WriteFvecs(sift_dir / "sift_query.fvecs", {{5.0F, 6.0F}});
    DatasetPrep::ConvertDataset(DatasetPrep::ConvertOptions{
        .output = OutputOptions(root / "sift-out"),
        .kind = DatasetPrep::DatasetKind::kSift,
        .raw_directory = sift_dir,
        .num_points_a = 1,
        .shuffle = false,
    });
    if (!CheckMetadata(root / "sift-out", 2, 1, 2)) {
        return 3;
    }

    const std::filesystem::path gist_dir = root / "gist-raw";
    std::filesystem::create_directories(gist_dir);
    WriteFvecs(gist_dir / "gist_base.fvecs", {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}});
    WriteFvecs(gist_dir / "gist_learn.fvecs", {{3.0F, 4.0F}});
    WriteFvecs(gist_dir / "gist_query.fvecs", {{5.0F, 6.0F}});
    DatasetPrep::ConvertDataset(DatasetPrep::ConvertOptions{
        .output = OutputOptions(root / "gist-out"),
        .kind = DatasetPrep::DatasetKind::kGist,
        .raw_directory = gist_dir,
        .num_points_a = 1,
        .shuffle = false,
    });
    if (!CheckMetadata(root / "gist-out", 2, 1, 2)) {
        return 4;
    }

    const std::filesystem::path trevi_dir = root / "trevi-raw";
    std::filesystem::create_directories(trevi_dir);
    WriteBmp24(trevi_dir / "image.bmp");
    DatasetPrep::ConvertDataset(DatasetPrep::ConvertOptions{
        .output = OutputOptions(root / "trevi-out"),
        .kind = DatasetPrep::DatasetKind::kTrevi,
        .raw_directory = trevi_dir,
        .num_points_a = 1,
        .shuffle = false,
    });
    if (!CheckMetadata(root / "trevi-out", 4096, 1, 1)) {
        return 5;
    }

    std::filesystem::remove_all(root);
    return 0;
}
