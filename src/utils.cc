#include "utils.h"

#include <numeric>
#include <random>

auto Utils::SampleFromWeights(const std::vector<double> &weights) -> unsigned int {
    double total_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_sum <= 0) {
        throw std::runtime_error("Total sum of weights must be positive.");
    }

    std::vector<double> cumulative_weights;
    cumulative_weights.reserve(weights.size());
    std::partial_sum(weights.begin(), weights.end(), std::back_inserter(cumulative_weights));

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, total_sum);
    double random_value = dis(gen);

    auto it = std::ranges::upper_bound(cumulative_weights, random_value);

    return static_cast<unsigned int>(std::distance(cumulative_weights.begin(), it));
}

// Define BMP structures with packing to ensure correct memory layout
#pragma pack(push, 1)
struct BitmapFileHeader {
    uint16_t type;       // Magic identifier: "BM"
    uint32_t size;       // File size in bytes
    uint16_t reserved1;  // Not used
    uint16_t reserved2;  // Not used
    uint32_t offset;     // Offset to image data in bytes from the start of the file
};

struct BitmapInfoHeader {
    uint32_t size;              // Header size in bytes
    int32_t width;              // Image width in pixels
    int32_t height;             // Image height in pixels
    uint16_t planes;            // Number of color planes (must be 1)
    uint16_t bit_count;         // Bits per pixel (e.g., 8, 24)
    uint32_t compression;       // Compression type (0 = uncompressed)
    uint32_t image_size;        // Image size in bytes (can be 0 for uncompressed)
    int32_t x_pixels_per_m;     // Horizontal resolution (pixels per meter)
    int32_t y_pixels_per_m;     // Vertical resolution (pixels per meter)
    uint32_t colors_used;       // Number of colors in the palette (0 = use default)
    uint32_t colors_important;  // Important colors (0 = all are important)
};
#pragma pack(pop)

// NOLINTBEGIN
std::vector<std::vector<uint8_t>> Utils::ReadBmpGrayscale(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error(std::format("Failed to open file: {}", path));
    }

    BitmapFileHeader file_header{};
    ifs.read(reinterpret_cast<char *>(&file_header), sizeof(file_header));

    if (file_header.type != 0x4D42) {  // Check for "BM" magic number (little-endian)
        throw std::runtime_error(std::format("Not a valid BMP file: {}", path));
    }

    BitmapInfoHeader info_header{};
    ifs.read(reinterpret_cast<char *>(&info_header), sizeof(info_header));

    if (info_header.compression != 0) {
        throw std::runtime_error(std::format("Unsupported compression type in BMP file: {}", path));
    }
    if (info_header.bit_count != 8 && info_header.bit_count != 24) {
        throw std::runtime_error(
            std::format("Unsupported bit depth ({} bpp), only 8-bit grayscale and 24-bit color are supported: {}",
                        info_header.bit_count, path));
    }

    const int32_t width = info_header.width;
    const int32_t height = std::abs(info_header.height);  // Height can be negative for top-down BMP
    const bool top_down = info_header.height < 0;

    // BMP rows are padded to a multiple of 4 bytes
    const int row_padding = (4 - (width * info_header.bit_count / 8) % 4) % 4;
    const int row_size_bytes = (width * info_header.bit_count / 8) + row_padding;

    std::vector<std::vector<uint8_t>> pixel_data(static_cast<size_t>(height),
                                                 std::vector<uint8_t>(static_cast<size_t>(width)));
    std::vector<uint8_t> row_buffer(static_cast<size_t>(row_size_bytes));

    ifs.seekg(file_header.offset, std::ios::beg);

    for (int32_t y = 0; y < height; ++y) {
        ifs.read(reinterpret_cast<char *>(row_buffer.data()), row_size_bytes);
        if (!ifs) {
            throw std::runtime_error(std::format("Error reading pixel data from BMP file: {}", path));
        }

        // BMP stores rows bottom-up unless height is negative
        int target_row = top_down ? y : (height - 1 - y);

        int buffer_idx = 0;
        for (int32_t x = 0; x < width; ++x) {
            if (info_header.bit_count == 8) {
                // 8-bit grayscale (assuming a grayscale palette or direct values)
                pixel_data[static_cast<size_t>(target_row)][static_cast<size_t>(x)] =
                    row_buffer[static_cast<size_t>(buffer_idx++)];
                // Note: This doesn't handle palettes explicitly. It assumes 8-bit BMPs
                // store grayscale intensity directly or have a standard grayscale palette.
            } else {  // 24-bit color (BGR)
                uint8_t b = row_buffer[static_cast<size_t>(buffer_idx++)];
                uint8_t g = row_buffer[static_cast<size_t>(buffer_idx++)];
                uint8_t r = row_buffer[static_cast<size_t>(buffer_idx++)];
                // Convert to grayscale using luminosity formula
                pixel_data[static_cast<size_t>(target_row)][static_cast<size_t>(x)] =
                    static_cast<uint8_t>(std::round((0.299 * r) + (0.587 * g) + (0.114 * b)));
            }
        }
    }

    return pixel_data;
}
// NOLINTEND