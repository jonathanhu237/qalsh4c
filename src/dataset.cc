#include "dataset.h"

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

#include "dataset_cuda.h"
#include "global.h"
#include "log.h"
#include "utils.h"

#ifdef QALSH4C_USE_OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#endif

namespace DatasetPrep {
namespace {

constexpr unsigned int kTreviPatchSize = 64;
constexpr unsigned int kMnistImageMagic = 2051;

using Clock = std::chrono::steady_clock;

std::string FormatDuration(double seconds) {
    const auto whole = static_cast<unsigned long long>(seconds);
    const unsigned long long hours = whole / 3600;
    const unsigned long long minutes = (whole % 3600) / 60;
    const unsigned long long secs = whole % 60;
    return std::format("{:02}:{:02}:{:02}", hours, minutes, secs);
}

std::string FormatElapsed(Clock::time_point start) {
    return FormatDuration(std::chrono::duration<double>(Clock::now() - start).count());
}

template <typename... Args>
void LogStage(bool enabled, std::format_string<Args...> fmt, Args&&... args) {
    if (enabled) {
        Log::info(fmt, std::forward<Args>(args)...);
    }
}

std::string PointSetDescription(const std::vector<Point>& points) {
    if (points.empty()) {
        return "0 points";
    }
    return std::format("{} points, {} dimensions", points.size(), points.front().size());
}

class ProgressReporter {
   public:
    ProgressReporter(std::string label, unsigned long long total, bool enabled)
        : label_(std::move(label)),
          total_(total),
          enabled_(enabled),
          interactive_(enabled && ISATTY(FILENO(stderr)) != 0),
          start_(std::chrono::steady_clock::now()),
          last_render_(start_) {}

    ~ProgressReporter() {
        StopHeartbeat();
    }

    void Start() {
        if (!enabled_ || started_) {
            return;
        }
        {
            std::scoped_lock lock(mutex_);
            started_ = true;
            current_ = 0;
            Render(true);
        }
        StartHeartbeat();
    }

    void Set(unsigned long long current) {
        if (!enabled_) {
            return;
        }
        std::scoped_lock lock(mutex_);
        current_ = std::min(current, total_);
        Render(false);
    }

    void Finish() {
        if (!enabled_) {
            return;
        }
        StopHeartbeat();
        std::scoped_lock lock(mutex_);
        current_ = total_;
        Render(true);
        if (interactive_) {
            std::fputc('\n', stderr);
        }
    }

   private:
    void StartHeartbeat() {
        std::scoped_lock lock(heartbeat_mutex_);
        if (heartbeat_thread_.joinable()) {
            return;
        }
        heartbeat_stop_ = false;
        heartbeat_thread_ = std::thread([this]() {
            std::unique_lock heartbeat_lock(heartbeat_mutex_);
            while (!heartbeat_cv_.wait_for(heartbeat_lock, std::chrono::seconds(1),
                                           [this] { return heartbeat_stop_; })) {
                heartbeat_lock.unlock();
                {
                    std::scoped_lock render_lock(mutex_);
                    Render(false);
                }
                heartbeat_lock.lock();
            }
        });
    }

    void StopHeartbeat() {
        {
            std::scoped_lock lock(heartbeat_mutex_);
            if (!heartbeat_thread_.joinable()) {
                return;
            }
            heartbeat_stop_ = true;
        }
        heartbeat_cv_.notify_all();
        heartbeat_thread_.join();
    }

    void Render(bool force) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        const double since_last = std::chrono::duration<double>(now - last_render_).count();
        if (!force && since_last < 0.2) {
            return;
        }

        const double percent = total_ == 0 ? 100.0 : (static_cast<double>(current_) / static_cast<double>(total_)) * 100.0;
        const double rate = elapsed <= 0.0 ? 0.0 : static_cast<double>(current_) / elapsed;

        if (interactive_) {
            last_render_ = now;
            constexpr unsigned int bar_width = 24;
            const auto filled = static_cast<unsigned int>(std::round(percent / 100.0 * bar_width));
            std::string bar(filled, '#');
            bar.append(bar_width - filled, '-');
            std::fprintf(stderr, "\r\033[2K%s [%s] %5.1f%%  %llu/%llu  %.1f/s  elapsed %s", label_.c_str(),
                         bar.c_str(), percent, current_, total_, rate, FormatDuration(elapsed).c_str());
            std::fflush(stderr);
            return;
        }

        const unsigned int bucket = static_cast<unsigned int>(percent / 5.0);
        if (force || bucket > last_logged_bucket_ || since_last >= 5.0) {
            last_render_ = now;
            last_logged_bucket_ = bucket;
            Log::info("{}: {:.1f}% {}/{} {:.1f}/s elapsed {}", label_, percent, current_, total_, rate,
                      FormatDuration(elapsed));
        }
    }

    std::string label_;
    unsigned long long total_;
    bool enabled_;
    bool interactive_;
    bool started_{false};
    unsigned long long current_{0};
    unsigned int last_logged_bucket_{0};
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_render_;
    std::mutex mutex_;
    std::mutex heartbeat_mutex_;
    std::condition_variable heartbeat_cv_;
    std::thread heartbeat_thread_;
    bool heartbeat_stop_{false};
};

std::string Trim(std::string_view value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

std::vector<std::string> SplitCommaSeparated(const std::string& value) {
    std::vector<std::string> parts;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t delimiter = value.find(',', begin);
        const size_t end = delimiter == std::string::npos ? value.size() : delimiter;
        std::string trimmed = Trim(std::string_view(value).substr(begin, end - begin));
        if (trimmed.empty()) {
            Log::error("Ground-truth selectors must be non-empty comma-separated values.");
        }
        parts.push_back(trimmed);
        if (delimiter == std::string::npos) {
            break;
        }
        begin = delimiter + 1;
    }
    return parts;
}

std::string TaskKey(const GroundTruthTask& task) {
    const std::string norm = std::abs(task.norm_order - 1.0) < Global::kEpsilon ? "l1" : "l2";
    return std::format("{}:{}", norm, ChamferDirectionValue(task.direction));
}

std::string GroundTruthDeviceValue(GroundTruthDevice ground_truth_device) {
    switch (ground_truth_device) {
        case GroundTruthDevice::kAuto:
            return "auto";
        case GroundTruthDevice::kCpu:
            return "cpu";
        case GroundTruthDevice::kCuda:
            return "cuda";
    }
    Log::error("Unsupported ground-truth device.");
}

std::optional<double>& MetadataField(DatasetMetadata& metadata, const GroundTruthTask& task) {
    if (std::abs(task.norm_order - 1.0) < Global::kEpsilon) {
        if (task.direction == ChamferDirection::kAb) {
            return metadata.chamfer_distance_l1_ab;
        }
        return metadata.chamfer_distance_l1_ba;
    }
    if (task.direction == ChamferDirection::kAb) {
        return metadata.chamfer_distance_l2_ab;
    }
    return metadata.chamfer_distance_l2_ba;
}

const std::vector<Point>& FromPoints(const std::vector<Point>& a, const std::vector<Point>& b, ChamferDirection direction) {
    return direction == ChamferDirection::kAb ? a : b;
}

const std::vector<Point>& ToPoints(const std::vector<Point>& a, const std::vector<Point>& b, ChamferDirection direction) {
    return direction == ChamferDirection::kAb ? b : a;
}

void ValidatePoints(const std::vector<Point>& points, const std::string& label) {
    if (points.empty()) {
        Log::error("{} has no points.", label);
    }
    const size_t dimensions = points.front().size();
    if (dimensions == 0) {
        Log::error("{} has zero dimensions.", label);
    }
    for (const auto& point : points) {
        if (point.size() != dimensions) {
            Log::error("{} contains inconsistent point dimensions.", label);
        }
    }
}

void ValidatePair(const std::vector<Point>& a, const std::vector<Point>& b) {
    ValidatePoints(a, "Point set A");
    ValidatePoints(b, "Point set B");
    if (a.front().size() != b.front().size()) {
        Log::error("Point set dimensions do not match: A has {}, B has {}.", a.front().size(), b.front().size());
    }
}

std::mt19937 MakeGenerator(std::optional<unsigned int> seed) {
    if (seed.has_value()) {
        return std::mt19937(*seed);
    }
    return Random::SeededGenerator();
}

void PrepareOutputDirectory(const std::filesystem::path& output_directory, bool overwrite) {
    const std::filesystem::path a_path = output_directory / "A.bin";
    const std::filesystem::path b_path = output_directory / "B.bin";
    const std::filesystem::path metadata_path = output_directory / "metadata.json";
    if (!overwrite && (std::filesystem::exists(a_path) || std::filesystem::exists(b_path) ||
                       std::filesystem::exists(metadata_path))) {
        Log::error("Dataset output already exists at {}. Use --overwrite to replace it.", output_directory.string());
    }

    std::filesystem::create_directories(output_directory);
    if (overwrite) {
        std::filesystem::remove(a_path);
        std::filesystem::remove(b_path);
        std::filesystem::remove(metadata_path);
        std::filesystem::remove_all(output_directory / "index");
    }
}

DatasetMetadata MakeMetadata(const std::vector<Point>& a, const std::vector<Point>& b) {
    ValidatePair(a, b);
    DatasetMetadata metadata;
    metadata.num_dimensions = static_cast<unsigned int>(a.front().size());
    metadata.num_points_a = static_cast<unsigned int>(a.size());
    metadata.num_points_b = static_cast<unsigned int>(b.size());
    return metadata;
}

void ComputeTasks(const std::vector<Point>& a, const std::vector<Point>& b, const std::vector<GroundTruthTask>& tasks,
                  DatasetMetadata& metadata, bool show_progress, bool overwrite,
                  GroundTruthDevice ground_truth_device, size_t cuda_batch_size) {
    if (tasks.empty()) {
        LogStage(show_progress, "No ground-truth selectors requested.");
        return;
    }
    for (const GroundTruthTask& task : tasks) {
        std::optional<double>& target = MetadataField(metadata, task);
        if (target.has_value() && !overwrite) {
            Log::error("Ground truth {} already exists. Use --overwrite to recompute it.", TaskKey(task));
        }
        const Clock::time_point start = Clock::now();
        LogStage(show_progress, "Computing ground truth {} with device mode {}.", TaskKey(task),
                 GroundTruthDeviceValue(ground_truth_device));
        const double value =
            ComputeDirectedChamfer(FromPoints(a, b, task.direction), ToPoints(a, b, task.direction), task.norm_order,
                                   std::format("Ground truth {}", TaskKey(task)), show_progress,
                                   ground_truth_device, cuda_batch_size);
        target = value;
        LogStage(show_progress, "Computed ground truth {} = {} in {}.", TaskKey(task), value, FormatElapsed(start));
    }
}

void WritePointSetToFile(const std::filesystem::path& file_path, const std::vector<Point>& points) {
    ValidatePoints(points, file_path.stem().string());
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs.is_open()) {
        Log::error("Failed to open point set file for writing: {}", file_path.string());
    }
    for (const Point& point : points) {
        ofs.write(reinterpret_cast<const char*>(point.data()),
                  static_cast<std::streamsize>(point.size() * sizeof(Coordinate)));
    }
}

void WritePointSetWithStage(const std::filesystem::path& file_path, const std::vector<Point>& points,
                            std::string_view label, bool show_progress) {
    const Clock::time_point start = Clock::now();
    LogStage(show_progress, "Writing {} to {} ({}).", label, file_path.string(), PointSetDescription(points));
    WritePointSetToFile(file_path, points);
    LogStage(show_progress, "Wrote {} in {}.", label, FormatElapsed(start));
}

void WritePreparedDataset(const std::vector<Point>& a, const std::vector<Point>& b, const CommonOutputOptions& options,
                          std::optional<unsigned int> seed_for_order = std::nullopt) {
    (void)seed_for_order;
    std::vector<GroundTruthTask> tasks = ParseGroundTruthTasks(options.ground_truth_specs);
    LogStage(options.show_progress, "Preparing dataset output directory {}.", options.output_directory.string());
    PrepareOutputDirectory(options.output_directory, options.overwrite);
    WritePointSetWithStage(options.output_directory / "A.bin", a, "A.bin", options.show_progress);
    WritePointSetWithStage(options.output_directory / "B.bin", b, "B.bin", options.show_progress);
    DatasetMetadata metadata = MakeMetadata(a, b);
    ComputeTasks(a, b, tasks, metadata, options.show_progress, true, options.ground_truth_device,
                 options.cuda_batch_size);
    const Clock::time_point metadata_start = Clock::now();
    LogStage(options.show_progress, "Writing metadata.json to {}.", (options.output_directory / "metadata.json").string());
    metadata.Save(options.output_directory / "metadata.json");
    LogStage(options.show_progress, "Wrote metadata.json in {}.", FormatElapsed(metadata_start));
}

uint32_t ReadBigEndianU32(std::ifstream& ifs) {
    unsigned char bytes[4]{};
    ifs.read(reinterpret_cast<char*>(bytes), 4);
    if (!ifs) {
        Log::error("Unexpected end of file while reading big-endian integer.");
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

uint16_t ReadLittleEndianU16(std::ifstream& ifs) {
    unsigned char bytes[2]{};
    ifs.read(reinterpret_cast<char*>(bytes), 2);
    if (!ifs) {
        Log::error("Unexpected end of file while reading little-endian integer.");
    }
    const auto value = static_cast<unsigned int>(bytes[0]) | (static_cast<unsigned int>(bytes[1]) << 8U);
    return static_cast<uint16_t>(value);
}

uint32_t ReadLittleEndianU32(std::ifstream& ifs) {
    unsigned char bytes[4]{};
    ifs.read(reinterpret_cast<char*>(bytes), 4);
    if (!ifs) {
        Log::error("Unexpected end of file while reading little-endian integer.");
    }
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

int32_t ReadLittleEndianI32(std::ifstream& ifs) {
    return static_cast<int32_t>(ReadLittleEndianU32(ifs));
}

std::vector<Point> LoadMnistImages(const std::filesystem::path& image_file, bool show_progress) {
    const Clock::time_point start = Clock::now();
    LogStage(show_progress, "Loading MNIST images from {}.", image_file.string());
    std::ifstream ifs(image_file, std::ios::binary);
    if (!ifs.is_open()) {
        Log::error("Failed to open MNIST IDX image file: {}", image_file.string());
    }
    const uint32_t magic = ReadBigEndianU32(ifs);
    if (magic != kMnistImageMagic) {
        Log::error("Unsupported MNIST IDX magic number {} in {}.", magic, image_file.string());
    }
    const uint32_t num_images = ReadBigEndianU32(ifs);
    const uint32_t rows = ReadBigEndianU32(ifs);
    const uint32_t cols = ReadBigEndianU32(ifs);
    if (num_images == 0 || rows == 0 || cols == 0) {
        Log::error("MNIST IDX file has invalid shape: images={}, rows={}, cols={}.", num_images, rows, cols);
    }

    std::vector<Point> points;
    points.reserve(num_images);
    const size_t dimensions = static_cast<size_t>(rows) * cols;
    std::vector<unsigned char> pixels(dimensions);
    for (uint32_t image_index = 0; image_index < num_images; ++image_index) {
        ifs.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
        if (!ifs) {
            Log::error("MNIST IDX file ended before all images were read.");
        }
        Point point;
        point.reserve(dimensions);
        for (unsigned char pixel : pixels) {
            point.push_back(static_cast<double>(pixel));
        }
        points.push_back(std::move(point));
    }
    LogStage(show_progress, "Loaded MNIST images: {} points, {} dimensions, elapsed {}.", points.size(), dimensions,
             FormatElapsed(start));
    return points;
}

std::vector<Point> LoadMnist(const std::filesystem::path& raw_directory, bool show_progress) {
    const Clock::time_point start = Clock::now();
    LogStage(show_progress, "Loading full MNIST image set from {}.", raw_directory.string());
    std::vector<Point> points = LoadMnistImages(raw_directory / "train-images-idx3-ubyte", show_progress);
    std::vector<Point> test_points = LoadMnistImages(raw_directory / "t10k-images-idx3-ubyte", show_progress);
    if (!points.empty() && !test_points.empty() && points.front().size() != test_points.front().size()) {
        Log::error("Cannot append MNIST train and test images with different dimensions.");
    }
    points.reserve(points.size() + test_points.size());
    std::move(test_points.begin(), test_points.end(), std::back_inserter(points));
    LogStage(show_progress, "Loaded full MNIST image set: {}, elapsed {}.", PointSetDescription(points),
             FormatElapsed(start));
    return points;
}

std::vector<Point> LoadP53(const std::filesystem::path& raw_directory, bool show_progress) {
    const std::filesystem::path file_path = raw_directory / "K9.data";
    const Clock::time_point start = Clock::now();
    LogStage(show_progress, "Loading P53 data from {}.", file_path.string());
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        Log::error("Failed to open P53 file: {}", file_path.string());
    }

    std::vector<std::vector<std::string>> candidate_rows;
    std::map<size_t, size_t> feature_width_counts;
    std::string line;
    while (std::getline(ifs, line)) {
        if (Trim(line).empty() || line.find('?') != std::string::npos) {
            continue;
        }
        std::vector<std::string> entries;
        std::stringstream ss(line);
        std::string entry;
        while (std::getline(ss, entry, ',')) {
            entries.push_back(Trim(entry));
        }
        if (entries.size() <= 2) {
            continue;
        }
        const size_t feature_width = entries.size() - 2;
        ++feature_width_counts[feature_width];
        candidate_rows.push_back(std::move(entries));
    }

    if (candidate_rows.empty()) {
        Log::error("P53 source has no complete numeric rows.");
    }

    const auto best_width =
        std::ranges::max_element(feature_width_counts, [](const auto& left, const auto& right) {
            if (left.second == right.second) {
                return left.first < right.first;
            }
            return left.second < right.second;
        });
    const size_t expected_feature_width = best_width->first;

    std::vector<Point> points;
    for (const std::vector<std::string>& entries : candidate_rows) {
        const size_t feature_width = entries.size() - 2;
        if (feature_width != expected_feature_width) {
            Log::warn("Skipping incomplete P53 row with {} feature columns; expected {}.", feature_width,
                      expected_feature_width);
            continue;
        }
        Point point;
        point.reserve(expected_feature_width);
        for (size_t i = 0; i < expected_feature_width; ++i) {
            point.push_back(std::stod(entries[i]));
        }
        points.push_back(std::move(point));
    }
    ValidatePoints(points, "P53 source");
    LogStage(show_progress, "Loaded P53 data: {}, elapsed {}.", PointSetDescription(points), FormatElapsed(start));
    return points;
}

std::vector<Point> LoadFvecsFile(const std::filesystem::path& file_path, bool show_progress) {
    const Clock::time_point start = Clock::now();
    LogStage(show_progress, "Loading fvecs file {}.", file_path.string());
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        Log::error("Failed to open fvecs file: {}", file_path.string());
    }

    std::vector<Point> points;
    std::optional<uint32_t> expected_dimensions;
    while (true) {
        uint32_t dimension = 0;
        ifs.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        if (ifs.eof()) {
            break;
        }
        if (!ifs) {
            Log::error("Failed to read fvecs dimension from {}.", file_path.string());
        }
        if (dimension == 0) {
            Log::error("fvecs vector has zero dimensions in {}.", file_path.string());
        }
        if (!expected_dimensions.has_value()) {
            expected_dimensions = dimension;
        } else if (*expected_dimensions != dimension) {
            Log::error("Inconsistent fvecs dimensions in {}.", file_path.string());
        }

        std::vector<float> values(dimension);
        ifs.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!ifs) {
            Log::error("Unexpected end of fvecs data in {}.", file_path.string());
        }
        Point point;
        point.reserve(dimension);
        for (float value : values) {
            point.push_back(static_cast<double>(value));
        }
        points.push_back(std::move(point));
    }
    ValidatePoints(points, std::format("fvecs source {}", file_path.string()));
    LogStage(show_progress, "Loaded fvecs file {}: {}, elapsed {}.", file_path.filename().string(),
             PointSetDescription(points), FormatElapsed(start));
    return points;
}

struct BmpImage {
    unsigned int width{0};
    unsigned int height{0};
    std::vector<double> grayscale;
};

double RgbToGray(unsigned char red, unsigned char green, unsigned char blue) {
    return (0.299 * static_cast<double>(red)) + (0.587 * static_cast<double>(green)) +
           (0.114 * static_cast<double>(blue));
}

BmpImage LoadBmp(const std::filesystem::path& file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        Log::error("Failed to open BMP file: {}", file_path.string());
    }
    if (ReadLittleEndianU16(ifs) != 0x4D42) {
        Log::error("Unsupported BMP file signature: {}", file_path.string());
    }
    (void)ReadLittleEndianU32(ifs);  // file size
    (void)ReadLittleEndianU16(ifs);  // reserved 1
    (void)ReadLittleEndianU16(ifs);  // reserved 2
    const uint32_t pixel_offset = ReadLittleEndianU32(ifs);
    const uint32_t dib_size = ReadLittleEndianU32(ifs);
    if (dib_size < 40) {
        Log::error("Unsupported BMP DIB header size {} in {}.", dib_size, file_path.string());
    }

    const int32_t width = ReadLittleEndianI32(ifs);
    const int32_t signed_height = ReadLittleEndianI32(ifs);
    const bool top_down = signed_height < 0;
    const unsigned int height = static_cast<unsigned int>(std::abs(signed_height));
    const uint16_t planes = ReadLittleEndianU16(ifs);
    const uint16_t bits_per_pixel = ReadLittleEndianU16(ifs);
    const uint32_t compression = ReadLittleEndianU32(ifs);
    (void)ReadLittleEndianU32(ifs);  // image size
    (void)ReadLittleEndianI32(ifs);  // x pixels per meter
    (void)ReadLittleEndianI32(ifs);  // y pixels per meter
    const uint32_t colors_used = ReadLittleEndianU32(ifs);
    (void)ReadLittleEndianU32(ifs);  // important colors

    if (width <= 0 || height == 0 || planes != 1 || compression != 0 ||
        (bits_per_pixel != 8 && bits_per_pixel != 24)) {
        Log::error("Unsupported BMP encoding in {}.", file_path.string());
    }
    if (dib_size > 40) {
        ifs.seekg(static_cast<std::streamoff>(14 + dib_size), std::ios::beg);
    }

    std::vector<double> palette;
    if (bits_per_pixel == 8) {
        const uint32_t palette_entries = colors_used == 0 ? 256 : colors_used;
        palette.reserve(palette_entries);
        for (uint32_t i = 0; i < palette_entries; ++i) {
            unsigned char entry[4]{};
            ifs.read(reinterpret_cast<char*>(entry), 4);
            if (!ifs) {
                Log::error("Unexpected end of BMP palette in {}.", file_path.string());
            }
            palette.push_back(RgbToGray(entry[2], entry[1], entry[0]));
        }
    }

    ifs.seekg(static_cast<std::streamoff>(pixel_offset), std::ios::beg);
    const auto unsigned_width = static_cast<unsigned int>(width);
    const unsigned int row_stride = ((static_cast<unsigned int>(bits_per_pixel) * unsigned_width + 31U) / 32U) * 4U;
    std::vector<unsigned char> row(row_stride);
    BmpImage image{.width = unsigned_width, .height = height, .grayscale = std::vector<double>(static_cast<size_t>(unsigned_width) * height)};
    for (unsigned int file_row = 0; file_row < height; ++file_row) {
        ifs.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
        if (!ifs) {
            Log::error("Unexpected end of BMP pixel data in {}.", file_path.string());
        }
        const unsigned int y = top_down ? file_row : (height - 1U - file_row);
        for (unsigned int x = 0; x < unsigned_width; ++x) {
            double gray = 0.0;
            if (bits_per_pixel == 24) {
                const size_t offset = static_cast<size_t>(x) * 3U;
                gray = RgbToGray(row[offset + 2], row[offset + 1], row[offset]);
            } else {
                const unsigned char index = row[x];
                gray = index < palette.size() ? palette[index] : static_cast<double>(index);
            }
            image.grayscale[static_cast<size_t>(y) * unsigned_width + x] = gray;
        }
    }
    return image;
}

std::string LowercaseExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return extension;
}

std::vector<Point> LoadTrevi(const std::filesystem::path& raw_directory, bool show_progress) {
    const Clock::time_point start = Clock::now();
    LogStage(show_progress, "Loading Trevi BMP patches from {}.", raw_directory.string());
    std::vector<std::filesystem::path> bmp_files;
    for (const auto& entry : std::filesystem::directory_iterator(raw_directory)) {
        if (entry.is_regular_file() && LowercaseExtension(entry.path()) == ".bmp") {
            bmp_files.push_back(entry.path());
        }
    }
    std::ranges::sort(bmp_files);
    if (bmp_files.empty()) {
        Log::error("No .bmp files found in directory: {}", raw_directory.string());
    }
    LogStage(show_progress, "Found {} Trevi BMP files.", bmp_files.size());

    std::vector<Point> patches;
    size_t file_index = 0;
    ProgressReporter progress("Loading Trevi BMP patches", bmp_files.size(), show_progress);
    progress.Start();
    for (const std::filesystem::path& bmp_file : bmp_files) {
        ++file_index;
        BmpImage image = LoadBmp(bmp_file);
        for (unsigned int y = 0; y + kTreviPatchSize <= image.height; y += kTreviPatchSize) {
            for (unsigned int x = 0; x + kTreviPatchSize <= image.width; x += kTreviPatchSize) {
                Point patch;
                patch.reserve(static_cast<size_t>(kTreviPatchSize) * kTreviPatchSize);
                for (unsigned int dy = 0; dy < kTreviPatchSize; ++dy) {
                    const size_t row_offset = static_cast<size_t>(y + dy) * image.width;
                    for (unsigned int dx = 0; dx < kTreviPatchSize; ++dx) {
                        patch.push_back(image.grayscale[row_offset + x + dx]);
                    }
                }
                patches.push_back(std::move(patch));
            }
        }
        progress.Set(file_index);
    }
    progress.Finish();
    ValidatePoints(patches, "Trevi patches");
    LogStage(show_progress, "Loaded Trevi patches: {}, elapsed {}.", PointSetDescription(patches),
             FormatElapsed(start));
    return patches;
}

std::vector<Point> LoadSourcePoints(const ConvertOptions& options, bool show_progress) {
    switch (options.kind) {
        case DatasetKind::kMnist:
            return LoadMnist(options.raw_directory, show_progress);
        case DatasetKind::kP53:
            return LoadP53(options.raw_directory, show_progress);
        case DatasetKind::kSift:
            return LoadFvecsFile(options.raw_directory / "sift_base.fvecs", show_progress);
        case DatasetKind::kGist:
            return LoadFvecsFile(options.raw_directory / "gist_base.fvecs", show_progress);
        case DatasetKind::kTrevi:
            return LoadTrevi(options.raw_directory, show_progress);
    }
    Log::error("Unsupported dataset kind.");
}

std::pair<std::vector<Point>, std::vector<Point>> SplitPoints(std::vector<Point> points,
                                                              std::optional<unsigned int> num_points_a, bool shuffle,
                                                              std::optional<unsigned int> seed, bool show_progress) {
    const Clock::time_point start = Clock::now();
    LogStage(show_progress, "Preparing source split for {}.", PointSetDescription(points));
    ValidatePoints(points, "Source dataset");
    if (shuffle) {
        const Clock::time_point shuffle_start = Clock::now();
        if (seed.has_value()) {
            LogStage(show_progress, "Shuffling source points with seed {}.", *seed);
        } else {
            LogStage(show_progress, "Shuffling source points with random seed.");
        }
        std::mt19937 gen = MakeGenerator(seed);
        std::ranges::shuffle(points, gen);
        LogStage(show_progress, "Shuffled source points in {}.", FormatElapsed(shuffle_start));
    } else {
        LogStage(show_progress, "Skipping shuffle; preserving source order.");
    }

    const size_t split = num_points_a.has_value() ? *num_points_a : points.size() / 2;
    if (split == 0 || split >= points.size()) {
        Log::error("Number of points in A ({}) must be greater than 0 and less than total points ({}).", split,
                   points.size());
    }

    std::vector<Point> a;
    std::vector<Point> b;
    a.reserve(split);
    b.reserve(points.size() - split);
    std::move(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(split), std::back_inserter(a));
    std::move(points.begin() + static_cast<std::ptrdiff_t>(split), points.end(), std::back_inserter(b));
    LogStage(show_progress, "Split source points into A={} and B={} in {}.", a.size(), b.size(),
             FormatElapsed(start));
    return {std::move(a), std::move(b)};
}

struct GmmParameters {
    Eigen::VectorXd core_mean;
    Eigen::VectorXd outlier_mean;
    Eigen::MatrixXd core_cholesky;
    Eigen::MatrixXd outlier_cholesky;
};

GmmParameters GenerateGmmParameters(unsigned int num_dimensions, double outlier_scale, std::mt19937& gen) {
    std::uniform_real_distribution<double> mean_dist(-512.0, 512.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    Eigen::VectorXd core_mean(num_dimensions);
    Eigen::VectorXd outlier_mean(num_dimensions);
    for (unsigned int i = 0; i < num_dimensions; ++i) {
        core_mean(static_cast<Eigen::Index>(i)) = mean_dist(gen);
        outlier_mean(static_cast<Eigen::Index>(i)) = mean_dist(gen);
    }

    Eigen::MatrixXd random_matrix(num_dimensions, num_dimensions);
    for (unsigned int row = 0; row < num_dimensions; ++row) {
        for (unsigned int col = 0; col < num_dimensions; ++col) {
            random_matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = normal_dist(gen);
        }
    }
    Eigen::MatrixXd core_covariance = (random_matrix * random_matrix.transpose()) +
                                      Eigen::MatrixXd::Identity(num_dimensions, num_dimensions);
    Eigen::LLT<Eigen::MatrixXd> llt(core_covariance);
    if (llt.info() != Eigen::Success) {
        Log::error("Failed to decompose generated GMM covariance.");
    }

    Eigen::MatrixXd core_cholesky = llt.matrixL();
    Eigen::MatrixXd outlier_cholesky = core_cholesky * std::sqrt(outlier_scale);
    return GmmParameters{.core_mean = std::move(core_mean),
                         .outlier_mean = std::move(outlier_mean),
                         .core_cholesky = std::move(core_cholesky),
                         .outlier_cholesky = std::move(outlier_cholesky)};
}

std::vector<Point> SampleGmm(unsigned int num_points, unsigned int num_dimensions, double outlier_fraction,
                             double outlier_scale, std::mt19937& gen) {
    if (num_points == 0 || num_dimensions == 0) {
        Log::error("GMM point count and dimensions must be positive.");
    }
    if (outlier_fraction < 0.0 || outlier_fraction > 1.0) {
        Log::error("GMM outlier fraction must be in [0, 1].");
    }
    if (outlier_scale <= 0.0) {
        Log::error("GMM outlier scale must be positive.");
    }

    GmmParameters params = GenerateGmmParameters(num_dimensions, outlier_scale, gen);
    std::bernoulli_distribution outlier_dist(outlier_fraction);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    std::vector<Point> points;
    points.reserve(num_points);
    for (unsigned int point_index = 0; point_index < num_points; ++point_index) {
        const bool is_outlier = outlier_dist(gen);
        Eigen::VectorXd z(num_dimensions);
        for (unsigned int dim = 0; dim < num_dimensions; ++dim) {
            z(static_cast<Eigen::Index>(dim)) = normal_dist(gen);
        }
        Eigen::VectorXd sample =
            (is_outlier ? params.outlier_mean : params.core_mean) +
            ((is_outlier ? params.outlier_cholesky : params.core_cholesky) * z);

        Point point;
        point.reserve(num_dimensions);
        for (unsigned int dim = 0; dim < num_dimensions; ++dim) {
            point.push_back(sample(static_cast<Eigen::Index>(dim)));
        }
        points.push_back(std::move(point));
    }
    return points;
}

}  // namespace

GroundTruthDevice ParseGroundTruthDevice(std::string_view value) {
    const std::string trimmed = Trim(value);
    if (trimmed == "auto") {
        return GroundTruthDevice::kAuto;
    }
    if (trimmed == "cpu") {
        return GroundTruthDevice::kCpu;
    }
    if (trimmed == "cuda") {
        return GroundTruthDevice::kCuda;
    }
    Log::error("Unsupported ground-truth device '{}'. Use one of: auto,cpu,cuda.", trimmed);
}

std::vector<GroundTruthTask> ParseGroundTruthTasks(const std::string& specs) {
    std::vector<GroundTruthTask> tasks;
    std::set<std::string> seen;
    if (specs.empty()) {
        return tasks;
    }
    if (!specs.empty() && Trim(specs).empty()) {
        Log::error("Ground-truth selectors must be non-empty comma-separated values.");
    }
    for (const std::string& spec : SplitCommaSeparated(specs)) {
        GroundTruthTask task;
        if (spec == "l1:ab") {
            task = GroundTruthTask{.norm_order = 1.0, .direction = ChamferDirection::kAb};
        } else if (spec == "l1:ba") {
            task = GroundTruthTask{.norm_order = 1.0, .direction = ChamferDirection::kBa};
        } else if (spec == "l2:ab") {
            task = GroundTruthTask{.norm_order = 2.0, .direction = ChamferDirection::kAb};
        } else if (spec == "l2:ba") {
            task = GroundTruthTask{.norm_order = 2.0, .direction = ChamferDirection::kBa};
        } else {
            Log::error("Unsupported ground-truth selector '{}'. Use comma-separated values from l1:ab,l1:ba,l2:ab,l2:ba.",
                       spec);
        }

        if (seen.insert(TaskKey(task)).second) {
            tasks.push_back(task);
        }
    }
    return tasks;
}

void WritePointSet(const std::filesystem::path& file_path, const std::vector<Point>& points) {
    WritePointSetToFile(file_path, points);
}

double ComputeDirectedChamfer(const std::vector<Point>& from, const std::vector<Point>& to, double norm_order,
                              const std::string& label, bool show_progress,
                              GroundTruthDevice ground_truth_device, size_t cuda_batch_size) {
    ValidatePair(from, to);

    auto compute_cpu = [&]() {
        LogStage(show_progress, "{}: running CPU exact scan.", label);
        ProgressReporter progress(label, from.size(), show_progress);
        progress.Start();
        std::vector<double> distances(from.size(), 0.0);
        std::atomic<unsigned long long> completed{0};

#ifdef QALSH4C_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
        for (long long source_index = 0; source_index < static_cast<long long>(from.size()); ++source_index) {
            const Point& source = from[static_cast<size_t>(source_index)];
            double min_distance = std::numeric_limits<double>::max();
            for (const Point& target : to) {
                min_distance = std::min(min_distance, Calc::LpDistance(source, target, norm_order));
            }
            distances[static_cast<size_t>(source_index)] = min_distance;
            const unsigned long long done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % 128 == 0 || done == from.size()) {
                progress.Set(done);
            }
        }
#else
        for (size_t source_index = 0; source_index < from.size(); ++source_index) {
            double min_distance = std::numeric_limits<double>::max();
            for (const Point& target : to) {
                min_distance = std::min(min_distance, Calc::LpDistance(from[source_index], target, norm_order));
            }
            distances[source_index] = min_distance;
            progress.Set(source_index + 1);
        }
#endif

        progress.Finish();
        return std::accumulate(distances.begin(), distances.end(), 0.0);
    };

    if (ground_truth_device == GroundTruthDevice::kCpu) {
        return compute_cpu();
    }

    const std::string cuda_label = std::format("{} ({})", label, GroundTruthDeviceValue(GroundTruthDevice::kCuda));
    ProgressReporter cuda_progress(cuda_label, from.size(), show_progress);
    CudaChamferOptions cuda_options;
    cuda_options.max_source_batch_points = cuda_batch_size;
    cuda_options.stage_callback = [&](std::string_view message) {
        LogStage(show_progress, "{}: {}.", cuda_label, message);
    };
    cuda_options.progress_start_callback = [&]() { cuda_progress.Start(); };
    CudaChamferResult cuda_result =
        ComputeDirectedChamferCuda(from, to, norm_order, [&](size_t completed) { cuda_progress.Set(completed); },
                                   cuda_options);

    if (cuda_result.status == CudaChamferStatus::kSuccess) {
        cuda_progress.Finish();
        return cuda_result.value;
    }

    if (cuda_result.status == CudaChamferStatus::kFailed) {
        Log::error("CUDA ground-truth computation failed: {}", cuda_result.message);
    }
    if (ground_truth_device == GroundTruthDevice::kCuda) {
        Log::error("CUDA ground-truth computation required but unavailable: {}", cuda_result.message);
    }

    Log::warn("CUDA ground-truth computation unavailable ({}); falling back to CPU.", cuda_result.message);
    return compute_cpu();
}

void GenerateGmmDataset(const GmmOptions& options) {
    const Clock::time_point total_start = Clock::now();
    LogStage(options.output.show_progress,
             "Generating GMM dataset: A={} points, B={} points, dimensions={}, outlier_fraction={}, outlier_scale={}.",
             options.num_points_a, options.num_points_b, options.num_dimensions, options.outlier_fraction,
             options.outlier_scale);
    std::mt19937 gen = MakeGenerator(options.seed);
    const Clock::time_point a_start = Clock::now();
    LogStage(options.output.show_progress, "Sampling GMM point set A.");
    std::vector<Point> a = SampleGmm(options.num_points_a, options.num_dimensions, options.outlier_fraction,
                                     options.outlier_scale, gen);
    LogStage(options.output.show_progress, "Sampled GMM point set A: {}, elapsed {}.", PointSetDescription(a),
             FormatElapsed(a_start));
    const Clock::time_point b_start = Clock::now();
    LogStage(options.output.show_progress, "Sampling GMM point set B.");
    std::vector<Point> b = SampleGmm(options.num_points_b, options.num_dimensions, options.outlier_fraction,
                                     options.outlier_scale, gen);
    LogStage(options.output.show_progress, "Sampled GMM point set B: {}, elapsed {}.", PointSetDescription(b),
             FormatElapsed(b_start));
    WritePreparedDataset(a, b, options.output);
    LogStage(options.output.show_progress, "Generated GMM dataset in {}.", FormatElapsed(total_start));
}

void ConvertDataset(const ConvertOptions& options) {
    const Clock::time_point total_start = Clock::now();
    LogStage(options.output.show_progress, "Converting raw dataset to {}.", options.output.output_directory.string());
    std::vector<Point> source_points = LoadSourcePoints(options, options.output.show_progress);
    auto [a, b] =
        SplitPoints(std::move(source_points), options.num_points_a, options.shuffle, options.seed,
                    options.output.show_progress);
    WritePreparedDataset(a, b, options.output);
    LogStage(options.output.show_progress, "Converted raw dataset in {}.", FormatElapsed(total_start));
}

void ComputeDatasetGroundTruth(const GroundTruthOptions& options) {
    if (options.ground_truth_specs.empty()) {
        Log::error("--ground-truth is required for dataset ground-truth.");
    }
    const Clock::time_point total_start = Clock::now();
    LogStage(options.show_progress, "Loading prepared dataset from {}.", options.dataset_directory.string());
    DatasetMetadata metadata;
    metadata.Load(options.dataset_directory / "metadata.json");
    const Clock::time_point load_a_start = Clock::now();
    LogStage(options.show_progress, "Loading A.bin ({} points, {} dimensions).", metadata.num_points_a,
             metadata.num_dimensions);
    std::vector<Point> a = PointSet::LoadPointsFromFile(options.dataset_directory / "A.bin", metadata.num_points_a,
                                                        metadata.num_dimensions);
    LogStage(options.show_progress, "Loaded A.bin in {}.", FormatElapsed(load_a_start));
    const Clock::time_point load_b_start = Clock::now();
    LogStage(options.show_progress, "Loading B.bin ({} points, {} dimensions).", metadata.num_points_b,
             metadata.num_dimensions);
    std::vector<Point> b = PointSet::LoadPointsFromFile(options.dataset_directory / "B.bin", metadata.num_points_b,
                                                        metadata.num_dimensions);
    LogStage(options.show_progress, "Loaded B.bin in {}.", FormatElapsed(load_b_start));
    ComputeTasks(a, b, ParseGroundTruthTasks(options.ground_truth_specs), metadata, options.show_progress,
                 options.overwrite, options.ground_truth_device, options.cuda_batch_size);
    const Clock::time_point metadata_start = Clock::now();
    LogStage(options.show_progress, "Writing updated metadata.json.");
    metadata.Save(options.dataset_directory / "metadata.json");
    LogStage(options.show_progress, "Wrote updated metadata.json in {}.", FormatElapsed(metadata_start));
    LogStage(options.show_progress, "Computed dataset ground truth in {}.", FormatElapsed(total_start));
}

}  // namespace DatasetPrep
