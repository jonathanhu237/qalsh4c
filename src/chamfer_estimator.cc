#include "chamfer_estimator.h"

#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "b_plus_tree.h"
#include "utils.h"

namespace qalsh_chamfer {

// ---------- ChamferEstimatorBuilder Implementation ----------
ChamferEstimatorBuilder::ChamferEstimatorBuilder()
    : num_points_(0),
      num_dimensions_(0),
      approximation_ratio_(0.0),
      bucket_width_(0.0),
      beta_(0.0),
      error_probability_(0.0),
      num_hash_tables_(0),
      collision_threshold_(0),
      page_size_(0),
      num_samples_(0),
      verbose_(false) {}

auto ChamferEstimatorBuilder::set_dataset_name(const std::string& dataset_name) -> ChamferEstimatorBuilder& {
    dataset_name_ = dataset_name;
    return *this;
}

auto ChamferEstimatorBuilder::set_parent_directory(const fs::path& parent_directory) -> ChamferEstimatorBuilder& {
    parent_directory_ = parent_directory;
    return *this;
}

auto ChamferEstimatorBuilder::set_num_points(unsigned int num_points) -> ChamferEstimatorBuilder& {
    num_points_ = num_points;
    return *this;
}

auto ChamferEstimatorBuilder::set_num_dimensions(unsigned int num_dimensions) -> ChamferEstimatorBuilder& {
    num_dimensions_ = num_dimensions;
    return *this;
}

auto ChamferEstimatorBuilder::ReadParamFromBinaryFile() -> ChamferEstimatorBuilder& {
    // Read the parameters from the file
    fs::path param_file_path = parent_directory_ / dataset_name_ / "index" / "index_params.bin";
    std::ifstream ifs(param_file_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open parameter file, you may need to run the indexer first.");
    }

    ifs.read(reinterpret_cast<char*>(&approximation_ratio_), sizeof(approximation_ratio_));
    ifs.read(reinterpret_cast<char*>(&bucket_width_), sizeof(bucket_width_));
    ifs.read(reinterpret_cast<char*>(&beta_), sizeof(beta_));
    ifs.read(reinterpret_cast<char*>(&error_probability_), sizeof(error_probability_));
    ifs.read(reinterpret_cast<char*>(&num_hash_tables_), sizeof(num_hash_tables_));
    ifs.read(reinterpret_cast<char*>(&collision_threshold_), sizeof(collision_threshold_));
    ifs.read(reinterpret_cast<char*>(&page_size_), sizeof(page_size_));

    dot_vectors_.resize(num_hash_tables_);
    for (unsigned int i = 0; i < num_hash_tables_; i++) {
        dot_vectors_[i].resize(num_dimensions_);
        ifs.read(reinterpret_cast<char*>(dot_vectors_[i].data()),
                 static_cast<std::streamsize>(sizeof(double) * num_dimensions_));
    }

    return *this;
}

auto ChamferEstimatorBuilder::set_num_samples(unsigned int num_samples) -> ChamferEstimatorBuilder& {
    if (num_samples == 0) {
        // Default to log(N) samples if num_samples is not set
        num_samples_ = static_cast<unsigned int>(std::log(num_points_));
    } else {
        num_samples_ = num_samples;
    }

    return *this;
}

auto ChamferEstimatorBuilder::set_verbose(bool verbose) -> ChamferEstimatorBuilder& {
    verbose_ = verbose;
    return *this;
}

auto ChamferEstimatorBuilder::Build() const -> std::unique_ptr<ChamferEstimator> {
    return std::unique_ptr<ChamferEstimator>(new ChamferEstimator(
        dataset_name_, parent_directory_, num_points_, num_dimensions_, approximation_ratio_, bucket_width_, beta_,
        error_probability_, num_hash_tables_, collision_threshold_, page_size_, dot_vectors_, num_samples_, verbose_));
}

// ---------- ChamferEstimator Implementation ----------
ChamferEstimator::ChamferEstimator(std::string dataset_name, fs::path parent_directory, unsigned int num_points,
                                   unsigned int num_dimensions, double approximation_ratio, double bucket_width,
                                   double beta, double error_probability, unsigned int num_hash_tables,
                                   unsigned int collision_threshold, unsigned int page_size,
                                   std::vector<std::vector<double>> dot_vectors, unsigned int num_samples, bool verbose)
    : dataset_name_(std::move(dataset_name)),
      parent_directory_(std::move(parent_directory)),
      num_points_(num_points),
      num_dimensions_(num_dimensions),
      approximation_ratio_(approximation_ratio),
      bucket_width_(bucket_width),
      beta_(beta),
      error_probability_(error_probability),
      num_hash_tables_(num_hash_tables),
      collision_threshold_(collision_threshold),
      page_size_(page_size),
      dot_vectors_(std::move(dot_vectors)),
      num_samples_(num_samples),
      verbose_(verbose) {}

auto ChamferEstimator::PrintConfiguration() const -> void {
    std::cout << std::format("---------- Chamfer Approximation Configuration ----------\n");
    std::cout << std::format("Dataset Name: {}\n", dataset_name_);
    std::cout << std::format("Parent Directory: {}\n", parent_directory_.string());
    std::cout << std::format("Number of Points: {}\n", num_points_);
    std::cout << std::format("Number of Dimensions: {}\n", num_dimensions_);
    std::cout << std::format("Approximation Ratio: {}\n", approximation_ratio_);
    std::cout << std::format("Bucket Width: {}\n", bucket_width_);
    std::cout << std::format("Beta: {}\n", beta_);
    std::cout << std::format("Error Probability: {}\n", error_probability_);
    std::cout << std::format("Number of Hash Tables: {}\n", num_hash_tables_);
    std::cout << std::format("Collision Threshold: {}\n", collision_threshold_);
    std::cout << std::format("Page Size: {}\n", page_size_);
    std::cout << std::format("Number of Samples: {}\n", num_samples_);
    std::cout << std::format("-----------------------------------------------------\n");
}

auto ChamferEstimator::Execute() const -> void {
    // Read the sets from the files
    fs::path set_A_file_path = parent_directory_ / dataset_name_ / "A.bin";
    std::vector<std::vector<double>> set_A =
        Utils::ReadSetFromFile(set_A_file_path, num_points_, num_dimensions_, "A", verbose_);

    fs::path set_B_file_path = parent_directory_ / dataset_name_ / "B.bin";
    std::vector<std::vector<double>> set_B =
        Utils::ReadSetFromFile(set_B_file_path, num_points_, num_dimensions_, "B", verbose_);

    // Generate the D arrays for both sets
    std::vector<double> D_A = GenerateDArrayForSet(set_A, set_B, "A", "B");
    std::vector<double> D_B = GenerateDArrayForSet(set_B, set_A, "B", "A");

    // Approximate the Chamfer distance
    double chamfer_A_to_B = ApproximateChamferDistance(set_A, set_B, D_A, "A", "B");
    double chamfer_B_to_A = ApproximateChamferDistance(set_B, set_A, D_B, "B", "A");
    double total_approximation = chamfer_A_to_B + chamfer_B_to_A;

    // Read the ground truth
    fs::path ground_truth_file_path = parent_directory_ / dataset_name_ / "gt.bin";
    std::ifstream ifs(ground_truth_file_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open ground truth file.");
    }

    double ground_truth = 0.0;
    ifs.read(reinterpret_cast<char*>(&ground_truth), sizeof(ground_truth));

    // Calculate the relative error
    double relative_error = std::fabs(total_approximation - ground_truth) / ground_truth;
    std::cout << std::format("Relative Error: {:.3f}%\n", relative_error * 100.0);
}

auto ChamferEstimator::ApproximateChamferDistance(const std::vector<std::vector<double>>& from_set,
                                                  const std::vector<std::vector<double>>& to_set,
                                                  const std::vector<double>& from_set_D_array,
                                                  const std::string& from_set_name,
                                                  const std::string& to_set_name) const -> double {
    double approximation = 0.0;
    double sum = std::accumulate(from_set_D_array.begin(), from_set_D_array.end(), 0.0);

    for (unsigned int i = 0; i < num_samples_; i++) {
        if (verbose_) {
            std::cout << std::format("Approximate Chamfer Distance from {} to {}...({}/{})\r", from_set_name,
                                     to_set_name, i + 1, num_samples_)
                      << std::flush;
        }

        unsigned int point_id = Utils::SampleFromWeights(from_set_D_array);
        const auto& point = from_set.at(point_id);
        const double min_distance = Utils::MinDistance(point, to_set);
        approximation += sum * min_distance / from_set_D_array.at(point_id);
    }

    if (verbose_) {
        std::cout << "\n";
    }

    return approximation / num_samples_;
}

auto ChamferEstimator::GenerateDArrayForSet(const std::vector<std::vector<double>>& set_from,
                                            const std::vector<std::vector<double>>& set_to,
                                            const std::string& set_from_name, const std::string& set_to_name) const
    -> std::vector<double> {
    std::vector<double> d_array(num_points_);

    for (unsigned int i = 0; i < num_points_; i++) {
        if (verbose_) {
            std::cout << std::format("Processing point in set {}...({}/{})\r", set_from_name, i + 1, num_points_)
                      << std::flush;
        }

        d_array[i] = CAnnSearch(set_from[i], set_to, set_to_name).first;
    }

    if (verbose_) {
        std::cout << "\n";
    }

    return d_array;
}

auto ChamferEstimator::CAnnSearch(const std::vector<double>& query, const std::vector<std::vector<double>>& dataset,
                                  const std::string& set_name) const -> Candidate {
    std::priority_queue<Candidate, std::vector<Candidate>> candidates;
    std::vector<bool> visited(num_points_, false);
    std::unordered_map<unsigned int, unsigned int> collision_count;
    std::vector<BPlusTreeSearcher> b_plus_tree_searchers;
    std::vector<double> keys(num_hash_tables_);
    double search_radius = 1.0;

    // Initialize the keys
    for (unsigned int i = 0; i < num_hash_tables_; i++) {
        keys[i] = Utils::DotProduct(query, dot_vectors_[i]);
    }

    // Initialize B+ trees
    b_plus_tree_searchers.reserve(num_hash_tables_);
    for (unsigned int j = 0; j < num_hash_tables_; j++) {
        const fs::path index_file_path =
            parent_directory_ / dataset_name_ / "index" / std::format("{}_idx_{}.bin", set_name, j);
        b_plus_tree_searchers.emplace_back(index_file_path, page_size_, keys[j]);
    }

    // c-ANN search
    while (candidates.size() < static_cast<size_t>(std::ceil(beta_ * num_points_))) {
        for (unsigned int j = 0; j < num_hash_tables_; j++) {
            std::vector<unsigned int> point_ids =
                b_plus_tree_searchers[j].IncrementalSearch(bucket_width_ * search_radius / 2.0);

            for (auto point_id : point_ids) {
                if (visited.at(point_id)) {
                    continue;
                }

                collision_count[point_id]++;
                if (collision_count.at(point_id) >= collision_threshold_) {
                    const std::vector<double>& point = dataset[point_id];
                    candidates.emplace(Utils::CalculateManhattan(point, query), point_id);

                    visited.at(point_id) = true;
                }
            }
        }

        if (!candidates.empty() && candidates.top().first <= approximation_ratio_ * search_radius) {
            break;
        }

        search_radius *= approximation_ratio_;
    }

    if (!candidates.empty()) {
        return candidates.top();
    }

    // Defense programming
    return {std::numeric_limits<double>::max(), std::numeric_limits<unsigned int>::max()};
}

};  // namespace qalsh_chamfer