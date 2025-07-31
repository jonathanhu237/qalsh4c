#include "command.h"

#include <chrono>
#include <format>
#include <iostream>
#include <ratio>

#include "estimator.h"

EstimateCommand::EstimateCommand(std::unique_ptr<Estimator> estimator, std::filesystem::path dataset_directory,
                                 bool in_memory)
    : estimator_(std::move(estimator)), dataset_directory_(std::move(dataset_directory)), in_memory_(in_memory) {}

void EstimateCommand::Execute() {
    estimator_->set_in_memory(in_memory_);

    auto start = std::chrono::high_resolution_clock::now();
    double estimation = estimator_->Estimate(dataset_directory_);
    auto end = std::chrono::high_resolution_clock::now();

    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory_ / "metadata.json");

    // Output the result.
    std::cout << std::format(
        "The result is as follows:\n"
        "    Time Consumed: {:.2f} ms\n"
        "    Estimated Chamfer distance: {}\n"
        "    Relative error: {:.2f}%\n",
        std::chrono::duration<double, std::milli>(end - start).count(), estimation,
        std::fabs(estimation - dataset_metadata.chamfer_distance) / dataset_metadata.chamfer_distance *
            100);  // NOLINT: readability-magic-numbers
}