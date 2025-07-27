#include "weights_generator.h"

#include <filesystem>
#include <vector>

#include "ann_searcher.h"
#include "point_set.h"
#include "types.h"

std::vector<double> UniformWeightsGenerator::Generate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata
    DatasetMetadata metadata;
    metadata.Load(dataset_directory / "metadata.toml");

    std::vector<double> weights(metadata.query_num_points, 1.0);
    return weights;
}

std::vector<double> QalshWeightsGenerator::Generate(const std::filesystem::path& dataset_directory) {
    // Load dataset metadata.
    DatasetMetadata dataset_metadata;
    dataset_metadata.Load(dataset_directory / "metadata.toml");
    auto base_set_reader{PointSetReaderFactory::Create(dataset_directory / "base.bin", dataset_metadata.data_type,
                                                       dataset_metadata.base_num_points,
                                                       dataset_metadata.num_dimensions)};
    auto query_set_reader{PointSetReaderFactory::Create(dataset_directory / "query.bin", dataset_metadata.data_type,
                                                        dataset_metadata.query_num_points,
                                                        dataset_metadata.num_dimensions)};

    // Generate weights based on QALSH algorithm.
    std::vector<double> weights(dataset_metadata.query_num_points);
    QalshAnnSearcher qalsh_ann_searcher;
    qalsh_ann_searcher.Init(dataset_directory / "qalsh_index");
    for (unsigned int i = 0; i < dataset_metadata.query_num_points; i++) {
        PointVariant query_point = query_set_reader->GetPoint(i);
        AnnResult result = qalsh_ann_searcher.Search(query_point);
        weights[i] = result.distance;  // Use the distance as the weight
    }

    return weights;
}