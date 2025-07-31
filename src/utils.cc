#include "utils.h"

double Utils::L1Distance(const Point &pt1, const Point &pt2) {
    if (pt1.size() != pt2.size()) {
        spdlog::error("Vectors must be of the same size");
    }

    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return static_cast<double>((v1 - v2).lpNorm<1>());
}

double Utils::DotProduct(const Point &pt1, const Point &pt2) {
    if (pt1.size() != pt2.size()) {
        spdlog::error("Points must be of the same size for dot product. vec1.size(): {}, vec2.size(): {}", pt1.size(),
                      pt2.size());
    }

    Eigen::Map<const Eigen::VectorXd> v1(pt1.data(), static_cast<Eigen::Index>(pt1.size()));
    Eigen::Map<const Eigen::VectorXd> v2(pt2.data(), static_cast<Eigen::Index>(pt2.size()));

    return v1.dot(v2);
}