// ExactScan must return what an ascending scalar scan returns, for L1 and L2, including where the L2 expansion
// ||x||^2 + ||y||^2 - 2 <x, y> loses most of its precision and where many targets tie.
#include "exact_scan.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ann_searcher.h"
#include "global.h"
#include "utils.h"

namespace {

constexpr Distance kInfinity = std::numeric_limits<Distance>::infinity();

void Check(bool value, const std::string& message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

Distance ExactDistance(const Point& target, const Point& query, float norm_order) {
    return Calc::LpDistanceBoundedRaw(target.data(), query.data(), query.size(), norm_order, kInfinity);
}

// The scalar scan of the disk path: ascending ids, strict improvements only.
AnnResult ScalarNearest(const std::vector<Point>& targets, const Point& query, float norm_order, Distance upper_bound) {
    AnnResult best{.distance = upper_bound, .point_id = 0};
    for (unsigned int point_id = 0; point_id < targets.size(); ++point_id) {
        const Distance distance = Calc::LpDistanceBoundedRaw(targets[point_id].data(), query.data(), query.size(),
                                                             norm_order, best.distance);
        if (distance < best.distance) {
            best = AnnResult{.distance = distance, .point_id = point_id};
        }
    }
    return best;
}

// Distances must match bit for bit; ids may differ only between targets at that same distance.
void CheckResults(const std::string& name, const std::vector<Point>& targets, const std::vector<Point>& queries,
                  float norm_order, const std::vector<Distance>& upper_bounds, const std::vector<AnnResult>& results) {
    Check(results.size() == queries.size(), name + ": result count");
    for (size_t index = 0; index < queries.size(); ++index) {
        const AnnResult expected = ScalarNearest(targets, queries[index], norm_order, upper_bounds[index]);
        const AnnResult& actual = results[index];
        const std::string where = name + ", query " + std::to_string(index);
        Check(actual.distance == expected.distance,
              where + ": distance " + std::to_string(actual.distance) + " instead of " +
                  std::to_string(expected.distance));
        if (actual.point_id != expected.point_id) {
            Check(ExactDistance(targets[actual.point_id], queries[index], norm_order) == actual.distance,
                  where + ": point " + std::to_string(actual.point_id) + " is not at the reported distance");
        }
    }
}

void CheckCase(const std::string& case_name, const std::vector<Point>& targets, const std::vector<Point>& queries,
               const std::vector<Distance>& upper_bounds) {
    const unsigned int dimensions = queries.empty() ? 1 : static_cast<unsigned int>(queries.front().size());
    std::vector<const Coordinate*> rows;
    for (const Point& query : queries) {
        rows.push_back(query.data());
    }
    std::vector<Coordinate> flat;
    for (const Point& target : targets) {
        flat.insert(flat.end(), target.begin(), target.end());
    }

    for (const float norm_order : {1.0F, 2.0F}) {
        const std::string name = case_name + " L" + std::to_string(static_cast<int>(norm_order));
        ExactScan whole(rows, dimensions, norm_order, upper_bounds);
        whole.ScanPoints(targets);
        CheckResults(name + " (in memory)", targets, queries, norm_order, upper_bounds, whole.Results());

        // The disk path feeds contiguous blocks of any size.
        ExactScan blocks(rows, dimensions, norm_order, upper_bounds);
        const auto num_targets = static_cast<unsigned int>(targets.size());
        unsigned int begin = 0;
        for (const unsigned int size : {1U, 7U, 50U, 3U}) {
            const unsigned int count = std::min(size, num_targets - begin);
            blocks.ScanBlock(flat.data() + static_cast<size_t>(begin) * dimensions, count, begin);
            begin += count;
        }
        blocks.ScanBlock(flat.data() + static_cast<size_t>(begin) * dimensions, num_targets - begin, begin);
        CheckResults(name + " (blocks)", targets, queries, norm_order, upper_bounds, blocks.Results());
    }
}

std::vector<Point> Gaussian(std::mt19937& gen, unsigned int count, unsigned int dimensions, float mean, float sigma) {
    std::normal_distribution<float> normal(mean, sigma);
    std::vector<Point> points(count, Point(dimensions));
    for (Point& point : points) {
        for (Coordinate& value : point) {
            value = normal(gen);
        }
    }
    return points;
}

std::vector<Distance> Unbounded(size_t count) { return std::vector<Distance>(count, kInfinity); }

}  // namespace

int main() {
    try {
        std::mt19937 gen(7);

        // 41 queries leave a partial group of queries in the L1 pass.
        for (const unsigned int dimensions : {1U, 3U, 16U, 17U, 100U}) {
            const std::vector<Point> targets = Gaussian(gen, 600, dimensions, 0.0F, 1.0F);
            const std::vector<Point> queries = Gaussian(gen, 41, dimensions, 0.0F, 1.0F);
            CheckCase("gaussian d=" + std::to_string(dimensions), targets, queries, Unbounded(queries.size()));
        }

        // Two clusters far from the queries' mean: the L2 expansion cancels almost every digit, so the shortlist
        // must keep whole clusters and still pick the exact winner.
        {
            std::vector<Point> targets = Gaussian(gen, 300, 32, 0.0F, 0.05F);
            const std::vector<Point> far_targets = Gaussian(gen, 300, 32, 8192.0F, 0.05F);
            targets.insert(targets.end(), far_targets.begin(), far_targets.end());
            std::vector<Point> queries = Gaussian(gen, 20, 32, 0.0F, 0.05F);
            const std::vector<Point> far_queries = Gaussian(gen, 20, 32, 8192.0F, 0.05F);
            queries.insert(queries.end(), far_queries.begin(), far_queries.end());
            CheckCase("two far clusters", targets, queries, Unbounded(queries.size()));
        }

        // Several targets within float rounding of each other around every query.
        {
            const std::vector<Point> bases = Gaussian(gen, 40, 24, 1000.0F, 50.0F);
            std::uniform_real_distribution<float> jitter(-1e-3F, 1e-3F);
            std::vector<Point> targets;
            for (int copy = 0; copy < 5; ++copy) {
                for (Point point : bases) {
                    for (Coordinate& value : point) {
                        value += jitter(gen);
                    }
                    targets.push_back(point);
                }
            }
            std::vector<Point> queries = bases;
            for (Point& query : queries) {
                query[0] += 2e-3F;
            }
            CheckCase("near duplicates", targets, queries, Unbounded(queries.size()));
        }

        // Repeated rows and queries that coincide with targets: zero distances and exact ties.
        {
            const std::vector<Point> rows = Gaussian(gen, 50, 10, 0.0F, 3.0F);
            std::vector<Point> targets;
            for (int copy = 0; copy < 3; ++copy) {
                targets.insert(targets.end(), rows.rbegin(), rows.rend());
            }
            std::vector<Point> queries(rows.begin(), rows.begin() + 10);
            const std::vector<Point> others = Gaussian(gen, 10, 10, 0.0F, 3.0F);
            queries.insert(queries.end(), others.begin(), others.end());
            CheckCase("exact duplicates", targets, queries, Unbounded(queries.size()));
        }

        // Integer pixels: many equal L1 sums, and float sums of squares beyond 2^24 that round.
        {
            std::uniform_int_distribution<int> pixel(0, 255);
            std::vector<Point> targets(300, Point(784));
            std::vector<Point> queries(30, Point(784));
            for (std::vector<Point>* points : {&targets, &queries}) {
                for (Point& point : *points) {
                    for (Coordinate& value : point) {
                        value = static_cast<Coordinate>(pixel(gen) < 128 ? 0 : pixel(gen));
                    }
                }
            }
            CheckCase("pixels", targets, queries, Unbounded(queries.size()));
        }

        // Upper bounds: none, exactly the nearest distance, below it, and above it.
        {
            const std::vector<Point> targets = Gaussian(gen, 400, 17, 0.0F, 1.0F);
            const std::vector<Point> queries = Gaussian(gen, 40, 17, 0.0F, 1.0F);
            for (const float norm_order : {1.0F, 2.0F}) {
                std::vector<Distance> upper_bounds(queries.size());
                for (size_t index = 0; index < queries.size(); ++index) {
                    const Distance nearest = ScalarNearest(targets, queries[index], norm_order, kInfinity).distance;
                    const Distance options[] = {kInfinity, nearest, nearest / 2.0F, nearest * 2.0F};
                    upper_bounds[index] = options[index % 4];
                }
                std::vector<const Coordinate*> rows;
                for (const Point& query : queries) {
                    rows.push_back(query.data());
                }
                ExactScan scan(rows, 17, norm_order, upper_bounds);
                scan.ScanPoints(targets);
                // A bound is a hint: within it the result is the nearest target, beyond it no target.
                const std::vector<AnnResult> results = scan.Results();
                const std::string name = "upper bounds L" + std::to_string(static_cast<int>(norm_order));
                for (size_t index = 0; index < queries.size(); ++index) {
                    const AnnResult nearest = ScalarNearest(targets, queries[index], norm_order, kInfinity);
                    const std::string where = name + ", query " + std::to_string(index);
                    if (nearest.distance <= upper_bounds[index]) {
                        Check(results[index].point_id != ExactScan::kNoTarget &&
                                  results[index].distance == nearest.distance,
                              where + ": the nearest target lies within the bound");
                        Check(ExactDistance(targets[results[index].point_id], queries[index], norm_order) ==
                                  results[index].distance,
                              where + ": the reported target is not at the reported distance");
                    } else {
                        Check(results[index].point_id == ExactScan::kNoTarget,
                              where + ": no target lies within the bound");
                    }
                }
            }
        }

        // More queries than one pass holds, a single query, and no targets.
        {
            const std::vector<Point> targets = Gaussian(gen, 700, 8, 0.0F, 1.0F);
            const std::vector<Point> queries = Gaussian(gen, 1103, 8, 0.0F, 1.0F);
            CheckCase("many queries", targets, queries, Unbounded(queries.size()));
            const std::vector<Point> single(queries.begin(), queries.begin() + 1);
            CheckCase("single query", targets, single, Unbounded(1));
            // With no targets, a bound stays a hint: the result reaches no target.
            {
                std::vector<const Coordinate*> rows{single.front().data()};
                for (const float norm_order : {1.0F, 2.0F}) {
                    ExactScan scan(rows, 8, norm_order, std::vector<Distance>{5.0F});
                    scan.ScanPoints({});
                    Check(scan.Results().front().point_id == ExactScan::kNoTarget, "no targets: no target reached");
                }
            }
        }

        // LinearScanAnnSearcher uses the batched scan for L1 and L2 unless the scalar ablation is selected.
        {
            auto targets = std::make_shared<std::vector<Point>>(Gaussian(gen, 500, 20, 0.0F, 1.0F));
            const std::vector<Point> queries = Gaussian(gen, 30, 20, 0.0F, 1.0F);
            const std::vector<Distance> upper_bounds = Unbounded(queries.size());
            for (const float norm_order : {1.0F, 2.0F}) {
                LinearScanAnnSearcher searcher;
                searcher.InitLoaded(targets, 20, norm_order);
                Check(searcher.UsesBatchedScan(), "L1 and L2 searches should use the batched scan");
                CheckResults("searcher", *targets, queries, norm_order, upper_bounds,
                             searcher.SearchBatched(queries, upper_bounds));
                Global::kUseScalarExactScan = true;
                Check(!searcher.UsesBatchedScan(), "--scalar-exact-scan should select the scalar kernel");
                Global::kUseScalarExactScan = false;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
