// Exact nearest-neighbor distance d(a, B) for every source point a, under l1 or l2.
//
// Usage: exact_nn <dataset_dir> <p: 1|2> <out.bin>
// Reads A.bin and B.bin (row-major float32) and the point counts from metadata.json, and writes |A| float64
// distances. Each thread takes a block of source points and streams B once for the whole block, so B is read
// |A| / kBlock times instead of |A| times. Distances accumulate in double per point pair.
//
// Build: g++ -O3 -march=native -fopenmp -std=c++20 exact_nn.cc -o exact_nn

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBlock = 32;

std::vector<float> ReadPoints(const std::string& path, std::size_t n, std::size_t d) {
    std::vector<float> data(n * d);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!in) {
        std::fprintf(stderr, "failed to read %s\n", path.c_str());
        std::exit(1);
    }
    return data;
}

std::size_t JsonInt(const std::string& text, const std::string& pattern) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex(pattern))) {
        std::fprintf(stderr, "metadata field not found: %s\n", pattern.c_str());
        std::exit(1);
    }
    return std::stoull(match[1].str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <dataset_dir> <p: 1|2> <out.bin>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    const int p = std::atoi(argv[2]);
    if (p != 1 && p != 2) {
        std::fprintf(stderr, "p must be 1 or 2\n");
        return 1;
    }
    std::ifstream meta_in(dir + "/metadata.json");
    std::stringstream meta;
    meta << meta_in.rdbuf();
    const std::string text = meta.str();
    const std::size_t d = JsonInt(text, "\"num_dimensions\"\\s*:\\s*(\\d+)");
    const std::size_t na = JsonInt(text, "\"a\"\\s*:\\s*(\\d+)");
    const std::size_t nb = JsonInt(text, "\"b\"\\s*:\\s*(\\d+)");

    const std::vector<float> a = ReadPoints(dir + "/A.bin", na, d);
    const std::vector<float> b = ReadPoints(dir + "/B.bin", nb, d);
    std::vector<double> best(na, std::numeric_limits<double>::infinity());

    const std::size_t num_blocks = (na + kBlock - 1) / kBlock;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t blk = 0; blk < num_blocks; ++blk) {
        const std::size_t lo = blk * kBlock;
        const std::size_t hi = std::min(na, lo + kBlock);
        double local[kBlock];
        std::fill(local, local + kBlock, std::numeric_limits<double>::infinity());
        for (std::size_t j = 0; j < nb; ++j) {
            const float* bj = b.data() + j * d;
            for (std::size_t i = lo; i < hi; ++i) {
                const float* ai = a.data() + i * d;
                double sum = 0.0;
                if (p == 1) {
#pragma omp simd reduction(+ : sum)
                    for (std::size_t k = 0; k < d; ++k) {
                        sum += std::fabs(static_cast<double>(ai[k]) - static_cast<double>(bj[k]));
                    }
                } else {
#pragma omp simd reduction(+ : sum)
                    for (std::size_t k = 0; k < d; ++k) {
                        const double diff = static_cast<double>(ai[k]) - static_cast<double>(bj[k]);
                        sum += diff * diff;
                    }
                }
                if (sum < local[i - lo]) {
                    local[i - lo] = sum;
                }
            }
        }
        for (std::size_t i = lo; i < hi; ++i) {
            best[i] = (p == 1) ? local[i - lo] : std::sqrt(local[i - lo]);
        }
    }

    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(best.data()), static_cast<std::streamsize>(best.size() * sizeof(double)));
    double total = 0.0;
    for (double v : best) {
        total += v;
    }
    std::printf("n_a=%zu n_b=%zu d=%zu p=%d CH=%.6f\n", na, nb, d, p, total);
    return 0;
}
