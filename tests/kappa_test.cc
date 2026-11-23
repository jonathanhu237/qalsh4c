#include "kappa.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

struct Case {
    const char* name;
    kappa::Parameters parameters;
    double quadrature;           // the same integral by 40-digit tanh-sinh quadrature (mpmath), as a reference
    std::uint64_t millionths;    // the certified bound, as the paper reports it
};

}  // namespace

int main() {
    // The float32 bucket widths and the (m, l) that QALSH4C uses at c = 2 with beta = 1 and delta_Q = 1/e.
    const Case cases[] = {
        {"l1, gamma = c", {1, 2.8284270763397217, 36, 18, 2.0}, 1.4161312537, 1416132},
        {"l2, gamma = c", {2, 2.719111919403076, 17, 12, 2.0}, 1.46820876852, 1468210},
        {"l1, gamma = c^2", {1, 2.8284270763397217, 36, 18, 4.0}, 2.80252954847, 2802531},
        {"l2, gamma = c^2", {2, 2.719111919403076, 17, 12, 4.0}, 2.91850366603, 2918505},
    };
    for (const Case& test : cases) {
        const kappa::Bound bound = kappa::CertifyKappa(test.parameters);
        const std::string name = test.name;
        Check(bound.lower <= test.quadrature && test.quadrature <= bound.upper,
              name + ": the enclosure misses the quadrature value");
        Check(bound.upper - bound.lower < 2.5 * kappa::kKappaSlack, name + ": the enclosure is too wide");
        Check(static_cast<double>(bound.millionths) >= bound.upper * 1e6, name + ": rounding went down");
        Check(bound.millionths == test.millionths,
              name + ": expected " + std::to_string(test.millionths) + ", got " + std::to_string(bound.millionths));
    }

    // Sample sizes of Table 1 (delta = 0.1).
    Check(kappa::SampleSize(1416132, 0.3, 0.1) == 47, "T for l1 at epsilon = 0.3");
    Check(kappa::SampleSize(1416132, 0.1, 0.1) == 417, "T for l1 at epsilon = 0.1");
    Check(kappa::SampleSize(1468210, 0.3, 0.1) == 53, "T for l2 at epsilon = 0.3");
    Check(kappa::SampleSize(1468210, 0.1, 0.1) == 469, "T for l2 at epsilon = 0.1");
    Check(kappa::SampleSize(2802531, 0.1, 0.1) == 1803, "T for l1 with gamma = c^2");
    Check(kappa::SampleSize(2918505, 0.1, 0.1) == 1919, "T for l2 with gamma = c^2");
    Check(kappa::SampleSize(1000000, 0.1, 0.1) == 1, "T when kappa = 1");

    // The integral diverges when l = m (h = 1), and gamma must exceed 1.
    bool rejected = false;
    try {
        kappa::CertifyKappa({1, 2.8284270763397217, 36, 36, 2.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected, "l = m must be rejected");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "kappa_test passed\n";
    return EXIT_SUCCESS;
}
