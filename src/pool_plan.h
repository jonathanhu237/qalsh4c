#pragma once

#include <optional>
#include <vector>

// Pilot statistics and sample budgets for pooled two-stage sampling.
//
// A source point a drawn with probability p_a = U_a / U has the ratios x_a = v_a / U_a, z_a = D_a / U_a and
// r_a = v_a / D_a = x_a / z_a, all in (0, 1] once exact target matches are removed, and mu = E_p[x] = S / U. The
// per-draw relative variances of the two stages are
//   lambda0 = (E_p[x^2] - mu^2) / mu^2              (drawing in proportion to U, as the pool and --direct do)
//   lambda1 = (E_p[z] E_p[x^2 / z] - mu^2) / mu^2  (drawing in proportion to D, as --full does)
// and K pool draws followed by T resamples give an estimate with relative variance
//   lambda0 / K + (1 - 1 / K) lambda1 / T.
namespace PoolPlan {

// Ratios observed at one pilot draw taken in proportion to U_a.
struct PilotDraw {
    double x;  // v_a / U_a
    double z;  // D_a / U_a
};

struct Coefficients {
    double lambda0;
    double lambda1;
};

// Point estimates from disjoint pairs of consecutive draws. H0 = (x - x')^2 / 2 and H1 = z z' (r - r')^2 / 2 have
// means mu^2 lambda0 and mu^2 lambda1; both pair means are divided by the squared mean of x. Needs at least two
// draws.
[[nodiscard]] Coefficients EstimateCoefficients(const std::vector<PilotDraw>& draws);

// Delta-method estimates of the relative standard errors of the two coefficient estimates. With pair means
// P_j = (x + x') / 2, mu = mean P_j and nu_i = mean H_ij, the error of lambda_i = nu_i / mu^2 is about
// sd_j[(H_ij - nu_i) / nu_i - 2 (P_j - mu) / mu] / sqrt(L) for L pairs. This is a stability heuristic, not a confidence
// bound; a coefficient whose pairs show no variation at all gets zero.
[[nodiscard]] Coefficients RelativeStandardErrors(const std::vector<PilotDraw>& draws);

// Upper bounds on lambda0 and lambda1 that hold together with probability at least 1 - alpha: a lower confidence
// bound on mu and upper confidence bounds on the pair means of H0 and H1, each at level alpha / 3. Returns nullopt
// when the lower bound on mu is not positive, so the pilot is too small to certify anything.
[[nodiscard]] std::optional<Coefficients> CertifyCoefficients(const std::vector<PilotDraw>& draws, double alpha);

// Two-sided empirical Bernstein radius for the mean of at least two values in [0, range] (Maurer and Pontil,
// 2009): the true mean lies within this distance of the sample mean with probability at least
// 1 - failure_probability.
[[nodiscard]] double EmpiricalBernsteinRadius(const std::vector<double>& values, double range,
                                              double failure_probability);

// Smallest T with lambda0 / K + (1 - 1 / K) lambda1 / T <= eta for a pool of K draws, or nullopt when no T is
// enough because lambda0 / K alone reaches eta.
[[nodiscard]] std::optional<unsigned int> SamplesForPool(Coefficients coefficients, double eta, unsigned int pool_size);

// Pool size from the square-root rule, which minimizes K * qalsh_cost + T * exact_cost subject to the sufficient
// condition lambda0 / K + lambda1 / T <= eta. At least one.
[[nodiscard]] unsigned int SquareRootPoolSize(Coefficients coefficients, double eta, double qalsh_cost,
                                              double exact_cost);

// Draws a one-stage design with per-draw relative variance lambda needs to reach relative variance eta.
[[nodiscard]] unsigned int OneStageSamples(double lambda, double eta);

}  // namespace PoolPlan
