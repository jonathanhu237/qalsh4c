#pragma once

#include <cstddef>
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

// A time-uniform confidence sequence for the mean of i.i.d. values in [0, 1]: the predictable plug-in empirical
// Bernstein sequence of Waudby-Smith and Ramdas (JRSS-B, 2023), kept as a running intersection. With probability at
// least 1 - delta it holds the mean at every count at once, so a rule that decides when to stop by looking at it keeps
// that guarantee at its random stopping time.
class ConfidenceSequence {
   public:
    explicit ConfidenceSequence(double delta);
    void Add(double value);
    [[nodiscard]] size_t Count() const { return count_; }
    [[nodiscard]] double Mean() const;  // Sample mean of the values so far
    [[nodiscard]] double Lower() const { return lower_; }
    [[nodiscard]] double Upper() const { return upper_; }
    // True when every mean in [Lower, Upper] lies within relative error epsilon of Mean(): then, with the sequence
    // holding the true mean, Mean() is within relative error epsilon of it.
    [[nodiscard]] bool CertifiesMean(double epsilon) const;

   private:
    // Bets never exceed this, as the authors recommend; it keeps -log(1 - lambda) finite and the sequence stable.
    static constexpr double kMaxBet = 0.5;

    double log_term_;  // log(2 / delta)
    size_t count_{0};
    double sum_{0.0};
    double sum_squared_deviations_{0.0};  // Sum of (x_i - mu_hat_i)^2
    double previous_mean_{0.5};           // mu_hat_{t-1}, with the prior pseudo-observation 1/2
    double previous_variance_{0.25};      // sigma_hat^2_{t-1}, with the prior pseudo-observation 1/4
    double sum_bets_{0.0};
    double sum_weighted_values_{0.0};
    double sum_penalties_{0.0};
    double lower_{0.0};
    double upper_{1.0};
};

// Pilot moments of one stratum whose draws come in proportion to U_a within the stratum: the mean of x and the
// pair means of H0 and H1. With U_h the stratum's total bound, drawing in proportion to U_a has per-draw variance
// U_h^2 E[H0], querying every point of the stratum and drawing in proportion to D_a has U_h^2 E[H1], and the
// stratum total is U_h E[x].
struct StratumMoments {
    double mean_ratio;
    double h0;
    double h1;
};

// Point estimates from disjoint pairs of consecutive draws, as for EstimateCoefficients. Needs at least two draws.
[[nodiscard]] StratumMoments EstimateStratumMoments(const std::vector<PilotDraw>& draws);

// A lower confidence bound on the mean of x and upper confidence bounds on the pair means of H0 and H1, each at
// level alpha / 3, so the three hold together with probability at least 1 - alpha. Returns nullopt when the lower
// bound on the mean is not positive. Needs at least four draws.
[[nodiscard]] std::optional<StratumMoments> CertifyStratumMoments(const std::vector<PilotDraw>& draws, double alpha);

// Plug-in coefficients of drawing in proportion to U_a over all strata, from pilots drawn within each stratum:
// stratum h has share pi_h = U_h / U of the bound total, and E[f] = sum_h pi_h E_h[f] for x, x^2, z and x^2 / z.
// Differences between the stratum means count toward lambda0 and lambda1, not only the variation within strata.
[[nodiscard]] Coefficients PooledCoefficients(const std::vector<std::vector<PilotDraw>>& strata,
                                              const std::vector<double>& shares);

// Draws per stratum for independent stratum estimates with per-draw standard deviations sigma_h: Neyman allocation
// T_h = sigma_h * sum_g sigma_g / (eta * total^2), rounded up and at least one, so that
// sum_h sigma_h^2 / T_h <= eta * total^2 with the fewest draws up to rounding.
[[nodiscard]] std::vector<unsigned int> StratifiedSamples(const std::vector<double>& sigmas, double eta,
                                                          double total);

}  // namespace PoolPlan
