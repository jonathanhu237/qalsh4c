#include <qalsh/qalsh.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("collision counter regression: " + message);
}

template <typename Callable>
void CheckInvalid(Callable&& callable) {
    bool rejected = false;
    try {
        callable();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected, "out-of-range point ID was accepted");
}

// The oracle counts upward in full-width integers and tracks acceptance
// separately. It does not share the production countdown or its sentinel.
class CountUpReference final : public qalsh::SearchStrategy {
   public:
    explicit CountUpReference(std::uint32_t threshold_override = 0)
        : threshold_override_(threshold_override) {}

    void set_candidate_rule(qalsh::CandidateRule rule) { rule_ = std::move(rule); }
    void set_stop_after(std::size_t limit) { stop_after_ = limit; }

    void start(const qalsh::QueryStart& query) override {
        threshold_ = threshold_override_ == 0U ? query.collision_threshold : threshold_override_;
        observed_.assign(query.num_points, 0U);
        accepted_.assign(query.num_points, false);
        finished_ = false;
    }

    qalsh::StrategyAction next(const qalsh::QuerySnapshot&) override {
        return finished_ ? qalsh::StrategyAction::Finish() : qalsh::StrategyAction::Scan(0U);
    }

    qalsh::HitDecision on_projection_hit(const qalsh::ProjectionHit& hit,
                                         const qalsh::QuerySnapshot&) override {
        if (finished_) return {};
        if (hit.point_id >= observed_.size()) throw std::invalid_argument("reference point ID");
        if (accepted_[hit.point_id]) return {};
        auto& count = observed_[hit.point_id];
        Check(count < std::numeric_limits<std::uint32_t>::max(), "reference counter overflow");
        ++count;
        const bool accept = rule_ ? rule_({hit, count}) : count >= threshold_;
        if (!accept) return {};
        accepted_[hit.point_id] = true;
        return {.candidate = hit.point_id};
    }

    void on_evaluation(const qalsh::EvaluationEvent&, const qalsh::QuerySnapshot& state) override {
        finished_ = state.evaluated_candidates >= stop_after_;
    }
    void on_scan_boundary(const qalsh::ScanBoundaryEvent&, const qalsh::QuerySnapshot&) override {}
    void on_radius_advanced(const qalsh::QuerySnapshot&) override {}

   private:
    std::uint32_t threshold_override_{0};
    std::uint32_t threshold_{0};
    qalsh::CandidateRule rule_;
    std::size_t stop_after_{std::numeric_limits<std::size_t>::max()};
    std::vector<std::uint32_t> observed_;
    std::vector<bool> accepted_;
    bool finished_{false};
};

using RuleEvent = std::tuple<std::uint32_t, qalsh::PointId, float, float, float, std::uint32_t>;
using Evaluation = std::tuple<qalsh::PointId, std::size_t, std::size_t>;
using StopEvent = std::tuple<std::size_t, std::size_t>;

qalsh::CandidateRule CustomRule(std::vector<RuleEvent>& trace) {
    return [&trace](const qalsh::CollisionObservation& observation) {
        const auto& hit = observation.hit;
        trace.emplace_back(hit.table_id, hit.point_id, hit.projected_value, hit.query_value,
                           hit.absolute_difference, observation.collision_count);
        // Even when index metadata has threshold 1, custom rules must observe
        // counts above the compact representation's maximum without wrapping.
        return observation.collision_count >= 260U + hit.point_id;
    };
}

qalsh::QueryStart Query(std::uint32_t threshold, std::uint32_t points) {
    qalsh::QueryStart query;
    query.num_points = points;
    query.num_dimensions = 1U;
    query.num_hash_tables = std::max(65536U, threshold);
    query.k = 1U;
    query.collision_threshold = threshold;
    return query;
}

void Configure(qalsh::DefaultQalshStrategy& strategy, bool custom,
                std::vector<RuleEvent>& rule_trace, std::vector<StopEvent>& stop_trace,
                std::size_t stop_after = std::numeric_limits<std::size_t>::max()) {
    strategy.set_candidate_rule(custom ? CustomRule(rule_trace) : qalsh::CandidateRule{});
    strategy.set_candidate_budget(std::nullopt);
    strategy.set_termination_rule([&stop_trace, stop_after](const qalsh::QuerySnapshot& state) {
        stop_trace.emplace_back(state.evaluated_candidates, state.projection_hits);
        return state.evaluated_candidates >= stop_after;
    });
}

void CompareDecision(const qalsh::HitDecision& actual, const qalsh::HitDecision& expected) {
    Check(actual.has_candidate() == expected.has_candidate(), "acceptance differs from count-up reference");
    if (actual.has_candidate()) {
        Check(actual.candidate == expected.candidate, "accepted point ID differs from reference");
    }
}

void CheckDirectCalls(std::uint32_t threshold_override) {
    qalsh::DefaultQalshStrategy typed(threshold_override), erased(threshold_override);
    qalsh::SearchStrategy& erased_base = erased;
    CountUpReference reference(threshold_override);
    struct Scenario {
        std::uint32_t threshold;
        std::uint32_t points;
        bool custom;
    };
    const std::array scenarios{Scenario{1U, 7U, false}, Scenario{255U, 3U, false},
                              Scenario{256U, 5U, false}, Scenario{65536U, 2U, false},
                              Scenario{1U, 4U, true}, Scenario{1U, 1U, false},
                              Scenario{255U, 6U, false}, Scenario{256U, 2U, false}};
    for (const auto& scenario : scenarios) {
        std::vector<RuleEvent> typed_rules, erased_rules, reference_rules;
        std::vector<StopEvent> typed_stops, erased_stops;
        Configure(typed, scenario.custom, typed_rules, typed_stops);
        Configure(erased, scenario.custom, erased_rules, erased_stops);
        reference.set_candidate_rule(scenario.custom ? CustomRule(reference_rules) : qalsh::CandidateRule{});
        const auto query = Query(scenario.threshold, scenario.points);
        typed.start(query);
        erased_base.start(query);
        reference.start(query);
        qalsh::QuerySnapshot state;
        const auto invalid_hit = qalsh::ProjectionHit{.point_id = scenario.points};
        CheckInvalid([&] { (void)typed.on_projection_hit(invalid_hit, state); });
        CheckInvalid([&] { (void)erased_base.on_projection_hit(invalid_hit, state); });
        const auto maximum_id = qalsh::ProjectionHit{.point_id = std::numeric_limits<qalsh::PointId>::max()};
        CheckInvalid([&] { (void)typed.on_projection_hit(maximum_id, state); });
        CheckInvalid([&] { (void)erased_base.on_projection_hit(maximum_id, state); });
        const auto threshold = threshold_override == 0U ? scenario.threshold : threshold_override;
        const auto rounds = scenario.custom ? 264U + scenario.points : threshold + 4U;
        std::size_t accepted = 0U;
        for (std::uint32_t round = 0; round < rounds; ++round) {
            for (std::uint32_t offset = 0; offset < scenario.points; ++offset) {
                const auto id = round % 2U == 0U ? offset : scenario.points - offset - 1U;
                const qalsh::ProjectionHit hit{.table_id = round % query.num_hash_tables,
                                                .point_id = id,
                                                .projected_value = static_cast<float>(id),
                                                .query_value = -0.5F,
                                                .absolute_difference = static_cast<float>(id) + 0.5F};
                ++state.projection_hits;
                const auto expected = reference.on_projection_hit(hit, state);
                CompareDecision(typed.on_projection_hit(hit, state), expected);
                CompareDecision(erased_base.on_projection_hit(hit, state), expected);
                accepted += expected.has_candidate() ? 1U : 0U;
            }
        }
        Check(accepted == scenario.points, "a point was lost or accepted twice");
        Check(typed_rules == reference_rules && erased_rules == reference_rules,
              "custom observations differ from full-width reference");
    }
}

struct WireEntry {
    float value;
    qalsh::PointId id;
};
static_assert(sizeof(WireEntry) == 8U);

struct RangeTrace {
    qalsh::SearchStrategy* strategy{nullptr};
    std::vector<Evaluation> evaluations;
    std::size_t final_hits{0};
};

template <typename Strategy>
RangeTrace RunRanges(Strategy& strategy, std::uint32_t points, std::size_t hit_count) {
    RangeTrace trace{.strategy = &strategy};
    qalsh::detail::RangeExecution execution{
        .strategy = &strategy,
        .state = {.projection_hits = 11U},
        .context = &trace,
        .evaluate = [](qalsh::detail::RangeExecution& execution, qalsh::PointId id) {
            auto& trace = *static_cast<RangeTrace*>(execution.context);
            ++execution.state.evaluated_candidates;
            trace.evaluations.emplace_back(id, execution.state.projection_hits,
                                          execution.state.evaluated_candidates);
            trace.strategy->on_evaluation({.point_id = id,
                                            .status = qalsh::EvaluationStatus::evaluated,
                                            .exact_distance = true},
                                           execution.state);
        }};
    std::array<WireEntry, 257U> entries{};
    constexpr std::array<std::size_t, 10> sizes{1U, 2U, 3U, 4U, 5U, 6U, 7U, 15U, 64U, 257U};
    qalsh::detail::DeliverRange<Strategy>(execution, {
        reinterpret_cast<const std::byte*>(entries.data()), 0U, false, 0U, -0.5F});
    Check(execution.state.projection_hits == 11U && trace.evaluations.empty(),
          "empty range changed the snapshot or invoked an evaluation");
    std::size_t consumed = 0U;
    for (std::size_t batch = 0; consumed < hit_count; ++batch) {
        const auto count = std::min(sizes[batch % sizes.size()], hit_count - consumed);
        for (std::size_t i = 0; i < count; ++i) {
            entries[i] = {static_cast<float>(i), static_cast<qalsh::PointId>((consumed + i) % points)};
        }
        const bool reverse = batch % 2U != 0U;
        const qalsh::detail::HitRange range{
            reinterpret_cast<const std::byte*>(&entries[reverse ? count - 1U : 0U]),
            count, reverse, 0U, -0.5F};
        qalsh::detail::DeliverRange<Strategy>(execution, range);
        consumed += count;
        Check(execution.state.projection_hits == 11U + consumed,
              "range hit accounting changed after a stop");
    }
    trace.final_hits = execution.state.projection_hits;
    return trace;
}

void CheckRangeDispatch() {
    constexpr std::uint32_t points = 5U;
    for (const auto threshold : {1U, 255U, 256U, 65536U}) {
        for (const bool custom : {false, true}) {
            for (const auto stop_after : {std::size_t{1U}, std::size_t{3U},
                                          std::numeric_limits<std::size_t>::max()}) {
                qalsh::DefaultQalshStrategy typed, erased;
                qalsh::SearchStrategy& erased_base = erased;
                CountUpReference reference;
                std::vector<RuleEvent> typed_rules, erased_rules, reference_rules;
                std::vector<StopEvent> typed_stops, erased_stops;
                Configure(typed, custom, typed_rules, typed_stops, stop_after);
                Configure(erased, custom, erased_rules, erased_stops, stop_after);
                reference.set_candidate_rule(custom ? CustomRule(reference_rules) : qalsh::CandidateRule{});
                reference.set_stop_after(stop_after);
                const auto query = Query(threshold, points);
                typed.start(query);
                erased_base.start(query);
                reference.start(query);
                const auto effective_threshold = custom ? 264U : threshold;
                const std::size_t hits = static_cast<std::size_t>(effective_threshold + 4U) * points * 2U;
                const auto actual = RunRanges(typed, points, hits);
                const auto dynamic = RunRanges(erased_base, points, hits);
                const auto expected = RunRanges(reference, points, hits);
                Check(actual.evaluations == expected.evaluations && dynamic.evaluations == expected.evaluations,
                      "range candidate order or evaluation snapshot differs from reference");
                Check(actual.final_hits == expected.final_hits && dynamic.final_hits == expected.final_hits,
                      "range final hit count differs from reference");
                Check(typed_rules == reference_rules && erased_rules == reference_rules,
                      "range custom-rule calls differ from reference");
                Check(typed_stops == erased_stops && typed_stops.size() == expected.evaluations.size(),
                      "stop callback timing differs between typed and erased paths");
                for (std::size_t i = 0; i < typed_stops.size(); ++i) {
                    const auto& [id, at_hit, evaluated] = expected.evaluations[i];
                    (void)id;
                    Check(typed_stops[i] == StopEvent{evaluated, at_hit},
                          "stop callback did not observe the accepted hit");
                }
                Check(expected.evaluations.size() == std::min<std::size_t>(points, stop_after),
                      "range did not accept the expected number of distinct points");
                if (stop_after < points) {
                    // Finished strategies historically ignore later hits
                    // before validating their point IDs; preserve that order.
                    const qalsh::ProjectionHit invalid{.point_id = points};
                    Check(!typed.on_projection_hit(invalid, {}).has_candidate() &&
                              !erased_base.on_projection_hit(invalid, {}).has_candidate(),
                          "finished strategy changed direct-hit behavior");
                }
            }
        }
    }
}
}  // namespace

int main() {
    CheckDirectCalls(0U);
    CheckDirectCalls(1U);
    CheckDirectCalls(255U);
    CheckDirectCalls(256U);
    CheckRangeDispatch();
}
