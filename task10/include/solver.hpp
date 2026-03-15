#ifndef TASK10_SOLVER_HPP
#define TASK10_SOLVER_HPP

#include "typedefs.hpp"
#include "sphere_grid.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>

#include <omp.h>

// Maximize minimum same-color distance on a sphere.
// Two-phase search over distance thresholds d: build conflict graph G(d)
// with edges between points closer than d, then K-color G(d).
// Phase 1 (CO): fast scan to narrow the feasible range.
// Phase 2 (TabuCol): stronger local search to refine.
template <typename Scalar>
class Solver {
public:
    enum class Method { Conflict, Tabu };

    using TimePoint = std::chrono::steady_clock::time_point;

    // All distances in kilometers.
    struct Parameters {
        Scalar phase1_min = Scalar(10);
        Scalar phase2_min = Scalar(30);
        Index max_iters   = Index(200000);
        Index n_threads   = Index(0);      // 0 = all CPUs
        Scalar lo_km      = Scalar(0);     // 0 = auto
        // Auto upper bound: hi = 4*R / sqrt(ceil(N/K)), from spherical packing
        Scalar hi_km      = Scalar(0);     // 0 = auto
        Index p1_cut      = Index(2000);   // Phase 1: skip when conflicts > this
        Index p2_cut      = Index(500);    // Phase 2: skip when conflicts > this
        Index p1_n_points = Index(10);     // Phase 1: subintervals per round
        Index p2_n_points = Index(5);      // Phase 2: subintervals per round
        // Phase 1 stops at 2*min_range, Phase 2 at min_range,
        // binary refinement at min_range/4.
        Scalar min_range  = Scalar(2);
        unsigned seed     = 42;
        // CSV: phase,round,point,distance_km,edges,conflicts,feasible,time_s
        std::string search_log_path;
        // 640 lines x 100 values, 1-indexed colors (1..1008)
        std::string coloring_path;

        inline bool check() const noexcept {
            return phase1_min > Scalar(0) && phase2_min > Scalar(0)
                && max_iters > Index(0) && lo_km >= Scalar(0)
                && (hi_km <= Scalar(0) || hi_km > lo_km);
        }
    };

    struct Result {
        Vector<Index> coloring;   // color per vertex, values in [0, K)
        Index conflicts;          // 0 = valid coloring
        Scalar chord_km;          // min chord distance between same-color points
        Scalar arc_km;            // min great-circle distance
        Index bs_steps;           // number of best-distance improvements
        double elapsed_s;
    };

    // Per-thread workspace, allocated once in Init()
    struct RunWorkspace {
        // gamma(v,c) = |{u in adj(v) : color(u) = c}|
        // Enables O(1) move delta: delta(v, old->new) = gamma(v,new) - gamma(v,old)
        RowMajorMatrix<StorageIndex> gamma;
        // tabu(v,c) = iteration when move (v -> c) stops being tabu
        RowMajorMatrix<StorageIndex> tabu;
        Vector<Index> color;
        Vector<Index> best_color;
        // Dense conflict list with O(1) add/remove via swap-with-last
        Vector<Index> conf_list;
        // conf_pos(v) = index in conf_list, or -1 if v is not conflicting
        Vector<Index> conf_pos;
        // Greedy init temporaries (per-thread to avoid races)
        Vector<Index> cnt;       // cnt(c) = same-color neighbor count
        Vector<Index> order;     // Fisher-Yates permutation
        Vector<Index> colored;   // 1 if vertex already assigned
        // CO workspace
        Vector<Index> is_col;    // 1 if vertex is colored (not in queue S)
        Vector<Index> q_count;   // displacement count for weight w(q) = 1 + q^p
        Vector<Scalar> scores;   // per-color conflict score
        Vector<Index> best_color_local;  // CO: best coloring across restarts
    };

    Solver(const SphereGrid<Scalar>& grid, Index k)
        : _grid(grid), _n(static_cast<Index>(grid.N())), _k(k) {}

    // Allocate all workspace. No allocation after this call.
    inline void Init(Index n_threads) {
        _n_threads = (n_threads > Index(0))
            ? n_threads
            : static_cast<Index>(omp_get_max_threads());
        omp_set_num_threads(static_cast<int>(_n_threads));

        _adj_ptr.resize(_n + Index(1));
        _degree.resize(_n);
        _fill.resize(_n);
        _best_color.resize(_n);

        _run_ws.resize(_n_threads);
        for (Index i = Index(0); i < _n_threads; ++i) {
            _run_ws[i].gamma.resize(_n, _k);
            _run_ws[i].tabu.resize(_n, _k);
            _run_ws[i].color.resize(_n);
            _run_ws[i].best_color.resize(_n);
            _run_ws[i].conf_list.resize(_n);
            _run_ws[i].conf_pos.resize(_n);
            _run_ws[i].cnt.resize(_k);
            _run_ws[i].order.resize(_n);
            _run_ws[i].colored.resize(_n);
            _run_ws[i].is_col.resize(_n);
            _run_ws[i].q_count.resize(_n);
            _run_ws[i].scores.resize(_k);
            _run_ws[i].best_color_local.resize(_n);
        }
    }

    // Two-phase multi-point stochastic search.
    //
    // Binary-search over distance d: build conflict graph G(d), try to K-color it.
    // Largest feasible d is the answer.
    //
    // Phase 1 (CO): wide scan, fast but weaker.
    // Phase 2 (TabuCol): narrow refinement, fresh greedy init (warm-start from
    //   CO traps TabuCol above the CO ceiling).
    Result Solve(const Parameters& params) {
        auto t0 = std::chrono::steady_clock::now();
        auto p1_deadline = t0 + std::chrono::duration_cast<
            std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(
                static_cast<double>(params.phase1_min) * 60.0));
        auto deadline = t0 + std::chrono::duration_cast<
            std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(
                static_cast<double>(
                    params.phase1_min + params.phase2_min) * 60.0));

        Init(params.n_threads);
        _max_iters = params.max_iters;

        // Spherical packing bound: area per color = 4*pi*R^2/K,
        // each color has ceil(N/K) points, spacing ~ 4*R/sqrt(ceil(N/K))
        Scalar lo_km = params.lo_km;
        Scalar hi_km = params.hi_km;
        if (hi_km <= Scalar(0)) {
            Scalar n_per_color = std::ceil(
                static_cast<Scalar>(_n) / static_cast<Scalar>(_k));
            hi_km = Scalar(4) * _grid.R()
                / (std::sqrt(n_per_color) * Scalar(1e3));
        }
        Scalar original_hi = hi_km;
        bool auto_range = (params.hi_km <= Scalar(0));

        std::cout << "================================================================\n"
                  << "  Sphere Coloring Solver\n"
                  << "================================================================\n"
                  << "Problem:  N=" << _n << " points, K=" << _k << " colors\n"
                  << "Range:    [" << lo_km << ", " << hi_km << "] km"
                  << (auto_range ? " (auto: 4R/sqrt(ceil(N/K)))" : " (user)") << "\n"
                  << "Threads:  " << _n_threads << "\n"
                  << "Seed:     " << params.seed << "\n"
                  << "\n"
                  << "Phase 1:  Multi-point CO (Conflict Optimizer)\n"
                  << "  Budget: " << params.phase1_min << " min\n"
                  << "  Method: CG:SHOP 2022, w(u)=1+q^1.2, sigma=0.15\n"
                  << "  CO iters: " << params.max_iters << ", restarts: 10\n"
                  << "  Points: N=" << params.p1_n_points << " initial, double on no feasible, max 60\n"
                  << "  Per-point time: 60/30/15s (adaptive by distance)\n"
                  << "  Conflict cut: " << params.p1_cut << "\n"
                  << "\n"
                  << "Phase 2:  Multi-point TabuCol (fresh greedy init)\n"
                  << "  Budget: " << params.phase2_min << " min\n"
                  << "  Method: steepest descent, tenure=0.6K, +2/500 iters\n"
                  << "  TabuCol iters: " << params.max_iters << " per run\n"
                  << "  Runs: " << std::min(_n_threads, Index(12)) << " parallel\n"
                  << "  Points: N=" << params.p2_n_points << " initial, 3 min/point\n"
                  << "  Conflict cut: " << params.p2_cut << "\n"
                  << "  Init: fresh greedy (no warm-start)\n"
                  << "\n"
                  << "Min range: " << params.min_range << " km\n"
                  << "================================================================\n"
                  << std::flush;

        std::ofstream search_log;
        if (!params.search_log_path.empty()) {
            search_log.open(params.search_log_path);
            search_log << "phase,round,point,distance_km,edges,conflicts,feasible,time_s\n";
        }

        Result best;
        best.coloring = Vector<Index>::Zero(_n);
        best.conflicts = _n;
        best.chord_km = lo_km;
        best.arc_km = Scalar(0);
        best.bs_steps = Index(0);

        Vector<Index> warm_start;
        bool need_phase1 = (hi_km - lo_km > Scalar(50));

        // =================================================================
        // Phase 1: Multi-point CO
        //
        // Divide [lo, hi] into N subintervals, sample one random d per interval.
        // Adaptive time per point: 60s near lo (harder), 15s near hi (easy to
        // reject). Best feasible raises lo. Conflicts > p1_cut shrinks hi.
        // No feasible found: double N for finer grid. Stop at N > 8x initial.
        // =================================================================
        if (need_phase1) {
            std::cout << "\n--- Phase 1: Multi-point CO ---\n"
                      << std::flush;

            Scalar lo_d = lo_km;
            Scalar hi_d = hi_km;
            Index n_points = params.p1_n_points;
            Index round = Index(0);
            std::mt19937 phase1_rng(params.seed);
            unsigned seed_offset = 0;

            while (std::chrono::steady_clock::now() < p1_deadline) {
                round++;
                Scalar span = hi_d - lo_d;
                if (span < params.min_range * Scalar(2)) break;

                Scalar sub_width = span / static_cast<Scalar>(n_points);
                Scalar round_best_km = Scalar(0);
                bool any_feasible = false;
                Scalar best_inf_d = hi_d;
                Index best_inf_conf = _n;

                std::cout << "  Round " << round
                          << ": [" << lo_d << ", " << hi_d
                          << "] N=" << n_points << "\n" << std::flush;

                for (Index i = Index(0); i < n_points; ++i) {
                    if (std::chrono::steady_clock::now() > p1_deadline) break;

                    Scalar sub_lo = lo_d
                        + static_cast<Scalar>(i) * sub_width;
                    Scalar offset = std::uniform_real_distribution<Scalar>(
                        Scalar(0), sub_width)(phase1_rng);
                    Scalar d_km = sub_lo + offset;

                    std::cout << "    [" << (i + Index(1)) << "/"
                              << n_points << "] d="
                              << d_km << " km" << std::flush;

                    _BuildAdjCSR(_KmToSq(d_km));
                    Index edges = _adj_ptr(_n) / Index(2);

                    // Lower distances get more time: closer to feasibility boundary
                    Scalar frac = (d_km - lo_d) / span;
                    int point_sec = (frac < Scalar(0.3)) ? 60 :
                                    (frac < Scalar(0.6)) ? 30 : 15;
                    auto point_deadline = std::min(p1_deadline,
                        std::chrono::steady_clock::now()
                        + std::chrono::seconds(point_sec));

                    Index conflicts = _ParallelSearch(
                        Method::Conflict, warm_start,
                        point_deadline, params.seed, seed_offset);
                    // Large prime stride avoids correlated seeds between points
                    seed_offset += static_cast<unsigned>(_n_threads) * 7919u;

                    double elapsed_s = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    int em = static_cast<int>(elapsed_s) / 60;
                    int es = static_cast<int>(elapsed_s) % 60;

                    if (conflicts == Index(0)) {
                        any_feasible = true;
                        if (d_km > round_best_km) {
                            round_best_km = d_km;
                            warm_start = _best_color;
                        }
                        if (d_km > best.chord_km) {
                            best.coloring = _best_color;
                            best.conflicts = Index(0);
                            best.chord_km = d_km;
                            best.bs_steps++;
                        }
                        std::cout << "  feasible  "
                                  << em << "m" << es << "s\n"
                                  << std::flush;
                        if (search_log.is_open())
                            search_log << "1," << round << ","
                                       << (i + Index(1)) << ","
                                       << d_km << "," << edges << ","
                                       << conflicts << ",1,"
                                       << elapsed_s << "\n";
                    } else {
                        std::cout << "  INFEASIBLE (" << conflicts
                                  << ")  " << em << "m" << es << "s\n"
                                  << std::flush;
                        if (search_log.is_open())
                            search_log << "1," << round << ","
                                       << (i + Index(1)) << ","
                                       << d_km << "," << edges << ","
                                       << conflicts << ",0,"
                                       << elapsed_s << "\n";
                        if (conflicts < best_inf_conf) {
                            best_inf_conf = conflicts;
                            best_inf_d = d_km;
                        }
                        // Too many conflicts: everything above is infeasible too
                        if (conflicts > params.p1_cut) {
                            if (d_km < hi_d) hi_d = d_km;
                            std::cout << "    (cut at " << d_km << " km, "
                                      << conflicts << " conf)\n"
                                      << std::flush;
                            break;
                        }
                    }
                }

                if (any_feasible) {
                    lo_d = round_best_km;
                    // Cap density at 4x initial to avoid too many points
                    n_points = std::min(
                        std::max(n_points, params.p1_n_points),
                        params.p1_n_points * Index(4));
                } else {
                    // Double density for finer sampling, stop after 3 doublings
                    Index max_p1 = params.p1_n_points * Index(8);
                    n_points = std::min(n_points * Index(2), max_p1);
                    if (hi_d - lo_d < params.min_range * Scalar(2) || n_points > max_p1) {
                        std::cout << "  Range exhausted, stopping.\n"
                                  << std::flush;
                        break;
                    }
                    std::cout << "  No feasible, retrying ["
                              << lo_d << ", " << hi_d << "] N="
                              << n_points << "\n" << std::flush;
                }
            }

            double p1_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            int p1m = static_cast<int>(p1_elapsed) / 60;
            int p1s = static_cast<int>(p1_elapsed) % 60;
            std::cout << "  Phase 1 summary:\n"
                      << "    Rounds: " << round << "\n"
                      << "    Best feasible: " << best.chord_km << " km\n"
                      << "    Final range: [" << lo_d << ", " << hi_d << "]\n"
                      << "    Time: " << p1m << "m" << p1s << "s\n"
                      << std::flush;

            lo_km = best.chord_km;
            hi_km = original_hi;
        }

        // =================================================================
        // Phase 2: Multi-point TabuCol (fresh greedy init)
        //
        // Same multi-point structure as Phase 1 but stronger solver.
        // Fresh greedy init (warm-start from CO traps TabuCol in local optima).
        // Binary refinement at end between best feasible and best infeasible.
        // =================================================================
        {
            std::cout << "\n--- Phase 2: Multi-point TabuCol ---\n"
                      << std::flush;

            // Diminishing returns beyond 12 parallel TabuCol runs
            Index n_runs_save = _n_threads;
            _n_runs = std::min(_n_threads, Index(12));

            Scalar lo_d = lo_km;
            Scalar hi_d = hi_km;
            Index n_points = params.p2_n_points;
            Index round = Index(0);
            // Separate seed space from Phase 1
            unsigned seed_offset = 1000000u;

            Vector<Index> no_warm;

            while (std::chrono::steady_clock::now() < deadline) {
                round++;
                Scalar span = hi_d - lo_d;
                if (span < params.min_range) break;

                Scalar sub_width = span / static_cast<Scalar>(n_points);
                std::mt19937 phase2_rng(params.seed + 12345u
                    + static_cast<unsigned>(round));

                std::cout << "  Round " << round
                          << ": [" << lo_d << ", " << hi_d
                          << "] N=" << n_points << "\n" << std::flush;

                Scalar round_best_km = Scalar(0);
                bool any_feasible = false;
                Scalar best_inf_d = hi_d;
                Index best_inf_conf = _n;

                for (Index i = Index(0); i < n_points; ++i) {
                    if (std::chrono::steady_clock::now() > deadline) break;

                    Scalar sub_lo = lo_d
                        + static_cast<Scalar>(i) * sub_width;
                    Scalar offset = std::uniform_real_distribution<Scalar>(
                        Scalar(0), sub_width)(phase2_rng);
                    Scalar d_km = sub_lo + offset;

                    std::cout << "    [" << (i + Index(1)) << "/"
                              << n_points << "] d="
                              << d_km << " km" << std::flush;

                    _BuildAdjCSR(_KmToSq(d_km));
                    Index edges = _adj_ptr(_n) / Index(2);

                    auto point_deadline = std::min(deadline,
                        std::chrono::steady_clock::now()
                        + std::chrono::seconds(180));

                    Index conflicts = _ParallelSearch(
                        Method::Tabu, no_warm,
                        point_deadline, params.seed, seed_offset);
                    seed_offset += static_cast<unsigned>(_n_runs) * 7919u;

                    double elapsed_s = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    int em = static_cast<int>(elapsed_s) / 60;
                    int es = static_cast<int>(elapsed_s) % 60;

                    if (conflicts == Index(0)) {
                        any_feasible = true;
                        if (d_km > round_best_km)
                            round_best_km = d_km;
                        if (d_km > best.chord_km) {
                            best.coloring = _best_color;
                            best.conflicts = Index(0);
                            best.chord_km = d_km;
                            best.bs_steps++;
                        }
                        std::cout << "  feasible  "
                                  << em << "m" << es << "s\n"
                                  << std::flush;
                        if (search_log.is_open())
                            search_log << "2," << round << ","
                                       << (i + Index(1)) << ","
                                       << d_km << "," << edges << ","
                                       << conflicts << ",1,"
                                       << elapsed_s << "\n";
                    } else {
                        if (conflicts < best_inf_conf) {
                            best_inf_conf = conflicts;
                            best_inf_d = d_km;
                        }
                        std::cout << "  INFEASIBLE (" << conflicts
                                  << ")  " << em << "m" << es << "s\n"
                                  << std::flush;
                        if (search_log.is_open())
                            search_log << "2," << round << ","
                                       << (i + Index(1)) << ","
                                       << d_km << "," << edges << ","
                                       << conflicts << ",0,"
                                       << elapsed_s << "\n";
                        // 2x p2_cut: boundary is clearly below this distance
                        if (conflicts > params.p2_cut * Index(2)) {
                            if (d_km < hi_d) hi_d = d_km;
                            std::cout << "    (cut hi=" << hi_d << ")\n"
                                      << std::flush;
                            break;
                        }
                        // p2_cut: skip rest of round (stochastic, don't shrink hi)
                        if (conflicts > params.p2_cut) {
                            std::cout << "    (skip rest of round)\n"
                                      << std::flush;
                            break;
                        }
                    }
                }

                if (any_feasible) {
                    Scalar progress = round_best_km - lo_d;
                    Scalar span = hi_d - lo_d;
                    lo_d = round_best_km;

                    // Stagnating: increase density for finer grid
                    if (progress <= span / Scalar(10)
                        || progress <= params.min_range) {
                        Index max_p2 = std::max(
                            params.p2_n_points * Index(8),
                            static_cast<Index>((hi_d - lo_d) / params.min_range));
                        n_points = std::min(n_points * Index(2), max_p2);
                    }
                } else {
                    Index max_p2 = std::max(
                        params.p2_n_points * Index(8),
                        static_cast<Index>((hi_d - lo_d) / params.min_range));
                    n_points = std::min(n_points * Index(2), max_p2);
                    if (hi_d - lo_d < params.min_range || n_points >= max_p2) {
                        std::cout << "  Range exhausted.\n" << std::flush;
                        break;
                    }
                    std::cout << "  No feasible, retrying ["
                              << lo_d << ", " << hi_d << "] N="
                              << n_points << "\n" << std::flush;
                }
            }

            // Binary refinement between best feasible and first infeasible.
            // Only if meaningful gap (> 1 km).
            if (best.chord_km > lo_km
                && hi_d - best.chord_km > Scalar(1)) {
                Scalar ref_lo = best.chord_km;
                // Cap window at 10 km above best to stay near the boundary
                Scalar ref_hi = std::min(hi_d,
                    best.chord_km + Scalar(10));

                std::cout << "  Refining [" << ref_lo << ", "
                          << ref_hi << "]\n" << std::flush;

                Index ref_step = Index(0);
                while (ref_hi - ref_lo > params.min_range / Scalar(4)) {
                    ref_step++;
                    if (std::chrono::steady_clock::now() > deadline) {
                        std::cout << "  Time limit reached.\n"
                                  << std::flush;
                        break;
                    }

                    Scalar mid = (ref_lo + ref_hi) / Scalar(2);
                    std::cout << "    d=" << mid << " km" << std::flush;

                    _BuildAdjCSR(_KmToSq(mid));
                    Index edges = _adj_ptr(_n) / Index(2);
                    auto step_deadline = std::min(deadline,
                        std::chrono::steady_clock::now()
                        + std::chrono::seconds(180));
                    Index conflicts = _ParallelSearch(
                        Method::Tabu, no_warm,
                        step_deadline, params.seed, seed_offset);
                    seed_offset += static_cast<unsigned>(_n_runs) * 7919u;

                    double elapsed_s = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    int em = static_cast<int>(elapsed_s) / 60;
                    int es = static_cast<int>(elapsed_s) % 60;

                    if (conflicts == Index(0)) {
                        if (mid > best.chord_km) {
                            best.coloring = _best_color;
                            best.conflicts = Index(0);
                            best.chord_km = mid;
                            best.bs_steps++;
                        }
                        ref_lo = mid;
                        std::cout << "  feasible  [" << ref_lo
                                  << ", " << ref_hi << "]  "
                                  << em << "m" << es << "s\n"
                                  << std::flush;
                        if (search_log.is_open())
                            search_log << "2," << round << ","
                                       << ref_step << ","
                                       << mid << "," << edges << ","
                                       << conflicts << ",1,"
                                       << elapsed_s << "\n";
                    } else {
                        ref_hi = mid;
                        std::cout << "  INFEASIBLE (" << conflicts
                                  << ")  [" << ref_lo << ", "
                                  << ref_hi << "]  "
                                  << em << "m" << es << "s\n"
                                  << std::flush;
                        if (search_log.is_open())
                            search_log << "2," << round << ","
                                       << ref_step << ","
                                       << mid << "," << edges << ","
                                       << conflicts << ",0,"
                                       << elapsed_s << "\n";
                    }
                }
            }

            _n_runs = n_runs_save;
        }

        // =================================================================
        // Verify and return
        // =================================================================
        if (best.conflicts == Index(0)) {
            // Brute-force true minimum (independent of graph threshold)
            auto [chord_m, arc_m] = _Verify(best.coloring);
            best.chord_km = chord_m / Scalar(1e3);
            best.arc_km = arc_m / Scalar(1e3);
        }

        auto t1 = std::chrono::steady_clock::now();
        best.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        int fm = static_cast<int>(best.elapsed_s) / 60;
        int fs = static_cast<int>(best.elapsed_s) % 60;

        std::cout << "\n================================================================\n"
                  << "  RESULT\n"
                  << "================================================================\n"
                  << "  Chord distance: " << best.chord_km << " km\n"
                  << "  Arc distance:   " << best.arc_km << " km\n"
                  << "  Conflicts:      " << best.conflicts << "\n"
                  << "  BS steps:       " << best.bs_steps << "\n"
                  << "  Total time:     " << fm << "m" << fs << "s\n"
                  << "================================================================\n"
                  << std::flush;

        if (!params.coloring_path.empty() && best.conflicts == Index(0)) {
            std::ofstream cfile(params.coloring_path);
            for (Index i = Index(0); i < _n; ++i) {
                cfile << (best.coloring(i) + Index(1));  // 1-indexed
                if ((i + Index(1)) % Index(100) == Index(0) || i == _n - Index(1))
                    cfile << "\n";
                else
                    cfile << ",";
            }
        }

        return best;
    }

private:
    // =====================================================================
    // Helpers
    // =====================================================================

    // chord_km -> squared meters. Graph edge (u,v) iff ||u-v||^2 < _KmToSq(d).
    static inline Scalar _KmToSq(Scalar km) {
        Scalar m = km * Scalar(1e3);
        return m * m;
    }

    inline Index _Degree(Index v) const {
        return _adj_ptr(v + Index(1)) - _adj_ptr(v);
    }

    // =====================================================================
    // CSR adjacency via spatial hashing (Teschner et al., 2003)
    //
    // Cubic cells of side = chord * 1.01. Each point compares against ~27
    // neighboring cells instead of O(N^2) all-pairs.
    // Two-pass: (1) count degrees in parallel, (2) fill CSR.
    // The spatial hash map is the only temporary allocation.
    // =====================================================================
    void _BuildAdjCSR(Scalar dist_sq) {
        Scalar chord = std::sqrt(dist_sq);
        // Cell slightly larger than chord: all pairs within chord
        // are guaranteed same-cell or adjacent-cell
        Scalar cell_sz = chord * Scalar(1.01);
        if (cell_sz < Scalar(1e-6)) {
            _adj_ptr.head(_n + Index(1)).setZero();
            return;
        }

        Scalar inv = Scalar(1) / cell_sz;

        struct CellKey {
            int x, y, z;
            bool operator==(const CellKey& o) const {
                return x == o.x && y == o.y && z == o.z;
            }
        };
        // Three large primes (Teschner et al., 2003)
        struct CellHash {
            size_t operator()(const CellKey& k) const {
                size_t h = size_t(k.x) * 73856093u;
                h ^= size_t(k.y) * 19349663u;
                h ^= size_t(k.z) * 83492791u;
                return h;
            }
        };

        std::unordered_map<CellKey, std::vector<Index>, CellHash> cells;
        cells.reserve(static_cast<size_t>(_n / Index(4)));

        for (Index i = Index(0); i < _n; ++i) {
            auto p = _grid.Point(static_cast<int>(i));
            int cx = static_cast<int>(std::floor(p(0) * inv));
            int cy = static_cast<int>(std::floor(p(1) * inv));
            int cz = static_cast<int>(std::floor(p(2) * inv));
            cells[{cx, cy, cz}].push_back(i);
        }

        std::vector<CellKey> keys;
        keys.reserve(cells.size());
        for (auto it = cells.begin(); it != cells.end(); ++it)
            keys.push_back(it->first);

        // Pass 1: count degrees. Thread-local accumulators merged via omp critical.
        _degree.head(_n).setZero();

        #pragma omp parallel num_threads(static_cast<int>(_n_threads))
        {
            Vector<Index> local_deg = Vector<Index>::Zero(_n);

            #pragma omp for schedule(dynamic, 64)
            for (Index ci = Index(0);
                 ci < static_cast<Index>(keys.size()); ++ci) {
                const auto& pts = cells.at(keys[ci]);
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dz = -1; dz <= 1; ++dz) {
                            CellKey nk{
                                keys[ci].x + dx,
                                keys[ci].y + dy,
                                keys[ci].z + dz};
                            auto it = cells.find(nk);
                            if (it == cells.end()) continue;
                            const auto& npts = it->second;
                            for (Index a : pts)
                                for (Index b : npts) {
                                    if (b <= a) continue;
                                    if (_grid.SqDist(
                                            static_cast<int>(a),
                                            static_cast<int>(b))
                                        < dist_sq) {
                                        local_deg(a)++;
                                        local_deg(b)++;
                                    }
                                }
                        }
            }

            #pragma omp critical
            _degree.head(_n).noalias() += local_deg;
        }

        // Prefix sums: _adj_ptr(v) = start of v's neighbor list
        _adj_ptr(Index(0)) = Index(0);
        for (Index i = Index(0); i < _n; ++i)
            _adj_ptr(i + Index(1)) = _adj_ptr(i) + _degree(i);

        Index total_entries = _adj_ptr(_n);
        if (_adj_data.size() < total_entries)
            _adj_data.resize(total_entries);

        // Pass 2: fill CSR neighbor lists (sequential)
        _fill.head(_n).setZero();
        for (Index ci = Index(0);
             ci < static_cast<Index>(keys.size()); ++ci) {
            const auto& pts = cells.at(keys[ci]);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        CellKey nk{
                            keys[ci].x + dx,
                            keys[ci].y + dy,
                            keys[ci].z + dz};
                        auto it = cells.find(nk);
                        if (it == cells.end()) continue;
                        const auto& npts = it->second;
                        for (Index a : pts)
                            for (Index b : npts) {
                                if (b <= a) continue;
                                if (_grid.SqDist(
                                        static_cast<int>(a),
                                        static_cast<int>(b))
                                    < dist_sq) {
                                    _adj_data(_adj_ptr(a) + _fill(a)++) =
                                        static_cast<StorageIndex>(b);
                                    _adj_data(_adj_ptr(b) + _fill(b)++) =
                                        static_cast<StorageIndex>(a);
                                }
                            }
                    }
        }

        Index max_deg = _degree.head(_n).maxCoeff();
        std::cout << " (" << total_entries / Index(2)
                  << " edges, deg=" << max_deg << ") " << std::flush;
    }

    // =====================================================================
    // Greedy initialization: assign each vertex the color with fewest
    // same-colored neighbors. Much fewer initial conflicts than random.
    // =====================================================================

    // Ordered greedy: vertices 0..N-1. Deterministic (run 0).
    inline void _OrderedGreedyInit(RunWorkspace& ws) {
        ws.color.head(_n).setZero();
        for (Index v = Index(0); v < _n; ++v) {
            Index deg = _Degree(v);
            Index base = _adj_ptr(v);
            ws.cnt.head(_k).setZero();
            for (Index d = Index(0); d < deg; ++d) {
                Index u = static_cast<Index>(_adj_data(base + d));
                if (u < v) ws.cnt(ws.color(u))++;
            }
            Index min_idx;
            ws.cnt.head(_k).minCoeff(&min_idx);
            ws.color(v) = min_idx;
        }
    }

    // Randomized greedy: Fisher-Yates shuffle, then min-conflict assign.
    // Different permutation per thread gives diverse starting colorings.
    inline void _RandomizedGreedyInit(RunWorkspace& ws,
                                      std::mt19937& rng) {
        for (Index i = Index(0); i < _n; ++i)
            ws.order(i) = i;
        for (Index i = _n - Index(1); i > Index(0); --i) {
            Index j = std::uniform_int_distribution<Index>(
                Index(0), i)(rng);
            std::swap(ws.order(i), ws.order(j));
        }

        ws.colored.head(_n).setZero();
        ws.color.head(_n).setZero();
        for (Index vi = Index(0); vi < _n; ++vi) {
            Index v = ws.order(vi);
            Index deg = _Degree(v);
            Index base = _adj_ptr(v);
            ws.cnt.head(_k).setZero();
            for (Index d = Index(0); d < deg; ++d) {
                Index u = static_cast<Index>(_adj_data(base + d));
                if (ws.colored(u)) ws.cnt(ws.color(u))++;
            }
            Index min_idx;
            ws.cnt.head(_k).minCoeff(&min_idx);
            ws.color(v) = min_idx;
            ws.colored(v) = Index(1);
        }
    }

    // =====================================================================
    // Conflict list: O(1) add/remove via swap-with-last.
    // Vertex is conflicting when gamma(v, color(v)) > 0.
    // =====================================================================
    inline void _UpdateConfList(
        Index v, const RowMajorMatrix<StorageIndex>& gamma, const Vector<Index>& color,
        Vector<Index>& conf_list, Vector<Index>& conf_pos,
        Index& conf_sz) const {

        bool is_conf = gamma(v, color(v)) > StorageIndex(0);
        if (is_conf && conf_pos(v) < Index(0)) {
            conf_pos(v) = conf_sz;
            conf_list(conf_sz++) = v;
        } else if (!is_conf && conf_pos(v) >= Index(0)) {
            Index pos = conf_pos(v);
            conf_list(pos) = conf_list(--conf_sz);
            conf_pos(conf_list(pos)) = pos;
            conf_pos(v) = Index(-1);
        }
    }

    // Count conflicting edges (each edge counted once: u > v only).
    inline Index _CountConflicts(const Vector<Index>& color) const {
        Index total = Index(0);
        for (Index v = Index(0); v < _n; ++v) {
            Index deg = _Degree(v);
            Index base = _adj_ptr(v);
            for (Index d = Index(0); d < deg; ++d) {
                Index u = static_cast<Index>(_adj_data(base + d));
                if (u > v && color(v) == color(u))
                    total++;
            }
        }
        return total;
    }

    // =====================================================================
    // Parallel search: n_runs independent CO or TabuCol runs.
    // Shared CSR graph, independent per-thread workspaces.
    // Atomic stop_flag: first run to reach 0 conflicts stops all others.
    //
    // Per-run seed = base_seed + seed_offset + r * 7919
    // Run 0: ordered greedy. Runs 1+: randomized greedy.
    // CO: uses warm_start if available. TabuCol: always fresh greedy.
    // =====================================================================
    Index _ParallelSearch(Method method,
                          const Vector<Index>& warm_start,
                          TimePoint deadline,
                          unsigned base_seed,
                          unsigned seed_offset) {

        Index n_runs = (method == Method::Tabu) ? _n_runs : _n_threads;
        Index actual_threads = std::min(n_runs, _n_threads);

        Vector<Index> run_results = Vector<Index>::Constant(n_runs, _n);
        std::atomic<bool> stop_flag{false};

        #pragma omp parallel num_threads(static_cast<int>(actual_threads))
        {
            #pragma omp for schedule(static, 1)
            for (Index r = Index(0); r < n_runs; ++r) {
                if (stop_flag.load(std::memory_order_relaxed)) {
                    run_results(r) = _n;
                    continue;
                }

                Index tid = static_cast<Index>(omp_get_thread_num());
                RunWorkspace& ws = _run_ws[tid];

                unsigned run_seed = base_seed + seed_offset
                    + static_cast<unsigned>(r) * 7919u;
                std::mt19937 rng(run_seed);

                // CO reuses warm_start; TabuCol always starts fresh
                bool use_warm = (warm_start.size() == _n
                    && method == Method::Conflict);
                if (use_warm) {
                    ws.color.head(_n) = warm_start.head(_n);
                } else if (r == Index(0)) {
                    _OrderedGreedyInit(ws);
                } else {
                    _RandomizedGreedyInit(ws, rng);
                }

                Index result = _n;
                if (method == Method::Conflict) {
                    result = _ConflictOptSingle(
                        ws, rng, stop_flag, deadline);
                } else {
                    result = _TabuColSingle(
                        ws, rng, stop_flag, deadline);
                }

                if (result == Index(0))
                    stop_flag.store(true, std::memory_order_relaxed);

                run_results(r) = result;
            }
        }

        Index best_run;
        run_results.head(n_runs).minCoeff(&best_run);

        // Static schedule: thread for run r = r % n_threads
        Index tid_best = best_run % _n_threads;
        _best_color.head(_n) =
            _run_ws[tid_best].color.head(_n);

        return run_results(best_run);
    }

    // =====================================================================
    // CO: Conflict Optimizer (Crombez et al., arXiv:2303.09632)
    //
    // Partial valid coloring. Conflicting vertices uncolored, queued in S.
    // Each step: dequeue v, assign best color (min weighted conflict score),
    // displace newly-conflicting neighbors back to S.
    //
    // w(u) = 1 + q(u)^1.2 penalizes frequently-displaced vertices to
    //   escape cycling. q(u) = number of times u was displaced.
    // Gaussian noise N(1, 0.15) on scores for diversification.
    // Restart when any q(u) > q_max: perturb 5% of best coloring.
    // =====================================================================
    Index _ConflictOptSingle(
        RunWorkspace& ws,
        std::mt19937& rng,
        std::atomic<bool>& stop_flag,
        TimePoint deadline) {

        Vector<Index>& color = ws.color;

        const Scalar co_p = Scalar(1.2);       // weight exponent
        const Scalar co_sigma = Scalar(0.15);   // noise std dev
        const Index co_max_restarts = Index(10);

        // q_max ~ 2000 * (75000/N)^2, clamped >= 100
        // Smaller problems tolerate more displacement cycles before restart
        Index q_max = static_cast<Index>(
            Scalar(2000) * std::pow(
                Scalar(75000) / Scalar(_n), Scalar(2)));
        if (q_max < Index(100)) q_max = Index(100);

        // Precompute w(q) = 1 + q^p to avoid pow() in hot loop
        if (_w_table.size() < q_max + Index(2))
            _w_table.resize(q_max + Index(2));
        for (Index q = Index(0); q <= q_max + Index(1); ++q)
            _w_table(q) = Scalar(1) + std::pow(
                static_cast<Scalar>(q), co_p);

        Index best_conflicts = _n;
        ws.best_color_local.head(_n) = color;

        for (Index restart = Index(0);
             restart < co_max_restarts; ++restart) {
            if (stop_flag.load(std::memory_order_relaxed)) break;
            if (std::chrono::steady_clock::now() > deadline) break;

            // Restart: perturb 5% of best coloring randomly
            if (restart > Index(0)) {
                if (best_conflicts < _n)
                    color = ws.best_color_local;
                Index n_perturb = std::max(Index(1), _n / Index(20));
                std::uniform_int_distribution<Index> vd(
                    Index(0), _n - Index(1));
                std::uniform_int_distribution<Index> cd(
                    Index(0), _k - Index(1));
                for (Index p = Index(0); p < n_perturb; ++p)
                    color(vd(rng)) = cd(rng);
            }

            // is_col(v) = 1 if colored, 0 if in queue S
            ws.is_col.head(_n).setOnes();
            ws.q_count.head(_n).setZero();
            std::queue<Index> S;

            // Seed queue with conflicting vertices
            for (Index v = Index(0); v < _n; ++v) {
                if (!ws.is_col(v)) continue;
                Index deg = _Degree(v);
                Index base = _adj_ptr(v);
                for (Index d = Index(0); d < deg; ++d) {
                    Index u = static_cast<Index>(_adj_data(base + d));
                    if (ws.is_col(u) && color(u) == color(v)) {
                        ws.is_col(v) = Index(0);
                        S.push(v);
                        ws.q_count(v)++;
                        break;
                    }
                }
            }

            Index uncolored = static_cast<Index>(S.size());
            if (uncolored == Index(0)) return Index(0);

            std::normal_distribution<Scalar> noise_dist(
                Scalar(1), co_sigma);
            bool do_restart = false;

            for (Index iter = Index(0);
                 !S.empty() && iter < _max_iters; ++iter) {

                // Check stop conditions every 1024 iters (amortize overhead)
                if ((iter & Index(1023)) == Index(0)
                    && (stop_flag.load(std::memory_order_relaxed)
                        || std::chrono::steady_clock::now() > deadline))
                    break;

                Index v = S.front();
                S.pop();
                if (ws.is_col(v)) continue;

                // Score each color c: sum of w(q(u)) over colored neighbors with color c
                ws.scores.head(_k).setZero();
                Index deg = _Degree(v);
                Index base = _adj_ptr(v);
                for (Index d = Index(0); d < deg; ++d) {
                    Index u = static_cast<Index>(_adj_data(base + d));
                    if (ws.is_col(u)) {
                        Index qu = std::min(
                            ws.q_count(u), q_max + Index(1));
                        ws.scores(color(u)) += _w_table(qu);
                    }
                }

                // Multiplicative Gaussian noise on non-zero scores
                for (Index c = Index(0); c < _k; ++c) {
                    if (ws.scores(c) > Scalar(0))
                        ws.scores(c) *= std::max(
                            Scalar(0.01), noise_dist(rng));
                }

                Index best_c;
                ws.scores.head(_k).minCoeff(&best_c);

                color(v) = best_c;
                ws.is_col(v) = Index(1);
                uncolored--;

                // Displace same-colored neighbors back to queue
                for (Index d = Index(0); d < deg; ++d) {
                    Index u = static_cast<Index>(_adj_data(base + d));
                    if (ws.is_col(u) && color(u) == best_c) {
                        ws.is_col(u) = Index(0);
                        ws.q_count(u)++;
                        uncolored++;
                        S.push(u);
                        // q exceeded threshold: search is cycling, restart
                        if (ws.q_count(u) > q_max) {
                            do_restart = true;
                            break;
                        }
                    }
                }

                if (do_restart) break;
                if (uncolored == Index(0)) return Index(0);
            }

            if (uncolored == Index(0)) return Index(0);

            Index conf = _CountConflicts(color);
            if (conf < best_conflicts) {
                best_conflicts = conf;
                ws.best_color_local = color;
            }
        }

        color = ws.best_color_local;
        return best_conflicts;
    }

    // =====================================================================
    // TabuCol (Hertz & de Werra, 1987): steepest descent with tabu list.
    //
    // Full scan: for all conflicting vertices x all K colors, find
    // best non-tabu move. Cost per iteration: O(conf_sz * K).
    //
    // Move delta: delta(v, old->new) = gamma(v,new) - gamma(v,old)
    // Tabu: after v: old->new, reverse move (v->old) banned for
    //   tenure + rand(0,10) iters. Random offset prevents synchronized cycling.
    // Aspiration: override tabu if total + delta < best_total (new global best).
    // Tenure: base = 0.6*K, +2 every 500 stagnant iters, capped at K/2.
    //   Reset to base on new best.
    // Stagnation exit: 20K iters no improvement AND best > 5000 conflicts.
    // Tie-breaking: prefer smaller vertex index (deterministic).
    // =====================================================================
    Index _TabuColSingle(
        RunWorkspace& ws,
        std::mt19937& rng,
        std::atomic<bool>& stop_flag,
        TimePoint deadline) {

        RowMajorMatrix<StorageIndex>& gamma = ws.gamma;
        RowMajorMatrix<StorageIndex>& tabu = ws.tabu;
        Vector<Index>& color = ws.color;
        Vector<Index>& best_color = ws.best_color;
        Vector<Index>& conf_list = ws.conf_list;
        Vector<Index>& conf_pos = ws.conf_pos;

        // gamma(v,c) = |{u in adj(v) : color(u) = c}|
        gamma.block(Index(0), Index(0), _n, _k).setZero();
        for (Index v = Index(0); v < _n; ++v) {
            Index deg = _Degree(v);
            Index base = _adj_ptr(v);
            for (Index d = Index(0); d < deg; ++d)
                gamma(v, color(
                    static_cast<Index>(_adj_data(base + d))))++;
        }

        // total = sum(gamma(v, color(v))) / 2  (each edge counted twice)
        Index total = Index(0);
        for (Index v = Index(0); v < _n; ++v)
            total += static_cast<Index>(gamma(v, color(v)));
        total /= Index(2);

        if (total == Index(0)) return Index(0);

        Index best_total = total;
        best_color.head(_n) = color.head(_n);

        conf_pos.head(_n).setConstant(Index(-1));
        Index conf_sz = Index(0);
        for (Index v = Index(0); v < _n; ++v) {
            if (gamma(v, color(v)) > StorageIndex(0)) {
                conf_pos(v) = conf_sz;
                conf_list(conf_sz++) = v;
            }
        }

        tabu.block(Index(0), Index(0), _n, _k).setZero();

        // Tenure proportional to K: more colors = more alternatives = longer bans
        Index base_tenure = std::max(Index(10),
            static_cast<Index>(
                Scalar(0.6) * static_cast<Scalar>(_k)));
        Index tenure = base_tenure;
        Index last_improve = Index(0);

        for (Index iter = Index(1);
             iter <= _max_iters && total > Index(0); ++iter) {

            // Check stop/stagnation every 256 iters (amortize overhead)
            if ((iter & Index(255)) == Index(0)) {
                if (stop_flag.load(std::memory_order_relaxed)
                    || std::chrono::steady_clock::now() > deadline)
                    break;
                // Many conflicts + no progress = distance too large, bail
                if (iter - last_improve > Index(20000)
                    && best_total > Index(5000))
                    break;
            }

            if (conf_sz == Index(0)) {
                color.head(_n) = best_color.head(_n);
                return Index(0);
            }

            // Full scan: best (vertex, color) move among conflicting vertices
            Index mv_v = Index(-1);
            Index mv_c = Index(-1);
            StorageIndex mv_delta =
                static_cast<StorageIndex>(_n + Index(1));

            for (Index ci = Index(0); ci < conf_sz; ++ci) {
                Index v = conf_list(ci);
                Index oc = color(v);
                StorageIndex gv_oc = gamma(v, oc);

                for (Index c = Index(0); c < _k; ++c) {
                    if (c == oc) continue;
                    // delta = gamma(v, new_c) - gamma(v, old_c)
                    StorageIndex delta = gamma(v, c) - gv_oc;

                    bool is_tabu = (tabu(v, c)
                        > static_cast<StorageIndex>(iter));
                    // Aspiration: override tabu if new global best
                    bool aspiration = (static_cast<Index>(
                        total + static_cast<Index>(delta))
                        < best_total);

                    if (is_tabu && !aspiration) continue;

                    if (delta < mv_delta
                        || (delta == mv_delta && v < mv_v)) {
                        mv_delta = delta;
                        mv_v = v;
                        mv_c = c;
                    }
                }
            }

            if (mv_v < Index(0)) continue;

            Index oc = color(mv_v);
            color(mv_v) = mv_c;
            total += static_cast<Index>(mv_delta);

            // Update gamma: each neighbor loses old_color count, gains new
            Index deg = _Degree(mv_v);
            Index base = _adj_ptr(mv_v);
            for (Index d = Index(0); d < deg; ++d) {
                Index u = static_cast<Index>(_adj_data(base + d));
                gamma(u, oc)--;
                gamma(u, mv_c)++;
                _UpdateConfList(u, gamma, color,
                    conf_list, conf_pos, conf_sz);
            }
            _UpdateConfList(mv_v, gamma, color,
                conf_list, conf_pos, conf_sz);

            // Ban reverse move (mv_v -> old_color) for tenure + rand(0,10)
            tabu(mv_v, oc) = static_cast<StorageIndex>(
                iter + tenure
                + std::uniform_int_distribution<Index>(
                    Index(0), Index(10))(rng));

            if (total < best_total) {
                best_total = total;
                best_color.head(_n) = color.head(_n);
                tenure = base_tenure;
                last_improve = iter;
            }

            // Grow tenure when stagnating to force wider exploration
            if (iter % Index(500) == Index(0) && total >= best_total) {
                tenure = std::min(tenure + Index(2), _k / Index(2));
            }
        }

        color.head(_n) = best_color.head(_n);
        return best_total;
    }

    // =====================================================================
    // Verification: brute-force true minimum same-color distance.
    // Independent of the graph threshold used during search.
    // arc = 2*R*arcsin(chord / (2*R))  (great-circle distance)
    // =====================================================================
    std::pair<Scalar, Scalar> _Verify(const Vector<Index>& coloring) {
        Vector<Index> group_sz = Vector<Index>::Zero(_k);
        for (Index i = Index(0); i < _n; ++i)
            group_sz(coloring(i))++;

        Index max_sz = group_sz.maxCoeff();
        DenseMatrix<Index> groups =
            DenseMatrix<Index>::Zero(max_sz, _k);
        Vector<Index> fill_idx = Vector<Index>::Zero(_k);
        for (Index i = Index(0); i < _n; ++i) {
            Index c = coloring(i);
            groups(fill_idx(c)++, c) = i;
        }

        Scalar global_min_sq = std::numeric_limits<Scalar>::max();

        #pragma omp parallel for schedule(dynamic, 4) \
            reduction(min : global_min_sq)
        for (Index c = Index(0); c < _k; ++c) {
            Index sz = group_sz(c);
            Scalar local_min = std::numeric_limits<Scalar>::max();
            for (Index i = Index(0); i < sz; ++i)
                for (Index j = i + Index(1); j < sz; ++j) {
                    Scalar d = _grid.SqDist(
                        static_cast<int>(groups(i, c)),
                        static_cast<int>(groups(j, c)));
                    if (d < local_min) local_min = d;
                }
            if (local_min < global_min_sq)
                global_min_sq = local_min;
        }

        Scalar chord_m = std::sqrt(global_min_sq);
        Scalar arc_m = Scalar(2) * _grid.R()
            * std::asin(chord_m / (Scalar(2) * _grid.R()));

        return {chord_m, arc_m};
    }

    // =====================================================================
    // Data members
    // =====================================================================

    const SphereGrid<Scalar>& _grid;
    Index _n;              // number of points
    Index _k;              // number of colors
    Index _n_threads = Index(1);
    Index _n_runs    = Index(1);   // parallel runs per distance point
    Index _max_iters = Index(200000);

    // CSR adjacency. StorageIndex (int32) halves memory vs int64:
    // 177M entries * 4 bytes = 708 MB vs 1.4 GB.
    Vector<Index> _adj_ptr;          // prefix sums
    Vector<StorageIndex> _adj_data;  // flat neighbor indices

    Vector<Index> _degree;
    Vector<Index> _fill;   // CSR fill counter for two-pass adjacency build
    Vector<Index> _best_color;

    // w_table(q) = 1 + q^p, precomputed for CO hot loop
    Vector<Scalar> _w_table;

    std::vector<RunWorkspace> _run_ws;
};

#endif // TASK10_SOLVER_HPP
