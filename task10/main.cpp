#include "typedefs.hpp"
#include "sphere_grid.hpp"
#include "solver.hpp"

#include <iostream>
#include <string>

#include <omp.h>

int main(int argc, char* argv[]) {
    int n_cells = 64000;
    int n_colors = 1008;
    int n_threads = 0;
    double phase1 = 10.0;
    double phase2 = 30.0;
    unsigned seed = 42;
    std::string search_log, coloring;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc)
            n_cells = std::stoi(argv[++i]);
        else if (arg == "--k" && i + 1 < argc)
            n_colors = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc)
            n_threads = std::stoi(argv[++i]);
        else if (arg == "--phase1" && i + 1 < argc)
            phase1 = std::stod(argv[++i]);
        else if (arg == "--phase2" && i + 1 < argc)
            phase2 = std::stod(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc)
            seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (arg == "--log" && i + 1 < argc)
            search_log = argv[++i];
        else if (arg == "--coloring" && i + 1 < argc)
            coloring = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: task10 [options]\n"
                      << "  --n N         Number of points (default: 64000)\n"
                      << "  --k K         Number of colors (default: 1008)\n"
                      << "  --threads T   Parallel runs (default: all CPUs)\n"
                      << "  --phase1 M    Phase 1 (CO) budget in min (default: 10)\n"
                      << "  --phase2 M    Phase 2 (TabuCol) budget in min (default: 30)\n"
                      << "  --seed S      Master RNG seed (default: 42)\n";
            return 0;
        }
    }

    SphereGrid<double> grid(n_cells);
    Solver<double> solver(grid, static_cast<Index>(n_colors));

    Solver<double>::Parameters params;
    params.phase1_min = phase1;
    params.phase2_min = phase2;
    params.n_threads = static_cast<Index>(n_threads);
    params.seed = seed;
    params.search_log_path = search_log;
    params.coloring_path = coloring;

    auto result = solver.Solve(params);
    return (result.conflicts == 0) ? 0 : 1;
}
