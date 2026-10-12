// Burgers' equation experiment runner.
//
// Every run of ./burgers is one "experiment". Its results go into their own folder
// so earlier experiments are never overwritten:
//
//     data/experiments/<experiment name>/<run name>/solution.bin + metadata.json
//     data/experiments/<experiment name>/experiment_config.json
//
// Examples (run from the build/ folder):
//
//     ./burgers                              # viscosity 0.8, end time 0.125 -> data/experiments/nu_0p8_t_0p125
//     ./burgers --nu 0.03                    # -> data/experiments/nu_0p03_t_0p125
//     ./burgers --nu 0.03 --end-time 1.0     # -> data/experiments/nu_0p03_t_1
//     ./burgers --nu 0.03 --only dx          # only the dx sweep
//     ./burgers --nu 0.03 --name my_test     # custom folder name
//     ./burgers --help                       # list every option
//
// Runs that already exist in the experiment folder are SKIPPED, so an interrupted
// experiment can be resumed by running the same command again. Use --overwrite to redo them.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "burger_scheme.h"
#include "burgers.h"
#include "initial_condition_generator.h"
#include "math_utils.h"
#include "metadata_writer.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Experiment settings (every value can be changed from the command line)
// ---------------------------------------------------------------------------
struct ExperimentSettings {
    double nu = 0.8;                 // kinematic viscosity for the NN test and dx sweep
    double end_time = 0.125;         // simulated time every NN-test / dx-sweep run reaches
    double domain_length = 20.0;

    unsigned nn_seed = 42;           // random starting wave for the NN test
    unsigned dx_seed = 7;            // random starting wave for the dx sweep

    double fine_dx = 0.005;          // fine grid (also used for the godunov_gt reference)
    double coarse_dx = 0.01;         // coarse grid in the NN test
    std::vector<double> dx_list = {0.01, 0.02, 0.04, 0.05, 0.08, 0.16};

    std::set<std::string> sections = {"nn", "dx"};   // also available: "wiki", "basic"

    std::string out_root = "../data/experiments";
    std::string name;                // empty -> built from nu and end time
    bool overwrite = false;
};

// 0.03 -> "0p03", 1.0 -> "1", 0.125 -> "0p125"
std::string numberTag(double value) {
    std::ostringstream ss;
    ss << std::setprecision(6) << value;
    std::string s = ss.str();
    for (char& c : s) {
        if (c == '.') c = 'p';
        if (c == '-') c = 'm';
    }
    return s;
}

// 0.08 -> "0p08" (two decimals, matches the original nn_dx_0p08 style names)
std::string dxTag(double dx) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << dx;
    std::string s = ss.str();
    for (char& c : s) {
        if (c == '.') c = 'p';
    }
    return s;
}

std::vector<double> parseNumberList(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) values.push_back(std::stod(item));
    }
    return values;
}

std::set<std::string> parseWordList(const std::string& text) {
    std::set<std::string> words;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) words.insert(item);
    }
    return words;
}

void printHelp() {
    std::cout <<
        "Usage: ./burgers [options]\n"
        "\n"
        "  --nu <value>            viscosity for the NN test and dx sweep (default 0.8)\n"
        "  --end-time <value>      simulated end time for those runs (default 0.125)\n"
        "  --only <list>           sections to run, comma separated: nn,dx,wiki,basic (default nn,dx)\n"
        "  --dx-list <list>        grid spacings for the dx sweep (default 0.01,0.02,0.04,0.05,0.08,0.16)\n"
        "  --coarse-dx <value>     coarse grid spacing in the NN test (default 0.01)\n"
        "  --fine-dx <value>       fine / reference grid spacing (default 0.005)\n"
        "  --nn-seed <int>         starting wave for the NN test (default 42)\n"
        "  --dx-seed <int>         starting wave for the dx sweep (default 7)\n"
        "  --name <text>           experiment folder name (default nu_<nu>_t_<end time>)\n"
        "  --out-root <path>       where experiment folders go (default ../data/experiments)\n"
        "  --overwrite             redo runs that already exist instead of skipping them\n"
        "  --help                  show this message\n";
}

ExperimentSettings parseArgs(int argc, char** argv) {
    ExperimentSettings s;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + arg);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") { printHelp(); std::exit(0); }
        else if (arg == "--nu")         s.nu = std::stod(next());
        else if (arg == "--end-time")   s.end_time = std::stod(next());
        else if (arg == "--only")       s.sections = parseWordList(next());
        else if (arg == "--dx-list")    s.dx_list = parseNumberList(next());
        else if (arg == "--coarse-dx")  s.coarse_dx = std::stod(next());
        else if (arg == "--fine-dx")    s.fine_dx = std::stod(next());
        else if (arg == "--nn-seed")    s.nn_seed = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--dx-seed")    s.dx_seed = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--name")       s.name = next();
        else if (arg == "--out-root")   s.out_root = next();
        else if (arg == "--overwrite")  s.overwrite = true;
        else {
            std::cerr << "Unknown option: " << arg << "\n\n";
            printHelp();
            std::exit(1);
        }
    }

    for (const std::string& section : s.sections) {
        if (section != "nn" && section != "dx" && section != "wiki" && section != "basic") {
            std::cerr << "Unknown section in --only: " << section << " (use nn, dx, wiki, basic)\n";
            std::exit(1);
        }
    }

    if (s.name.empty()) {
        s.name = "nu_" + numberTag(s.nu) + "_t_" + numberTag(s.end_time);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Number of steps needed to reach end_time. Uses the same time step formula as
// BurgersSolver1d::solve(), and rounds UP so every run reaches at least end_time.
int stepsFor(double dx, double max_u, double nu, double end_time) {
    double dt_advection = BurgersSolver1d::ALPHA * dx / max_u;
    double dt_diffusion = BurgersSolver1d::BETA * dx * dx / std::max(nu, 1e-8);
    double dt = std::min(dt_advection, dt_diffusion);
    return std::max(1, static_cast<int>(std::ceil(end_time / dt - 1e-9)));
}

class ExperimentRunner {
public:
    explicit ExperimentRunner(const ExperimentSettings& settings)
        : settings(settings), dir((fs::path(settings.out_root) / settings.name).string()) {
        fs::create_directories(dir);
    }

    const std::string& folder() const { return dir; }

    // Builds the solver, solves it, saves solution + metadata. Skips runs that already exist.
    // Extra metadata providers (e.g. the random initial condition) can be passed at the end.
    template <typename... Providers>
    void run(const std::string& run_name,
             std::unique_ptr<BurgerScheme> scheme,
             const SolverConfig& config,
             const std::function<double(double)>& initial_condition,
             const Providers&... providers) {
        fs::path run_dir = fs::path(dir) / run_name;
        if (!settings.overwrite && fs::exists(run_dir / "solution.bin") && fs::exists(run_dir / "metadata.json")) {
            std::cout << "[skip] " << run_name << " already exists (use --overwrite to redo)\n";
            skipped++;
            return;
        }

        std::cout << "\n[run ] " << run_name << " (" << config.time_steps << " steps, dx = "
                  << config.spatial_step_size << ", nu = " << config.kinematic_viscosity << ")\n";

        BurgersSolver1d solver(std::move(scheme), config, initial_condition);
        solver.solve();
        solver.saveBinarySolution(dir, run_name);
        MetadataWriter(solver, providers...).write(run_dir / "metadata.json");

        std::cout << "[done] " << run_name << " in " << solver.getRuntimeSeconds() << " s"
                  << " | nan detected: " << solver.wasNanDetected() << "\n";
        completed++;
    }

    // Saves the settings to experiment_config.json. If the folder already holds an experiment
    // made with different physics settings, stop instead of mixing incompatible runs together.
    void writeConfig() const {
        fs::path path = fs::path(dir) / "experiment_config.json";
        nlohmann::json config;
        if (fs::exists(path)) {
            std::ifstream in(path);
            config = nlohmann::json::parse(in, nullptr, false);
            if (config.is_discarded()) config = nlohmann::json::object();
        }

        auto checkSame = [&](const std::string& key, double value) {
            if (config.contains(key) && std::abs(config[key].get<double>() - value) > 1e-12) {
                std::cerr << "Error: experiment folder '" << dir << "' was made with " << key << " = "
                          << config[key].get<double>() << ", but this run uses " << value << ".\n"
                          << "Use a different --name (or delete that folder) so results don't get mixed.\n";
                std::exit(1);
            }
            config[key] = value;
        };
        checkSame("kinematic_viscosity", settings.nu);
        checkSame("end_time", settings.end_time);
        checkSame("domain_length", settings.domain_length);
        checkSame("nn_seed", settings.nn_seed);
        checkSame("dx_seed", settings.dx_seed);
        checkSame("fine_dx", settings.fine_dx);
        checkSame("coarse_dx", settings.coarse_dx);
        config["nn_seed"] = settings.nn_seed;   // store seeds as whole numbers
        config["dx_seed"] = settings.dx_seed;

        // dx_list and sections grow as more runs are added to the same experiment
        std::set<double> dx_values(settings.dx_list.begin(), settings.dx_list.end());
        if (config.contains("dx_list")) {
            for (double dx : config["dx_list"]) dx_values.insert(dx);
        }
        std::set<std::string> sections(settings.sections.begin(), settings.sections.end());
        if (config.contains("sections")) {
            for (const std::string& section : config["sections"]) sections.insert(section);
        }

        config["experiment_name"] = settings.name;
        config["dx_list"] = std::vector<double>(dx_values.begin(), dx_values.end());
        config["sections"] = std::vector<std::string>(sections.begin(), sections.end());

        std::ofstream out(path);
        out << std::setw(4) << config << "\n";
    }

    int completed = 0;
    int skipped = 0;

private:
    const ExperimentSettings& settings;
    std::string dir;
};

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

// NN test: fine and coarse runs of one random wave, with and without the NN.
// Reference for these runs: godunov_fine_grain.
void runNNTest(ExperimentRunner& runner, const ExperimentSettings& s) {
    RandomInitialConditionConfig ic_config;
    RandomInitialCondition ic(ic_config, false, true, s.nn_seed);
    double max_u = compute_max_abs(ic, s.domain_length);

    SolverConfig fine = {
        .kinematic_viscosity = s.nu,
        .time_steps = stepsFor(s.fine_dx, max_u, s.nu, s.end_time),
        .domain_length = s.domain_length,
        .spatial_step_size = s.fine_dx,
    };
    SolverConfig coarse = {
        .kinematic_viscosity = s.nu,
        .time_steps = stepsFor(s.coarse_dx, max_u, s.nu, s.end_time),
        .domain_length = s.domain_length,
        .spatial_step_size = s.coarse_dx,
    };

    runner.run("godunov_fine_grain",   std::make_unique<Godunov>(),  fine,   ic, ic);
    runner.run("ftcs_fine_grain",      std::make_unique<FTCS>(),     fine,   ic, ic);
    runner.run("godunov_coarse_grain", std::make_unique<Godunov>(),  coarse, ic, ic);
    runner.run("ftcs_coarse_grain",    std::make_unique<FTCS>(),     coarse, ic, ic);
    runner.run("ftcs_coarse_grain_nn", std::make_unique<FTCS>(true), coarse, ic, ic);
}

// dx sweep: Godunov reference, then FTCS with and without the NN at each grid spacing.
// Reference for these runs: godunov_gt.
void runDxSweep(ExperimentRunner& runner, const ExperimentSettings& s) {
    RandomInitialConditionConfig ic_config;
    RandomInitialCondition ic(ic_config, false, true, s.dx_seed);
    double max_u = compute_max_abs(ic, s.domain_length);

    SolverConfig gt_config = {
        .kinematic_viscosity = s.nu,
        .time_steps = stepsFor(s.fine_dx, max_u, s.nu, s.end_time),
        .domain_length = s.domain_length,
        .spatial_step_size = s.fine_dx,
    };
    runner.run("godunov_gt", std::make_unique<Godunov>(), gt_config, ic, ic);

    for (double dx : s.dx_list) {
        SolverConfig config = {
            .kinematic_viscosity = s.nu,
            .time_steps = stepsFor(dx, max_u, s.nu, s.end_time),
            .domain_length = s.domain_length,
            .spatial_step_size = dx,
        };
        std::string tag = dxTag(dx);
        runner.run("ftcs_dx_" + tag, std::make_unique<FTCS>(),     config, ic, ic);   // no NN
        runner.run("nn_dx_" + tag,   std::make_unique<FTCS>(true), config, ic, ic);   // with NN
    }
}

// Wikipedia test cases (Godunov only, fixed viscosities; not affected by --nu).
void runWiki(ExperimentRunner& runner) {
    SolverConfig config = {
        .kinematic_viscosity = 1.0,   // overwritten below
        .time_steps = 20000,
        .domain_length = 24.0,        // longer than wikipedia's plot so the boundaries don't interfere
        .spatial_step_size = 0.01,
    };

    // u(x,0) = e^(-x^2/2), centered at x = 12
    std::function<double(double)> gaussian = [](double x) {
        double xs = x - 12.0;
        return std::exp(-(xs * xs) / 2.0);
    };
    // u(x,0) = e^(-(x-1)^2/2) - e^(-(x+1)^2/2), centered at x = 12
    std::function<double(double)> n_wave = [](double x) {
        double xs = x - 12.0;
        return std::exp(-std::pow(xs - 1.0, 2) / 2.0) - std::exp(-std::pow(xs + 1.0, 2) / 2.0);
    };

    for (double v : {1.0, 0.1, 0.01}) {
        config.kinematic_viscosity = v;
        std::string v_str = (v == 1.0) ? "1.0" : (v == 0.1) ? "0.1" : "0.01";
        runner.run("wiki_gaussian_" + v_str, std::make_unique<Godunov>(), config, gaussian);
        runner.run("wiki_n_wave_" + v_str,   std::make_unique<Godunov>(), config, n_wave);
    }
}

// Basic demos: step function, sine wave, random wave, FTCS vs conservative FTCS.
void runBasic(ExperimentRunner& runner) {
    SolverConfig step_config = {
        .kinematic_viscosity = 0.01, .time_steps = 2000, .domain_length = 2.0, .spatial_step_size = 0.01,
    };
    std::function<double(double)> step = [](double x) { return (x >= 0.5 && x <= 1.0) ? 2.0 : 1.0; };
    runner.run("step_function", std::make_unique<Godunov>(), step_config, step);

    SolverConfig sine_config = {
        .kinematic_viscosity = 0.01, .time_steps = 5000, .domain_length = 2.0 * M_PI, .spatial_step_size = 0.01,
    };
    std::function<double(double)> sine = [](double x) { return std::sin(x); };
    runner.run("sine_wave", std::make_unique<Godunov>(), sine_config, sine);

    SolverConfig random_config = {
        .kinematic_viscosity = 0.01, .time_steps = 10000, .domain_length = 10.0, .spatial_step_size = 0.01,
    };
    RandomInitialConditionConfig ic_config;
    RandomInitialCondition f(ic_config, false, true);
    std::cout << f.toString() << std::endl;
    runner.run("random_function", std::make_unique<Godunov>(), random_config, f, f);

    RandomInitialCondition g(ic_config, false, true);
    runner.run("ftcs",              std::make_unique<FTCS>(),             random_config, g, g);
    runner.run("ftcs_conservative", std::make_unique<FTCSConservative>(), random_config, g, g);
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    ExperimentSettings settings;
    try {
        settings = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        printHelp();
        return 1;
    }

    ExperimentRunner runner(settings);
    runner.writeConfig();

    std::cout << "=== Experiment: " << settings.name << " ===\n"
              << "folder:    " << runner.folder() << "\n"
              << "viscosity: " << settings.nu << "\n"
              << "end time:  " << settings.end_time << "\n"
              << "sections: ";
    for (const std::string& section : settings.sections) std::cout << " " << section;
    std::cout << "\n";

    if (settings.sections.count("wiki"))  runWiki(runner);
    if (settings.sections.count("basic")) runBasic(runner);
    if (settings.sections.count("nn"))    runNNTest(runner, settings);
    if (settings.sections.count("dx"))    runDxSweep(runner, settings);

    std::cout << "\n=== Finished " << settings.name << ": " << runner.completed << " run(s) completed, "
              << runner.skipped << " skipped ===\n"
              << "Evaluate with (from the repo root):\n"
              << "    PYTHONPATH=lib/python python3 evaluate.py --experiment " << settings.name << "\n";
    return 0;
}