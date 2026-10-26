#include <iostream>
#include <random>
#include <filesystem>
#include <memory>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <vector>

#include "initial_condition_generator.h"
#include "burgers.h"
#include "metadata_writer.h"

// how many training samples to generate
const int DEFAULT_NUM_SAMPLES = 5;
const std::string TRAINING_DIR = "../training_data";

// what scheme we are using for the configuration
const auto SCHEME_FACTORY = []() {
    return std::make_unique<Godunov>();
};

// Reynolds numbers to choose between per sample; kinematic viscosity is then
// derived as (|maxU| * domain_length) / reynolds_number
const std::vector<int> REYNOLDS_NUMBERS = {1000};

const int DEFAULT_TIME_STEPS = 10000;
const double DX_MIN = 0.001;
const double DX_MAX = 0.1;
const double SPATIAL_STEP_OVERRIDE_SENTINEL = -1.0;

// n (number of terms)
const int N_MIN = 1;
const int N_MAX = 20;

// domain length
const double DOMAIN_MIN = 2.0;
const double DOMAIN_MAX = 100.0;

// amplitude range
const double AMP_MIN = -10.0;
const double AMP_MAX = 10.0;

// frequency multiplier range
const double FREQ_MIN = 0.1;
const double FREQ_MAX = 20.0;

// wrap-around frequency multiplier
const int WRAP_K_MIN = 1;
const int WRAP_K_MAX = 20;
const int WRAP_K_DELTA_MIN = 2;
const int WRAP_K_DELTA_MAX = 10;

RandomInitialConditionConfig generateRandomInitialConditionConfig(unsigned int seed) {
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> n_dist(N_MIN, N_MAX);
    std::uniform_real_distribution<double> domain_dist(DOMAIN_MIN, DOMAIN_MAX);
    std::uniform_real_distribution<double> amp_dist(AMP_MIN, AMP_MAX);
    std::uniform_real_distribution<double> freq_dist(FREQ_MIN, FREQ_MAX);
    std::uniform_int_distribution<int> wrap_k_dist(WRAP_K_MIN, WRAP_K_MAX);

    RandomInitialConditionConfig cfg;

    cfg.n = n_dist(rng);
    cfg.domain_length = domain_dist(rng);

    // Amplitude
    double a1 = amp_dist(rng);
    double a2 = amp_dist(rng);
    cfg.amp_min = std::min(a1, a2);
    cfg.amp_max = std::max(a1, a2);

    // Frequency multiplier
    double f1 = freq_dist(rng);
    double f2 = freq_dist(rng);
    cfg.frequency_multiplier_min = std::min(f1, f2);
    cfg.frequency_multiplier_max = std::max(f1, f2);

    // Wrap-around frequency multiplier
    const int wrap_delta_max = std::min(WRAP_K_DELTA_MAX, WRAP_K_MAX - WRAP_K_MIN);
    std::uniform_int_distribution<int> wrap_delta_dist(WRAP_K_DELTA_MIN, wrap_delta_max);
    // Sample a min frequency and a delta to ensure min != max
    int wrap_min = wrap_k_dist(rng);
    int wrap_delta = wrap_delta_dist(rng);
    // Clamp if min + delta would exceed the max boundary
    if (wrap_min + wrap_delta > WRAP_K_MAX) {
        wrap_min = WRAP_K_MAX - wrap_delta;
    }
    cfg.wrap_around_frequency_multiplier_min = wrap_min;
    cfg.wrap_around_frequency_multiplier_max = wrap_min + wrap_delta;

    return cfg;
}

int findNextSampleNumber() {
    int max_sample = -1;
    
    if (!std::filesystem::exists(TRAINING_DIR)) {
        return 0;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(TRAINING_DIR)) {
        if (entry.is_directory()) {
            std::string folder_name = entry.path().filename().string();
            if (folder_name.substr(0, 7) == "sample_") {
                try {
                    int sample_num = std::stoi(folder_name.substr(7));
                    max_sample = std::max(max_sample, sample_num);
                } catch (...) {
                    // ignore folders that don't match the pattern
                }
            }
        }
    }
    
    return max_sample + 1;
}

void deleteExistingSamples() {
    if (std::filesystem::exists(TRAINING_DIR)) {
        std::cout << "Clearing existing training data...\n";
        for (const auto& entry : std::filesystem::directory_iterator(TRAINING_DIR)) {
            if (entry.is_directory() && entry.path().filename().string().substr(0, 7) == "sample_") {
                std::filesystem::remove_all(entry.path());
            }
        }
    }
}

void parseCommandLineArguments(int argc, char* argv[], int& num_samples, bool& append_mode, int& time_steps, double& spatial_step_size, unsigned int &seed) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--samples" || arg == "-s") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                std::exit(1);
            }
            try {
                num_samples = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << "\n";
                std::exit(1);
            }
        }
        else if (arg == "--append" || arg == "-a") {
            append_mode = true;
        }
        else if (arg == "--spatial-step" || arg == "-ss") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                std::exit(1);
            }
            try {
                spatial_step_size = std::stod(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << "\n";
                std::exit(1);
            }
        }
        else if (arg == "--num-time-steps" || arg == "-nts") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                std::exit(1);
            }
            try {
                time_steps = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << "\n";
                std::exit(1);
            }
        }
        else if (arg == "--seed" || arg == "-sd") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                std::exit(1);
            }
            try {
                seed = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value for " << arg << ": " << argv[i] << "\n";
                std::exit(1);
            }
        }
        else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            std::cerr << "Valid arguments:\n";
            std::cerr << "  --samples, -s <num>            Number of samples to generate\n";
            std::cerr << "  --append, -a                   Append to existing samples\n";
            std::cerr << "  --spatial-step, -ss <value>    Spatial step size (dx); dt is computed from this\n";
            std::cerr << "  --num-time-steps, -nts <num>   Number of time steps\n";
            std::cerr << "  --seed, -sd <num>              Seed for randomly generated functions\n";
            std::exit(1);
        }
    }
}

int main(int argc, char* argv[]) {

    int num_samples = DEFAULT_NUM_SAMPLES;
    bool append_mode = false;
    int time_steps = DEFAULT_TIME_STEPS;
    double spatial_step_size = SPATIAL_STEP_OVERRIDE_SENTINEL;
    unsigned int seed = std::random_device{}();

    parseCommandLineArguments(argc, argv, num_samples, append_mode, time_steps, spatial_step_size, seed);
    
    int start_sample_index = 0;
    
    if (append_mode) {
        start_sample_index = findNextSampleNumber();
        std::cout << "Append mode: starting from sample " << start_sample_index << "\n";
    } else {
        deleteExistingSamples();
        std::cout << "Starting from sample 0\n";
    }
    
    if (spatial_step_size > 0.0) {
        std::cout << "Generating " << num_samples << " samples with "
                  << time_steps << " time steps and fixed spatial step size " << spatial_step_size
                  << " (dt computed per sample from stability conditions)\n";
    } else {
        std::cout << "Generating " << num_samples << " samples with "
                  << time_steps << " time steps and dx sampled log-uniformly per sample ["
                  << DX_MIN << ", " << DX_MAX << "]\n";
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int sample_index = start_sample_index; sample_index < start_sample_index + num_samples; ++sample_index) {
        // make sure each sample is different, but if we do seperate runs with same seed we can get same functions
        unsigned int current_seed = seed + (sample_index - start_sample_index);

        auto sample_start = std::chrono::high_resolution_clock::now();

        RandomInitialConditionConfig cfg = generateRandomInitialConditionConfig(current_seed);
        RandomInitialCondition f(cfg, false, true, current_seed);
        
        // pick a Reynolds number and derive kinematic viscosity from it
        std::mt19937 reynolds_rng(current_seed);
        std::uniform_int_distribution<int> reynolds_index_distribution(0, static_cast<int>(REYNOLDS_NUMBERS.size()) - 1);
        int reynolds_number = REYNOLDS_NUMBERS[reynolds_index_distribution(reynolds_rng)];
        double kinematic_viscosity = (f.getMaxU() * cfg.domain_length) / reynolds_number;

        double sample_dx;
        if (spatial_step_size > 0.0) {
            sample_dx = spatial_step_size;
        } else {
            std::mt19937 dx_rng(current_seed ^ 0xDEADBEEF);
            std::uniform_real_distribution<double> log_dx_dist(std::log(DX_MIN), std::log(DX_MAX));
            sample_dx = std::exp(log_dx_dist(dx_rng));
        }

        SolverConfig solver_cfg;
        solver_cfg.kinematic_viscosity = kinematic_viscosity;
        solver_cfg.reynolds_number = reynolds_number;
        solver_cfg.time_steps = time_steps;
        solver_cfg.domain_length = cfg.domain_length;
        solver_cfg.spatial_step_size = sample_dx;

        BurgersSolver1d solver(
            SCHEME_FACTORY(),
            solver_cfg,
            f
        );

        solver.solve();
        std::cout << "random function was nan detected: " << solver.wasNanDetected() << std::endl;

        std::ostringstream folder_name;
        folder_name << "sample_" << std::setw(6) << std::setfill('0') << sample_index;

        solver.saveBinarySolution(TRAINING_DIR, folder_name.str());
        MetadataWriter(solver, f).write(
            std::filesystem::path(TRAINING_DIR) / folder_name.str() / "metadata.json"
        );

        auto sample_end = std::chrono::high_resolution_clock::now();
        auto sample_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sample_end - sample_start);

        std::cout << "Generated sample " << sample_index << " in "
                  << TRAINING_DIR << "/" << folder_name.str()
                  << " (dx=" << sample_dx << ", took " << sample_duration.count() / 1000.0 << "s)\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    std::cout << "\n===========================================\n";
    std::cout << "Total execution time: " << total_duration.count() / 1000.0 << "s\n";
    std::cout << "Average time per sample: " << (total_duration.count() / 1000.0) / num_samples << "s\n";
    std::cout << "===========================================\n";

    return 0;
}
