#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <chrono>
#include <string>

#include "burgers.h"


BurgersSolver1d::BurgersSolver1d(std::unique_ptr<BurgerScheme> scheme, const SolverConfig& config)
    :
        scheme(std::move(scheme)),
        kinematic_viscosity(config.kinematic_viscosity),
        reynolds_number(config.reynolds_number),
        time_steps(config.time_steps),
        domain_length(config.domain_length),
        spatial_step_size(config.spatial_step_size),
        solution_history(),
        nan_detected(false)
{}

BurgersSolver1d::BurgersSolver1d(
    std::unique_ptr<BurgerScheme> scheme,
    const SolverConfig& config,
    const std::function<double(double)>& initial_conditions
)
    :
        scheme(std::move(scheme)),
        kinematic_viscosity(config.kinematic_viscosity),
        reynolds_number(config.reynolds_number),
        time_steps(config.time_steps),
        domain_length(config.domain_length),
        spatial_step_size(config.spatial_step_size),
        solution_history(),
        initial_conditions(initial_conditions),
        nan_detected(false)
{
    max_u = compute_max_abs(initial_conditions, domain_length);
}

void BurgersSolver1d::setInitialConditions(const std::function<double(double)>& initial_conditions) {
    this->initial_conditions = initial_conditions;
    max_u = compute_max_abs(initial_conditions, domain_length);
}


void BurgersSolver1d::solve(double cq) {
    if (!initial_conditions) {
        throw std::runtime_error("No Initial Condition Function.");
    }
    
    nan_detected = false;
    num_domain_points = std::floor(domain_length / spatial_step_size);
    u.assign(num_domain_points, 0.0);

    double dt_advection = ALPHA * spatial_step_size / max_u;
    double effective_viscosity = std::max(kinematic_viscosity, 1e-8);
    double dt_diffusion = BETA * (spatial_step_size * spatial_step_size) / effective_viscosity;
    double time_step_size = std::min(dt_advection, dt_diffusion);

    computed_time_step_size = time_step_size;

    std::cout << "Computing with " << num_domain_points << " domain points.\n";
    std::cout << "max_u = " << max_u << ", dt_advection = " << dt_advection
              << ", dt_diffusion = " << dt_diffusion << " => using dt = " << time_step_size << "\n";

    std::cout << "Setting initial conditions...\n";
    for (int i = 0; i < num_domain_points; ++i) {
        double x = i * spatial_step_size;
        u[i] = initial_conditions(x);
    }

    solution_history.clear();
    std::vector<double> u_next(num_domain_points, 0.0);
    solution_history.push_back(u);

    std::cout << "Solving...\n";
    auto solve_start = std::chrono::steady_clock::now();
    for (int time_step = 0; time_step < time_steps; ++time_step) {
        scheme->calculateNextU(
            u, 
            u_next, 
            cq, 
            num_domain_points, 
            time_step_size, 
            spatial_step_size, 
            kinematic_viscosity
        );       
        
        // figure out if the solver crashed anywhere
        if (!nan_detected) {
            for (int i = 0; i < num_domain_points; i++) {
                if (std::isnan(u_next[i])) {
                    nan_detected = true;
                    break;
                }
            }
        }

        std::swap(u, u_next);
        solution_history.push_back(u);
    }
    auto solve_end = std::chrono::steady_clock::now();
    runtime_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
}

std::vector<std::vector<double>> BurgersSolver1d::getSolution() const {
    return solution_history;
}

void BurgersSolver1d::saveBinarySolution(const std::string& base_folder, const std::string& run_name) const {

    if (!std::filesystem::exists(base_folder)) {
        std::filesystem::create_directory(base_folder);
    }

    // Create (or recreate) run folder
    std::string run_folder = base_folder + "/" + run_name;
    if (std::filesystem::exists(run_folder)) {
        std::cout << "Overwriting existing folder: " << run_folder << std::endl;
        std::filesystem::remove_all(run_folder); // delete everything inside
    }
    std::filesystem::create_directory(run_folder);
    std::cout << "Writing results to: " << run_folder << std::endl;

	// Create and open binary file to hold solution data
    std::string filename = run_folder + "/solution.bin";
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::cout << "Writing binary solution to: " << filename << std::endl;

	// Add solution data to binary file, writing one timestep at a time
    for (size_t t = 0; t < solution_history.size(); ++t) {

        const std::vector<double>& u_t = solution_history[t];

        file.write(
            reinterpret_cast<const char*>(u_t.data()),
            u_t.size() * sizeof(double)
        );

        if (!file) {
            std::cerr << "Error writing timestep " << t << std::endl;
            return;
        }
    }
    file.close();
    std::cout << "Binary solution written successfully.\n";

}

void BurgersSolver1d::appendMetadata(nlohmann::json& metadata) const {
    metadata["solver"] = {
        {"time_steps", solution_history.size()},
        {"time_step_size", computed_time_step_size},
        {"num_domain_points", num_domain_points},
        {"spatial_step_size", spatial_step_size},
        {"domain_length", domain_length},
        {"kinematic_viscosity", kinematic_viscosity},
        {"reynolds_number", reynolds_number},
        {"scheme_name", scheme->getName()},
        {"use_nn", scheme->usesNN()},
        {"runtime_ms", runtime_seconds * 1000.0},
        {"nan_detected", nan_detected}
    };
}

bool BurgersSolver1d::wasNanDetected() const {
    return nan_detected;
}

double BurgersSolver1d::getMaxU() const {
    return max_u;
}

int BurgersSolver1d::getNumDomainPoints() const {
    return num_domain_points;
}

double BurgersSolver1d::getSpatialStepSize() const {
    return spatial_step_size;
}

double BurgersSolver1d::getRuntimeSeconds() const {
    return runtime_seconds;
}

std::string BurgersSolver1d::getSchemeName() const {
    return scheme->getName();
}
