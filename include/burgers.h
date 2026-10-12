#pragma once
#include <functional>
#include <vector>
#include <memory>

#include "burger_scheme.h"
#include "math_utils.h"
#include "metadata_writer.h"

struct SolverConfig {
    // viscosity coefficient for viscous burgers
    double kinematic_viscosity;
    // Reynolds number used to derive kinematic viscosity; 0 if not set via Re
    int reynolds_number = 0;
    // how many iterations we will simulate
    int time_steps;
    // how long is the x axis are equation is defined on
    double domain_length;
    // spacing between adjacent grid points; dt is computed from this
    double spatial_step_size;
};

class BurgersSolver1d : public IMetadataProvider {

public:
    // CFL safety factor for the advection stability condition: dt_advection = ALPHA * dx / max_u
    inline static const double ALPHA = 0.4;
    // Safety factor for the explicit diffusion stability condition: dt_diffusion = BETA * dx^2 / nu
    inline static const double BETA = 0.2;

private:

    // scheme for solving
    std::unique_ptr<BurgerScheme> scheme;

    // kinematic viscosity as shown in the burgers equation
    const double kinematic_viscosity;

    // Reynolds number used to derive kinematic viscosity; 0 if not set via Re
    const int reynolds_number;

    // how many times will we run our solver
    const int time_steps;

    // how long is our initial condition domain (x axis)
    const double domain_length;

    // spacing between adjacent grid points (input); dt is derived from this
    const double spatial_step_size;

    // the current velocity field
    std::vector<double> u;

    // stores all u's as we solve
    std::vector<std::vector<double>> solution_history;

    // the function that we are using to define our initial conditions
    std::function<double(double)> initial_conditions;

    // how many points we calculate the value for in our domain; set during solve()
    int num_domain_points;

    // the time step size computed during solve() from the stability conditions
    double computed_time_step_size;

    // wall-clock time spent in solve(), in seconds
    double runtime_seconds = 0.0;

    // max absolute value of the initial condition, computed when IC is set
    double max_u = 0.0;

    // tracks if a NaN was found during the solve
    bool nan_detected;

public:
    
    BurgersSolver1d(
        const std::unique_ptr<BurgerScheme> scheme, 
        const SolverConfig& config);

    BurgersSolver1d(
        const std::unique_ptr<BurgerScheme> scheme, 
        const SolverConfig& config, 
        const std::function<double(double)>& initialize_conditions);

    void setInitialConditions(const std::function<double(double)>& initialize_conditions);

    void solve(double cq = 2.0);

    std::vector<std::vector<double>> getSolution() const;

    // Writes the solve to to solution.bin in the folder
    // Metadata is written separately use MetadataWriter with this object as a provider
    void saveBinarySolution(const std::string& base_folder, const std::string& run_name) const;

    void appendMetadata(nlohmann::json& metadata) const override;

    bool wasNanDetected() const;

    double getMaxU() const;

    int getNumDomainPoints() const;

    double getSpatialStepSize() const;

    double getRuntimeSeconds() const;

    std::string getSchemeName() const;
};