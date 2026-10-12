#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include "burger_scheme.h"
#include "burgers.h"
#include "initial_condition_generator.h"
#include "metadata_writer.h"

int main() {
    
    // Select tests to run
    bool RUN_WIKI     = false;
    bool RUN_BASIC    = false;
    bool RUN_NN_TEST  = true;
    bool RUN_DX_SWEEP = true;

    if (RUN_WIKI) {
        std::cout << "Wikipedia Test Cases" << std::endl;

        SolverConfig wiki_config_base = {
            .kinematic_viscosity = 1.0, // Will be overwritten later
            .time_steps = 20000,
            .domain_length = 24.0, // longer than shown on wikipedia to prevent like the boundary conditions from messing stuff up
            .spatial_step_size = 0.01
        };

        // Gaussian Initial Condition: u(x,0) = e^(-x^2/2)
        std::function<double(double)> gaussian_function = [](double x) -> double {
            double shifted_x = x - 12.0; // Center the plot at x=12 to mimic [-12, 12]
            return std::exp(-(shifted_x * shifted_x) / 2.0);
        };

        // N-wave type Initial Condition: u(x,0) = e^(-(x-1)^2/2) - e^(-(x+1)^2/2)
        std::function<double(double)> n_wave_function = [](double x) -> double {
            double shifted_x = x - 12.0; // Center the plot at x=6 to mimic [-12, 12]
            double part1 = std::exp(-(std::pow(shifted_x - 1.0, 2)) / 2.0);
            double part2 = std::exp(-(std::pow(shifted_x + 1.0, 2)) / 2.0);
            return part1 - part2;
        };

        std::vector<double> viscosities = {1.0, 0.1, 0.01};

        for (double v : viscosities) {
            wiki_config_base.kinematic_viscosity = v;

            std::cout << "Solving Gaussian for v=" << v << "...\n";
            BurgersSolver1d gaussian_solver(std::make_unique<Godunov>(), wiki_config_base, gaussian_function);

            gaussian_solver.solve();

            std::cout << "gaussian function was nan detected: " << gaussian_solver.wasNanDetected() << std::endl;
            std::string v_str = (v == 1.0) ? "1.0" : (v == 0.1) ? "0.1" : "0.01";
            gaussian_solver.saveBinarySolution("../data", "wiki_gaussian_" + v_str);
            MetadataWriter(gaussian_solver).write("../data/wiki_gaussian_" + v_str + "/metadata.json");

            std::cout << "Solving N-wave for v=" << v << "...\n";
            BurgersSolver1d n_wave_solver(std::make_unique<Godunov>(), wiki_config_base, n_wave_function);

            n_wave_solver.solve();
            std::cout << "nwave function was nan detected: " << n_wave_solver.wasNanDetected() << std::endl;
            n_wave_solver.saveBinarySolution("../data", "wiki_n_wave_" + v_str);
            MetadataWriter(n_wave_solver).write("../data/wiki_n_wave_" + v_str + "/metadata.json");
        }
    }

    if (RUN_BASIC) {
        SolverConfig step_function_config = {
            .kinematic_viscosity = 0.01,
            .time_steps = 2000,
            .domain_length = 2.0,
            .spatial_step_size = 0.01
        };

        // 1 everywhere in domain
        // for 0.5 <= x <= 1 the function value is 2
        std::function<double(double)> step_function = [](double x) -> double {
            if (x >= 0.5 && x <= 1.0) {
                return 2.0;
            } else {
                return 1.0;
            }
        };

        BurgersSolver1d step_function_solver(std::make_unique<Godunov>(), step_function_config, step_function);

        step_function_solver.solve();
        step_function_solver.saveBinarySolution("../data", "step_function");
        MetadataWriter(step_function_solver).write("../data/step_function/metadata.json");
        std::cout << "step function was nan detected: " << step_function_solver.wasNanDetected() << std::endl;

        SolverConfig sine_wave_config = {
            .kinematic_viscosity = 0.01,
            .time_steps = 5000,
            .domain_length = 2.0 * M_PI,
            .spatial_step_size = 0.01
        };

        // initial condition: one full sine wave over [0, 2π]
        std::function<double(double)> sine_function = [](double x) -> double { 
            return std::sin(x);
        };

        BurgersSolver1d solver(std::make_unique<Godunov>(), sine_wave_config, sine_function);

        solver.solve();
        solver.saveBinarySolution("../data", "sine_wave");
        MetadataWriter(solver).write("../data/sine_wave/metadata.json");
        std::cout << "sine wave was nan detected: " << solver.wasNanDetected() << std::endl;

        RandomInitialConditionConfig functionConfig; // default values
        RandomInitialCondition f(functionConfig, false, true);
        std::cout << f.toString() << std::endl;

        SolverConfig random_function_config = {
            .kinematic_viscosity = 0.01,
            .time_steps = 10000,
            .domain_length = 10.0,
            .spatial_step_size = 0.01,
        };

        BurgersSolver1d random_function_solver(std::make_unique<Godunov>(), random_function_config, f);

        random_function_solver.solve();
        random_function_solver.saveBinarySolution("../data", "random_function");
        MetadataWriter(random_function_solver, f).write("../data/random_function/metadata.json");
        std::cout << "random function was nan detected: " << random_function_solver.wasNanDetected() << std::endl;
        
        RandomInitialConditionConfig ftcs_random_function_config;
        RandomInitialCondition g(ftcs_random_function_config, false, true);
        
        SolverConfig ftcs_solver_config = {
            .kinematic_viscosity = 0.01,
            .time_steps = 10000,
            .domain_length = 10.0,
            .spatial_step_size = 0.01,
        };
        
        BurgersSolver1d ftcs(std::make_unique<FTCS>(), ftcs_solver_config, g);
        BurgersSolver1d ftcs_conservative(std::make_unique<FTCSConservative>(), ftcs_solver_config, g);
        
        ftcs.solve();
        ftcs.saveBinarySolution("../data", "ftcs");
        MetadataWriter(ftcs, g).write("../data/ftcs/metadata.json");
        
        ftcs_conservative.solve();
        ftcs_conservative.saveBinarySolution("../data", "ftcs_conservative");
        MetadataWriter(ftcs_conservative, g).write("../data/ftcs_conservative/metadata.json");
    }

    if (RUN_NN_TEST) {
        RandomInitialConditionConfig nnRandomFunctionConfig;
        RandomInitialCondition nn_g(nnRandomFunctionConfig, false, true, 42);

        SolverConfig fine_grain_config = {
            .kinematic_viscosity = 0.8,
            .time_steps = 20000,
            .domain_length = 20.0,
            .spatial_step_size = 0.005,
        };

        SolverConfig coarse_grain_config = {
            .kinematic_viscosity = 0.8,
            .time_steps = 5000,
            .domain_length = 20.0,
            .spatial_step_size = 0.01,
        };

        BurgersSolver1d godunov_fine_grain_solver(std::make_unique<Godunov>(), fine_grain_config, nn_g);
        BurgersSolver1d ftcs_fine_grain_solver(std::make_unique<FTCS>(), fine_grain_config, nn_g);
        BurgersSolver1d godunov_coarse_grain_solver(std::make_unique<Godunov>(), coarse_grain_config, nn_g);
        BurgersSolver1d ftcs_coarse_grain_solver(std::make_unique<FTCS>(), coarse_grain_config, nn_g);
        BurgersSolver1d ftcs_coarse_grain_nn_solver(std::make_unique<FTCS>(true), coarse_grain_config, nn_g);

        godunov_fine_grain_solver.solve();
        godunov_fine_grain_solver.saveBinarySolution("../data", "godunov_fine_grain");
        MetadataWriter(godunov_fine_grain_solver, nn_g).write("../data/godunov_fine_grain/metadata.json");
        std::cout << "godunov fine grain was nan detected: " << godunov_fine_grain_solver.wasNanDetected() << std::endl;

        ftcs_fine_grain_solver.solve();
        ftcs_fine_grain_solver.saveBinarySolution("../data", "ftcs_fine_grain");
        MetadataWriter(ftcs_fine_grain_solver, nn_g).write("../data/ftcs_fine_grain/metadata.json");
        std::cout << "ftcs fine grain was nan detected: " << ftcs_fine_grain_solver.wasNanDetected() << std::endl;

        godunov_coarse_grain_solver.solve();
        godunov_coarse_grain_solver.saveBinarySolution("../data", "godunov_coarse_grain");
        MetadataWriter(godunov_coarse_grain_solver, nn_g).write("../data/godunov_coarse_grain/metadata.json");
        std::cout << "godunov coarse grain was nan detected: " << godunov_coarse_grain_solver.wasNanDetected() << std::endl;

        ftcs_coarse_grain_solver.solve();
        ftcs_coarse_grain_solver.saveBinarySolution("../data", "ftcs_coarse_grain");
        MetadataWriter(ftcs_coarse_grain_solver, nn_g).write("../data/ftcs_coarse_grain/metadata.json");
        std::cout << "ftcs coarse grain was nan detected: " << ftcs_coarse_grain_solver.wasNanDetected() << std::endl;

        ftcs_coarse_grain_nn_solver.solve();
        ftcs_coarse_grain_nn_solver.saveBinarySolution("../data", "ftcs_coarse_grain_nn");
        MetadataWriter(ftcs_coarse_grain_nn_solver, nn_g).write("../data/ftcs_coarse_grain_nn/metadata.json");
        std::cout << "ftcs coarse grain nn was nan detected: " << ftcs_coarse_grain_nn_solver.wasNanDetected() << std::endl;
    }

    if (RUN_DX_SWEEP) {
        // dx sweep: Godunov ground truth vs NN-FTCS at various spatial step sizes
        RandomInitialConditionConfig dx_sweep_ic_config;
        RandomInitialCondition dx_sweep_ic(dx_sweep_ic_config, false, true, 7);

        SolverConfig godunov_gt_config = {
            .kinematic_viscosity = 0.8,
            .time_steps = 20000,
            .domain_length = 20.0,
            .spatial_step_size = 0.005,
        };

        BurgersSolver1d godunov_gt_solver(std::make_unique<Godunov>(), godunov_gt_config, dx_sweep_ic);
        godunov_gt_solver.solve();
        godunov_gt_solver.saveBinarySolution("../data", "godunov_gt");
        MetadataWriter(godunov_gt_solver, dx_sweep_ic).write("../data/godunov_gt/metadata.json");
        std::cout << "godunov gt nan detected: " << godunov_gt_solver.wasNanDetected() << std::endl;

        std::vector<std::pair<double, std::string>> nn_dx_sweep = {
            {0.01, "0p01"}, {0.02, "0p02"}, {0.04, "0p04"},
            {0.05, "0p05"}, {0.08, "0p08"}, {0.16, "0p16"}
        };

        for (auto& [dx, dx_str] : nn_dx_sweep) {
            double end_time = 0.125;
            double dt = 0.2 * dx * dx / 0.8;
            int steps = static_cast<int>(end_time / dt);
            SolverConfig dx_config = {
                .kinematic_viscosity = 0.8,
                .time_steps = steps,
                .domain_length = 20.0,
                .spatial_step_size = dx,
            };

            // plain FTCS baseline with the same grid / IC but no NN call
            std::string ftcs_name = "ftcs_dx_" + dx_str;
            BurgersSolver1d ftcs_dx_solver(std::make_unique<FTCS>(), dx_config, dx_sweep_ic);
            ftcs_dx_solver.solve();
            ftcs_dx_solver.saveBinarySolution("../data", ftcs_name);
            MetadataWriter(ftcs_dx_solver, dx_sweep_ic).write("../data/" + ftcs_name + "/metadata.json");
            std::cout << ftcs_name << " nan detected: " << ftcs_dx_solver.wasNanDetected() << std::endl;

            // NN-FTCS on the same grid / IC
            std::string nn_name = "nn_dx_" + dx_str;
            BurgersSolver1d nn_dx_solver(std::make_unique<FTCS>(true), dx_config, dx_sweep_ic);
            nn_dx_solver.solve();
            nn_dx_solver.saveBinarySolution("../data", nn_name);
            MetadataWriter(nn_dx_solver, dx_sweep_ic).write("../data/" + nn_name + "/metadata.json");
            std::cout << nn_name << " nan detected: " << nn_dx_solver.wasNanDetected() << std::endl;
        }
    }

}