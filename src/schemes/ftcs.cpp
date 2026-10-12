#include <iostream>
#include <torch/script.h>

#include "burger_scheme.h"

FTCS::FTCS(bool use_nn) 
    :   BurgerScheme(use_nn) {
    
    if (use_nn) {
        model = torch::jit::load(MODEL_PATH);
        model.eval();
    }
}

void FTCS::calculateNextU(const std::vector<double>& u, std::vector<double>& u_next, double cq, int num_domain_points, double time_step_size, double spatial_step_size, double kinematic_viscosity) {
    for (int i = 1; i < num_domain_points - 1; ++i) {
        u_next[i] = u[i]
            - u[i] * time_step_size / (2.0 * spatial_step_size) * (u[i + 1] - u[i - 1])
            + (kinematic_viscosity + calculateArtificialViscosity(u, cq, spatial_step_size, time_step_size, i, num_domain_points, kinematic_viscosity)) * time_step_size / (spatial_step_size * spatial_step_size)
                * (u[i + 1] - 2 * u[i] + u[i - 1]);
    }

    // wrap around
    // for some reason adding this artificial viscosity drastically slows the sim down.
    // u_next[0] = u[0]
    //     - u[0] * time_step_size / spatial_step_size * (u[0] - u[num_domain_points - 2])
    //     + (kinematic_viscosity + calculateArtificialViscosity(u, cq, spatial_step_size, 0, num_domain_points)) * time_step_size / (spatial_step_size * spatial_step_size)
    //     * (u[1] - 2 * u[0] + u[num_domain_points - 2]);
    u_next[0] = u[0]
        - u[0] * time_step_size / (2.0 * spatial_step_size) * (u[1] - u[num_domain_points - 2])
        + (kinematic_viscosity + calculateArtificialViscosity(u, cq, spatial_step_size, time_step_size, 0, num_domain_points, kinematic_viscosity))
          * time_step_size / (spatial_step_size * spatial_step_size)
          * (u[1] - 2 * u[0] + u[num_domain_points - 2]);
    u_next[num_domain_points - 1] = u_next[0];
}

double FTCS::calculateArtificialViscosity(const std::vector<double>& u, double cq, double spatial_step_size, double time_step_size, int i, int num_domain_points, double kinematic_viscosity) const {
    // linear artificial viscosity model, not quadratic RVN
    double ux = (i == 0) ? (u[i + 1] - u[num_domain_points - 2]) / (2.0 * spatial_step_size) : (u[i + 1] - u[i - 1]) / (2.0 * spatial_step_size);
    // Only take artificial viscosity when in compression
    double artificial_viscosity;
    if (ux >= 0) {
        artificial_viscosity = 0.0;
    } else if (use_nn) {
        // circular array: first and last elements are identical, so valid range is [0, num_domain_points-2]
        // wrap indices using modulo on (num_domain_points - 1)
        auto wrap = [&](int idx) {
            int n = num_domain_points - 1;
            return ((idx % n) + n) % n;
        };

        // Must match generate_feature_matrices.py exactly:
        // dt, dx, u_i, (u_{i+r} - u_i) for r = -U_RADIUS..-1 and 1..U_RADIUS, nu
        std::vector<float> features;
        features.push_back(static_cast<float>(time_step_size));
        features.push_back(static_cast<float>(spatial_step_size));
        features.push_back(static_cast<float>(u[i]));

        for (int r = -U_RADIUS; r <= U_RADIUS; ++r) {
            if (r == 0) continue;   // center difference is always 0, so Python drops it
            features.push_back(static_cast<float>(u[wrap(i + r)] - u[i]));
        }

        features.push_back(static_cast<float>(kinematic_viscosity));

        torch::Tensor input = torch::from_blob(features.data(), {1, (long)features.size()}).clone();
        auto output = model.forward({input}).toTensor();
        cq = static_cast<double>(output.item<float>());
        cq = std::max(cq, 0.0);   // negative cq = negative viscosity (anti-smoothing), never allow it
        
        // uncomment if you want to see the actual cq values the network is pumping out
        // auto flat = input.flatten();
        // for (int j = 0; j < flat.size(0); ++j) {
        //     std::cout << flat[j].item<float>();
        //     if (j < flat.size(0) - 1) std::cout << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << "cq is: " << cq << std::endl;

        artificial_viscosity = cq * spatial_step_size * spatial_step_size * std::abs(ux);
    } else {
        artificial_viscosity = cq * spatial_step_size * spatial_step_size * std::abs(ux);
    }

    return artificial_viscosity;
}

std::string FTCS::getName() const {
    return "FTCS";
}
