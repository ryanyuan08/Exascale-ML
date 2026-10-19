#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <torch/script.h>

#include "burger_scheme.h"

FTCS::FTCS(bool use_nn) : BurgerScheme(use_nn) {

    if (use_nn) {
        // Each timestep forwards a small batch (only cells in compression), so
        // Torch's intra-op thread pool costs more than it saves and made run
        // times vary by up to an order of magnitude. One thread is faster and
        // deterministic for this workload.
        torch::set_num_threads(1);
        model = torch::jit::load(MODEL_PATH);
        model.eval();
    }
}

void FTCS::calculateNextU(const std::vector<double> &u, std::vector<double> &u_next, double cq, int num_domain_points,
                          double time_step_size, double spatial_step_size, double kinematic_viscosity) {
    // The plain (non-NN) path keeps the original scalar loop; only the NN path
    // is batched, so its numerics stay exactly as before.
    if (!use_nn) {
        for (int i = 1; i < num_domain_points - 1; ++i) {
            u_next[i] = u[i] - u[i] * time_step_size / (2.0 * spatial_step_size) * (u[i + 1] - u[i - 1]) +
                        (kinematic_viscosity + calculateArtificialViscosity(u, cq, spatial_step_size, time_step_size, i,
                                                                            num_domain_points, kinematic_viscosity)) *
                            time_step_size / (spatial_step_size * spatial_step_size) * (u[i + 1] - 2 * u[i] + u[i - 1]);
        }

        // wrap around
        // for some reason adding this artificial viscosity drastically slows the sim down.
        // u_next[0] = u[0]
        //     - u[0] * time_step_size / spatial_step_size * (u[0] - u[num_domain_points - 2])
        //     + (kinematic_viscosity + calculateArtificialViscosity(u, cq, spatial_step_size, 0, num_domain_points)) *
        //     time_step_size / (spatial_step_size * spatial_step_size)
        //     * (u[1] - 2 * u[0] + u[num_domain_points - 2]);
        u_next[0] = u[0] - u[0] * time_step_size / (2.0 * spatial_step_size) * (u[1] - u[num_domain_points - 2]) +
                    (kinematic_viscosity + calculateArtificialViscosity(u, cq, spatial_step_size, time_step_size, 0,
                                                                        num_domain_points, kinematic_viscosity)) *
                        time_step_size / (spatial_step_size * spatial_step_size) *
                        (u[1] - 2 * u[0] + u[num_domain_points - 2]);
        u_next[num_domain_points - 1] = u_next[0];
        return;
    }

    // Batched NN path.
    //
    // The network predicts cq per grid cell, but a separate forward pass per
    // cell is dominated by tensor construction and dispatch overhead (roughly
    // 15 us per call for a 12 -> 128 MLP). Instead, gather every cell in
    // compression (ux < 0, the only cells that receive artificial viscosity)
    // into one feature matrix and run a single forward pass per time step.
    //
    // The grid is periodic and the last cell duplicates cell 0, so index
    // arithmetic wraps modulo (num_domain_points - 1).
    const int n = num_domain_points;
    const int periodic_points = n - 1;
    auto wrap = [periodic_points](int idx) { return ((idx % periodic_points) + periodic_points) % periodic_points; };

    // Pass 1: central-difference gradients and the list of compression cells.
    ux_buffer_.resize(n);
    compression_indices_.clear();
    compression_indices_.reserve(n);
    // Neighbors match the original per-cell loop: i = 0 wraps to n-2 on the
    // left, while i = n-2 uses the duplicate cell n-1 on the right.
    for (int i = 0; i < n - 1; ++i) {
        const int im = (i == 0) ? n - 2 : i - 1;
        const int ip = i + 1;
        ux_buffer_[i] = (u[ip] - u[im]) / (2.0 * spatial_step_size);
        if (ux_buffer_[i] < 0.0) {
            compression_indices_.push_back(i);
        }
    }

    // Pass 2: build the feature matrix, one row per compression cell.
    // Feature order must match generate_feature_matrices.py exactly:
    // dt, dx, u_i, (u_{i+r} - u_i) for r = -U_RADIUS..-1 and 1..U_RADIUS, nu.
    const int num_compression = static_cast<int>(compression_indices_.size());
    cq_buffer_.assign(num_compression, static_cast<float>(cq));
    if (num_compression > 0) {
        feature_buffer_.resize(static_cast<size_t>(num_compression) * FEATURE_COUNT);
        float *features = feature_buffer_.data();
        for (int i : compression_indices_) {
            *features++ = static_cast<float>(time_step_size);
            *features++ = static_cast<float>(spatial_step_size);
            *features++ = static_cast<float>(u[i]);
            for (int r = -U_RADIUS; r <= U_RADIUS; ++r) {
                if (r == 0) continue; // center difference is always 0, so Python drops it
                *features++ = static_cast<float>(u[wrap(i + r)] - u[i]);
            }
            *features++ = static_cast<float>(kinematic_viscosity);
        }

        // One forward pass for the whole batch. No per-cell .item() calls: the
        // outputs are copied out in a single contiguous block.
        torch::NoGradGuard no_grad;
        torch::Tensor input = torch::from_blob(feature_buffer_.data(), {num_compression, FEATURE_COUNT},
                                               torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor output = model.forward({input}).toTensor().contiguous();
        std::memcpy(cq_buffer_.data(), output.data_ptr<float>(), static_cast<size_t>(num_compression) * sizeof(float));

        // Negative cq means negative viscosity (anti-smoothing), never allow it.
        for (float &value : cq_buffer_) {
            value = std::max(value, 0.0f);
        }
    }

    // Pass 3: update every physical cell using the batched cq values.
    // compression_indices_ is ascending, so a moving cursor matches each cell
    // with its network output.
    size_t compression_cursor = 0;
    for (int i = 0; i < n - 1; ++i) {
        const int im = (i == 0) ? n - 2 : i - 1;
        const int ip = i + 1;

        double artificial_viscosity = 0.0;
        if (ux_buffer_[i] < 0.0) {
            const double cq_nn = static_cast<double>(cq_buffer_[compression_cursor++]);
            artificial_viscosity = cq_nn * spatial_step_size * spatial_step_size * std::abs(ux_buffer_[i]);
        }

        u_next[i] = u[i] - u[i] * time_step_size / (2.0 * spatial_step_size) * (u[ip] - u[im]) +
                    (kinematic_viscosity + artificial_viscosity) * time_step_size /
                        (spatial_step_size * spatial_step_size) * (u[ip] - 2 * u[i] + u[im]);
    }
    u_next[n - 1] = u_next[0];
}

double FTCS::calculateArtificialViscosity(const std::vector<double> &u, double cq, double spatial_step_size,
                                          double /*time_step_size*/, int i, int num_domain_points,
                                          double /*kinematic_viscosity*/) const {
    // linear artificial viscosity model, not quadratic RVN
    double ux = (i == 0) ? (u[i + 1] - u[num_domain_points - 2]) / (2.0 * spatial_step_size)
                         : (u[i + 1] - u[i - 1]) / (2.0 * spatial_step_size);
    // Only take artificial viscosity when in compression. The NN variant of
    // this calculation now lives in the batched path of calculateNextU.
    if (ux >= 0) {
        return 0.0;
    }
    return cq * spatial_step_size * spatial_step_size * std::abs(ux);
}

std::string FTCS::getName() const { return "FTCS"; }
