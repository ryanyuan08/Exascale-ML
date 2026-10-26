#include "burger_scheme.h"
#include <vector>

/*
    Godunov scheme for 1D Burgers equation:

        u_t + (u^2/2)_x = nu u_xx

    - First-order finite volume
    - Exact Riemann solver for Burgers
    - Conservative flux-difference form
    - Periodic boundary conditions
    - No artificial viscosity
*/
void Godunov::calculateNextU(
    const std::vector<double>& u,
    std::vector<double>& u_next,
    double /*cq*/,
    int n,
    double dt,
    double dx,
    double nu)
{
    // Calculate the Godunov flux at each interface.
    // Periodic boundary conditions connect the last point to the first.
    std::vector<double> F(n);

    for (int i = 0; i < n; ++i) {
        int ip = (i + 1) % n;
        F[i] = godunovFlux(u[i], u[ip]);
    }

    // Update every grid point using convection and diffusion.
    for (int i = 0; i < n; ++i) {
        // Neighboring indices with periodic boundary conditions.
        int ip = (i + 1) % n;
        int im = (i - 1 + n) % n;

        // Convective term: conservative Godunov flux difference.
        double convective = -(dt / dx) * (F[i] - F[im]);

        // Diffusion term: centered finite-difference approximation.
        double diffusion = (nu * dt / (dx * dx))
                         * (u[ip] - 2.0 * u[i] + u[im]);

        // Advance the solution by one time step.
        u_next[i] = u[i] + convective + diffusion;
    }
}

// Exact Godunov flux for Burgers equation
// Solves Riemann problem between states uLeft and uRight, returns physical flux f(u*) at interface

double Godunov::godunovFlux(double uLeft, double uRight) const
{
    // Burgers flux: f(u) = 0.5 u^2

    if (uLeft > uRight) {
        // Shock case [Shock speed from Rankine-Hugoniot: s = (uLeft + uRight)/2]
        double s = 0.5 * (uLeft + uRight);

        if (s > 0.0)
            return 0.5 * uLeft * uLeft;
        else
            return 0.5 * uRight * uRight;
    }
    else {
        // Rarefaction case
        if (uLeft >= 0.0)
            return 0.5 * uLeft * uLeft;
        else if (uRight <= 0.0)
            return 0.5 * uRight * uRight;
        else
            return 0.0;  // fan contains u = 0
    }
}


std::string Godunov::getName() const {
    return "Godunov";
}
