#pragma once

#include <string>
#include <torch/script.h>
#include <vector>

#include "nn.h"

class BurgerScheme {
  public:
    explicit BurgerScheme(bool use_nn = false) : use_nn(use_nn) {}

    // destructor
    virtual ~BurgerScheme() = default;

    // = 0 enforces its implementation in subclasses
    virtual void calculateNextU(const std::vector<double> &u, std::vector<double> &u_next, double cq,
                                int num_domain_points, double time_step_size, double spatial_step_size,
                                double kinematic_viscosity) = 0;

    // get the name of the scheme
    virtual std::string getName() const = 0;

    // whether this scheme calls the neural network during the solve
    virtual bool usesNN() const { return use_nn; }

  protected:
    bool use_nn;

    virtual double calculateArtificialViscosity(const std::vector<double> &, // u
                                                double,                      // cq
                                                double,                      // spatial_step_size
                                                double,                      // time_step_size
                                                int,                         // i
                                                int,                         // num_domain_points
                                                double) const                // kinematic_viscosity
    {
        return 0.0; // default: no artificial viscosity
    }
};

class FTCS : public BurgerScheme {
  public:
    explicit FTCS(bool use_nn = false);

    void calculateNextU(const std::vector<double> &u, std::vector<double> &u_next, double cq, int num_domain_points,
                        double time_step_size, double spatial_step_size, double kinematic_viscosity) override;

    std::string getName() const override;

  protected:
    mutable torch::jit::script::Module model;

    double calculateArtificialViscosity(const std::vector<double> &u, double cq, double spatial_step_size,
                                        double time_step_size, int i, int num_domain_points,
                                        double kinematic_viscosity) const override;
};

class FTCSConservative : public BurgerScheme {
  public:
    void calculateNextU(const std::vector<double> &u, std::vector<double> &u_next, double cq, int num_domain_points,
                        double time_step_size, double spatial_step_size, double kinematic_viscosity) override;

    std::string getName() const override;

  protected:
    double calculateArtificialViscosity(const std::vector<double> &u, double cq, double spatial_step_size,
                                        double time_step_size, int i, int num_domain_points,
                                        double kinematic_viscosity) const override;
};

class LaxWendroff : public BurgerScheme {
  public:
    explicit LaxWendroff(bool use_nn = false) : BurgerScheme(use_nn) {}

    void calculateNextU(const std::vector<double> &u, std::vector<double> &u_next, double cq, int num_domain_points,
                        double time_step_size, double spatial_step_size, double kinematic_viscosity) override;

    std::string getName() const override;

  protected:
    double calculateArtificialViscosity(const std::vector<double> &u, double cq, double spatial_step_size,
                                        double time_step_size, int i, int num_domain_points,
                                        double kinematic_viscosity) const override;
};

class Godunov : public BurgerScheme {
  public:
    explicit Godunov(bool use_nn = false) : BurgerScheme(use_nn) {}

    void calculateNextU(const std::vector<double> &u, std::vector<double> &u_next, double /*cq (unused)*/, int n,
                        double dt, double dx, double kinematic_viscosity) override;

    std::string getName() const override;

  protected:
    double godunovFlux(double uL, double uR) const;
};
