#pragma once

#include <string>
#include <random>
#include <vector>
#include <ostream>

#include "metadata_writer.h"


// shape of the power spectrum S(k)
enum class SpectrumType {
    Gaussian,
    PowerLaw,
    Flat
};

// settings for one random curve
struct GaussianRandomFieldConfig {
    double domain_length = 10.0;
    int max_mode = 40;
    SpectrumType spectrum = SpectrumType::Gaussian;
    double cutoff_mode = 5.0;           // bigger = rougher (Gaussian only)
    double power_law_exponent = 2.0;    // PowerLaw only
    double mean = 0.0;
    double target_max_u = 1.0;          // curve is scaled so its max equals this
};

class GaussianRandomFieldInitialCondition : public IMetadataProvider {

public:

    // one sine wave, same layout as RandomInitialCondition
    struct Term {
        double amplitude;
        double frequency;
        double phase_shift;
    };

    GaussianRandomFieldInitialCondition(
                   const GaussianRandomFieldConfig& config,
                   unsigned seed = std::random_device{}());

    // value of the curve at x
    double operator()(double x) const;

    double getMaxU() const;

    double getDomainLength() const;

    // strength S(k) of wave number k
    static double spectrum(const GaussianRandomFieldConfig& config, int k);

    void appendMetadata(nlohmann::json& metadata) const override;

    std::string toString() const;

private:
    GaussianRandomFieldConfig config;
    std::vector<Term> terms;
    double bias;
    double max_u;
};