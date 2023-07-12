#ifndef LIBSDOD_DPM_SOLVER_H
#define LIBSDOD_DPM_SOLVER_H

#include <list>
#include <cmath>
#include <vector>


namespace libsdod {

class DPMSolver {
public:
    DPMSolver(unsigned int timesteps, double lin_start, double lin_end);

    void prepare(unsigned int steps, std::vector<float>& model_ts);
    void update(unsigned int step, std::vector<float>& x,  std::vector<float>& y);

    auto& get_all_t() const { return all_t; }
    auto& get_all_log_alpha() const { return all_log_alpha; }
    auto& get_ts() const { return ts; }
    auto& get_log_alphas() const { return log_alphas; }
    auto& get_lambdas() const { return lambdas; }
    auto& get_sigmas() const { return sigmas; }
    auto& get_alphas() const { return alphas; }
    auto& get_phis() const { return phis; }
    auto& get_i2rs() const { return i2rs; }

private:
    unsigned int total_timesteps;

    // defined for the original time steps (e.g., 1000)
    std::vector<double> all_t;
    std::vector<double> all_log_alpha;

    // defined for a particular number of ODE steps (e.g., 20)
    std::vector<double> ts;
    std::vector<double> log_alphas;
    std::vector<double> lambdas;
    std::vector<double> sigmas;
    std::vector<double> alphas;
    std::vector<double> phis;
    std::vector<double> i2rs;

    std::vector<float> prev_y;
};

}

#endif // LIBSDOD_DPM_SOLVER_H
