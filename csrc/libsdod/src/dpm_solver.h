#ifndef LIBSDOD_DPM_SOLVER_H
#define LIBSDOD_DPM_SOLVER_H

#include <list>
#include <cmath>
#include <vector>


namespace libsdod {

class DPMSolver {
public:
    using value_type = float;

public:
    DPMSolver(unsigned int timesteps, value_type lin_start, value_type lin_end);

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
    std::vector<value_type> all_t;
    std::vector<value_type> all_log_alpha;

    // defined for a particular number of ODE steps (e.g., 20)
    std::vector<value_type> ts;
    std::vector<value_type> log_alphas;
    std::vector<value_type> lambdas;
    std::vector<value_type> sigmas;
    std::vector<value_type> alphas;
    std::vector<value_type> phis;
    std::vector<value_type> i2rs;

    std::vector<float> prev_y;
};

}

#endif // LIBSDOD_DPM_SOLVER_H
