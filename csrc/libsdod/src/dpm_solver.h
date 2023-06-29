#ifndef LIBSDOD_DPM_SOLVER_H
#define LIBSDOD_DPM_SOLVER_H

#include <list>
#include <cmath>
#include <vector>


namespace libsdod {

template <class T>
void (std::vector<T>& buffer, T start, T end, unsigned int num_steps, unsigned int offset=0) {
    
}

class DPMSolver {
public:
    DPMSolver(unsigned int timesteps, double lin_start, double lin_end) {
        linspace(all_t, 0.0, 1.0, timesteps+1, 1);

        // calculate betas
        linspace(all_log_alpha, std::sqrt(lin_start), std::sqrt(lin_end), timesteps);
        for (auto& b : all_log_alpha)
            b = b*b;

        for (auto& b : all_log_alpha)
            b = 1 - b;

        for (auto i : range(1, timesteps)) {
            all_log_alpha[i] *= all_log_alpha[i-1];
        }

        for (auto& a : all_log_alpha) {
            a = 0.5 * std::log(a);
        }
    }

    void prepare(unsigned int steps);

    void update(unsigned int step, std::vector<float>& x);

private:
    // defined for the original time steps (e.g., 1000)
    std::vector<double> all_t;
    std::vector<double> all_log_alpha;

    // defined for a particular number of ODE steps (e.g., 20)
    std::vector<double> ts;
    std::vector<double> sigmas;
    std::vector<double> alphas;
    std::vector<double> hs;
    std::vector<double> e_negh_;

    std::list<std::vector<float>> prev;
};

}

#endif // LIBSDOD_DPM_SOLVER_H
