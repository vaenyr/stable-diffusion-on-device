#include "dpm_solver.h"
#include "utils.h"

#include <iostream>
#include <string>



int main() {
    libsdod::DPMSolver s(1000, 0.00085, 0.0120);
    std::cout << libsdod::format("all t: {}", s.get_all_t()) << std::endl;
    std::cout << libsdod::format("all log alpha: {}", s.get_all_log_alpha()) << std::endl;

    unsigned int steps;
    std::vector<unsigned int> ts;
    while (std::cin >> steps) {
        s.prepare(steps, ts);
        std::cout << libsdod::format("{}", ts) << std::endl;
        std::cout << libsdod::format("ts: {}", s.get_ts()) << std::endl;
        std::cout << libsdod::format("log alphas: {}", s.get_log_alphas()) << std::endl;
        std::cout << libsdod::format("lambdas: {}", s.get_lambdas()) << std::endl;
        std::cout << libsdod::format("sigmas: {}", s.get_sigmas()) << std::endl;
        std::cout << libsdod::format("alphas: {}", s.get_alphas()) << std::endl;
        std::cout << libsdod::format("phis: {}", s.get_phis()) << std::endl;
        std::cout << libsdod::format("i2rs: {}", s.get_i2rs()) << std::endl;
    }

    return 0;
}
