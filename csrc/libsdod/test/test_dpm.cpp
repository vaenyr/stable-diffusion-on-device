#include "dpm_solver.h"
#include "utils.h"

#include <iostream>
#include <string>
#include <sstream>


std::string format_long(std::vector<libsdod::DPMSolver::value_type> const& v) {
    std::ostringstream ss;
    ss << "\n    size: " << v.size();
    unsigned int i = 0;
    while (i < v.size()) {
        if (i % 10 == 0)
            ss << "\n    ";
        ss << libsdod::format("{}", v[i]) << " ";
        i += 1;
    }
    if (v.empty())
        ss << "\n";
    return ss.str();
}



int main() {
    libsdod::DPMSolver s(1000, 0.00085, 0.0120);
    std::cout << libsdod::format("all t: {}", format_long(s.get_all_t())) << std::endl;
    std::cout << libsdod::format("all log alpha: {}", format_long(s.get_all_log_alpha())) << std::endl;

    unsigned int steps;
    std::vector<float> ts;
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

        std::vector<float> x;
        std::vector<float> y;
        x.resize(5);
        y.resize(5);
        for (auto i : libsdod::range(steps))
            s.update(i, x, y);
    }

    return 0;
}
