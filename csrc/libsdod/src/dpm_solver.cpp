#include "dpm_solver.h"
#include "errors.h"
#include "utils.h"

#include <iostream>
#include <cassert>


namespace libsdod {
namespace {

template <class T>
void linspace(std::vector<T>& buffer, T start, T end, unsigned int num_steps, unsigned int offset=0) {
    buffer.resize(num_steps-offset);
    double step = static_cast<double>(end - start) / (num_steps - 1);

    int insert = 0;
    for (auto i : range(num_steps)) {
        (void)i;
        if (!offset)
            buffer[insert++] = start;
        else
            --offset;
        start += step;
    }
}

inline double _interpolate(double x, double x1, double y1, double x2, double y2) {
    double a = (y2 - y1) / (x2 - x1);
    return a*(x - x1) + y1;
}


double interpolate(double x, std::vector<double> const& xs, std::vector<double> const& ys, unsigned int& hint) {
    // note: this method assumes xs is ascending, and x is descending over multiple calls with the same hint
    // initialize hint to xs.size()
    if (x < xs.front() || x > xs.back())
        return _interpolate(x, xs.back(), ys.back(), xs.front(), ys.front());
    while (xs[hint-1] > x) {
        assert(hint > 1);
        --hint;
    }
    //rev_hint should be larger than x
    //rev_hint-1 should be smaller than x
    assert(hint > 0 && hint < xs.size());
    assert(xs[hint] > x);
    assert(xs[hint-1] < x);

    if (!hint || hint > xs.size())
        throw libsdod_exception(ErrorCode::INTERNAL_ERROR, "Unreachable", __func__, __FILE__, STR(__LINE__));
    return _interpolate(x, xs[hint-1], ys[hint-1], xs[hint], ys[hint]);
}


template <class T>
void scale(std::vector<T>& v, T a) {
    for (auto& e : v)
        e *= a;
}


template <class T>
void accumulate(std::vector<T>& v1, std::vector<T> const& v2, T a) {
    for (auto i : range(v1.size()))
        v1[i] += a*v2[i];
}


template <class T>
void normalize(std::vector<T>& out, std::vector<T> const& v1, std::vector<T> const& v2, T a, T b) {
    for (auto i : range(v1.size()))
        out[i] = (v1[i] + a*v2[i]) / b;
}


}
}

using namespace libsdod;


DPMSolver::DPMSolver(unsigned int timesteps, double lin_start, double lin_end) : total_timesteps(timesteps) {
    linspace(all_t, 0.0, 1.0, timesteps+1, 1);

    // calculate sqrt(betas)
    linspace(all_log_alpha, std::sqrt(lin_start), std::sqrt(lin_end), timesteps);

    // calculate log cum_alphas
    auto cum = 1.0;
    for (auto& b : all_log_alpha) {
        b = 1 - b*b; //alpha
        cum *= b; //cumulative product of alphas
        b = 0.5 * std::log(cum); //log cum alpha
    }
}


void DPMSolver::prepare(unsigned int steps, std::vector<float>& model_ts) {
    double first_t = 1.0;
    double last_t = 1.0 / total_timesteps;
    linspace(ts, first_t, last_t, steps+1);

    model_ts.resize(ts.size());
    log_alphas.resize(ts.size());
    lambdas.resize(ts.size());
    sigmas.resize(ts.size());
    alphas.resize(ts.size());
    phis.resize(ts.size());
    i2rs.resize(ts.size());

    unsigned int interpolate_hint = all_t.size();
    for (auto i : range(ts.size())) {
        model_ts[i] = ((ts[i] - 1.0 / total_timesteps)*1000); // TODO: the choice of 1000 seems quite arbitrary in the DPM code, but it is what it is...
        log_alphas[i] = interpolate(ts[i], all_t, all_log_alpha, interpolate_hint);
        lambdas[i] = log_alphas[i] - (0.5 * std::log(1  - std::exp(2 * log_alphas[i])));
        sigmas[i] = std::sqrt(1 - std::exp(2 * log_alphas[i]));
        alphas[i] = std::exp(log_alphas[i]);

        if (i)
            phis[i] = std::expm1(-(lambdas[i] - lambdas[i-1]));
        else
            phis[i] = INFINITY;

        if (i>=2)
            i2rs[i] = 1.0 / (2*((lambdas[i-1] - lambdas[i-2]) / (lambdas[i] - lambdas[i-1])));
        else
            i2rs[i] = INFINITY;
    }
}

using fs = std::initializer_list<double>;


void DPMSolver::update(unsigned int step, std::vector<float>& x,  std::vector<float>& y) {
    auto order = (step == 0 ? 1 : (step < 10 ? std::min<unsigned int>(2, ts.size() - step) : 2));
    // switch from noise prediction to data prediction
    normalize<float>(y, x, y, -sigmas[step+1], 1.0/alphas[step+1]); // y = (x + (-sigma)*y) * (1/alpha) = (x - sigma*y) / alpha

    switch (order) {
    case 1:
#ifdef LIBSDOD_DEBUG
        std::cout << format("SS DPM, t: {}, lambda: {}, log(a): {}, sigma: {}, alpha: {}, phi: {}",
            fs{ ts[step], ts[step+1] },
            fs{ lambdas[step], lambdas[step+1] },
            fs{ log_alphas[step], log_alphas[step+1] },
            fs{ sigmas[step], sigmas[step+1] },
            alphas[step+1],
            phis[step+1]) << std::endl;
#endif
        scale<float>(x, sigmas[step+1]/sigmas[step]);
        accumulate<float>(x, y, -alphas[step+1]*phis[step+1]);
        break;

    case 2:
#ifdef LIBSDOD_DEBUG
        std::cout << format("MS DPM, t: {}, lambda: {}, log(a): {}, sigma: {}, alpha: {}, phi: {}, i2r: {}",
            fs{ ts[step-1], ts[step], ts[step+1] },
            fs{ lambdas[step-1], lambdas[step], lambdas[step+1] },
            fs{ log_alphas[step], log_alphas[step+1] },
            fs{ sigmas[step], sigmas[step+1] },
            alphas[step+1],
            phis[step+1],
            i2rs[step+1]) << std::endl;
#endif
        scale<float>(x, sigmas[step+1]/sigmas[step]);
        accumulate<float>(x, prev_y, alphas[step+1]*phis[step+1]*i2rs[step+1]);
        accumulate<float>(x, y, -alphas[step+1]*phis[step+1]*(1 + i2rs[step+1]));
        break;

    default:
        throw libsdod_exception(ErrorCode::INTERNAL_ERROR, "Unreachable", __func__, __FILE__, STR(__LINE__));
    }

    if (prev_y.empty())
        prev_y = y;
    else
        std::swap(y, prev_y);
}
