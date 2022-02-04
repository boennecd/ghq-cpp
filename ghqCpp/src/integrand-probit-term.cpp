#include "integrand-probit-term.h"
#include "ghq-lp-utils.h"
#include <algorithm>
#include <Rmath.h> // Rf_dnorm4 and Rf_pnorm5

namespace ghqCpp {

template<bool comp_grad>
mixed_probit_term<comp_grad>::mixed_probit_term
  (double const s, double const eta, arma::vec const &z):
  s{s}, eta{eta}, z{z} { }

template<bool comp_grad>
void mixed_probit_term<comp_grad>::eval
  (double const *points, size_t const n_points, double * __restrict__ outs,
   simple_mem_stack<double> &mem) const {
  // compute the linear predictor
  double * const __restrict__ lps{mem.get(n_points)};
  std::fill(lps, lps + n_points, eta);
  for(size_t j = 0; j < n_vars(); ++j)
    for(size_t i = 0; i < n_points; ++i)
      lps[i] += points[i + j * n_points] * z[j];

  std::for_each(lps, lps + n_points, [&](double &x) { x /= s; });

  // set the output
  for(size_t i = 0; i < n_points; ++i, ++outs)
    *outs = Rf_pnorm5(lps[i], 0, 1, 1, 0);

  if constexpr (comp_grad){
    double * const __restrict__ d_etas{outs};

    // the derivative w.r.t. eta and s
    for(size_t i = 0; i < n_points; ++i){
      d_etas[i] = Rf_dnorm4(lps[i], 0, 1, 0) / s;
      outs[i + n_points] = -d_etas[i] * lps[i];
    }
    outs += 2 * n_points;

    // the derivatives w.r.t. z
    double const * point_ij{points};
    for(size_t j = 0; j < n_vars(); ++j)
      for(size_t i = 0; i < n_points; ++i, ++outs, ++point_ij)
        *outs = d_etas[i] * *point_ij;
  }
}

template<bool comp_grad>
double mixed_probit_term<comp_grad>::log_integrand
  (double const *point, simple_mem_stack<double> &mem) const {
  double lp{eta};
  for(size_t i = 0; i < n_vars(); ++i)
    lp += point[i] * z[i];
  lp /= s;
  return Rf_pnorm5(lp, 0, 1, 1, 1);
}

template<bool comp_grad>
double mixed_probit_term<comp_grad>::log_integrand_grad
  (double const *point, double * __restrict__ grad,
   simple_mem_stack<double> &mem) const {
  double lp{eta};
  for(size_t i = 0; i < n_vars(); ++i)
    lp += point[i] * z[i];
  lp /= s;
  double const log_pnrm{Rf_pnorm5(lp, 0, 1, 1, 1)},
               log_dnrm{Rf_dnorm4(lp, 0, 1, 1)},
               d_lp{std::exp(log_dnrm - log_pnrm)};

  for(size_t i = 0; i < n_vars(); ++i)
    grad[i] = z[i] * d_lp / s;

  return log_pnrm;
}

template<bool comp_grad>
void mixed_probit_term<comp_grad>::log_integrand_hess
  (double const *point, double *hess,
   simple_mem_stack<double> &mem) const {
  double lp{eta};
  for(size_t i = 0; i < n_vars(); ++i)
    lp += point[i] * z[i];
  lp /= s;

  double const log_pnrm{Rf_pnorm5(lp, 0, 1, 1, 1)},
               log_dnrm{Rf_dnorm4(lp, 0, 1, 1)},
                  ratio{std::exp(log_dnrm - log_pnrm)},
                d_lp_sq{-(lp * ratio + ratio * ratio)};

  for(size_t j = 0; j < n_vars(); ++j)
    for(size_t i = 0; i < n_vars(); ++i)
      hess[i + j * n_vars()] = z[i] * z[j] * d_lp_sq;

  std::for_each(hess, hess + n_vars() * n_vars(),
                [&](double &x){ x /= s * s; });
}

template class mixed_probit_term<false>;
template class mixed_probit_term<true>;

} // namespace ghqCpp
