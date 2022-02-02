#ifndef INTEGRAND_COND_PBVN_H
#define INTEGRAND_COND_PBVN_H

#include "ghq.h"

namespace ghqCpp {

/**
 * Computes the expectation of
 *
 *   E(Phi^{(2)}(0, eta + V.U, Psi))
 *
 * where Phi^{(2)} is the CDF of a bivariate normal distribution over the
 * rectangle (-Inf, 0)^2 with the passed conditional mean and variance.
 *
 * The derivatives are computed w.r.t. eta, V and Psi.
 */
template<bool comp_grad = false>
class cond_pbvn final : public ghq_problem {
  arma::vec const &eta;
  arma::mat const &Psi;
  arma::mat const Sigma_chol, V_Sigma_chol_t;

  size_t const v_n_vars = Sigma_chol.n_cols,
               v_n_out  = comp_grad ? 7 + V_Sigma_chol_t.n_elem : 1;

public:
  cond_pbvn(arma::vec const &eta, arma::mat const &Psi, arma::mat const &V,
            arma::mat const &Sigma);

  size_t n_vars() const { return v_n_vars; }
  size_t n_out() const { return v_n_out; }

  void eval
    (double const *points, size_t const n_points, double * __restrict__ outs,
     simple_mem_stack<double> &mem) const;

  double log_integrand
    (double const *point, simple_mem_stack<double> &mem) const;

  double log_integrand_grad
    (double const *point, double * __restrict__ grad,
     simple_mem_stack<double> &mem) const;

  void log_integrand_hess
    (double const *point, double *hess,
     simple_mem_stack<double> &mem) const;
};

} // namespace ghqCpp

#endif
