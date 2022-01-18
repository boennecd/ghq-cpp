#ifndef INTEGRAND_EXPECTED_SURVIVAL_H
#define INTEGRAND_EXPECTED_SURVIVAL_H

#include "ghq.h"

/**
 * computes the expected survival. That is
 *
 *   E(exp(-sum w_i[i] * exp(eta[i] + M.U)))
 *
 * given n weights and offsets w and eta and a matrix M x R. U is assumed to be
 * a R dimensional random variable which is ~ N(0, Sigma).
 *
 * The derivatives are computed w.r.t. the vector eta and the matrix M
 */
template<bool comp_grad = false>
class expected_survival_term final : public ghq_problem {
  arma::vec const &eta, &weights;
  arma::mat const Sigma_chol, M_Sigma_chol_t;

  size_t const v_n_vars = M_Sigma_chol_t.n_cols,
               v_n_out
    {comp_grad ? 1  + eta.n_elem + M_Sigma_chol_t.n_rows * M_Sigma_chol_t.n_cols
               : 1};

public:
  expected_survival_term
  (arma::vec const &eta, arma::vec const &weights, arma::mat const &M,
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

#endif
