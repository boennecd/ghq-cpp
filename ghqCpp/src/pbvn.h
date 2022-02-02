#ifndef PBVN_H
#define PBVN_H

#include <RcppArmadillo.h>
#include <array>
#include <Rmath.h> // Rf_dnorm4, Rf_pnorm5 etc.
#include <algorithm>
#include <cmath>

namespace ghqCpp {

/**
 * computes the integral
 *
 *   int_(-inf)^0int_(-inf)^0phi(x; mu, Sigma) dx =
 *     int_(-inf)^(-mu_1)int_(-inf)^(-mu_2)phi(x; 0, Sigma) dx
 *
 * method = 1 yields the method by Drezner further extended by Genz in
 *
 *   https://doi.org/10.1023/B:STCO.0000035304.20635.31
 *
 * method = 0 yields a Gauss–Legendre quadrature based solution. This is less
 * precise and slower.
 */
template<int method = 1>
double pbvn(double const *mu, double const *Sigma){
  static_assert(method == 1 || method == 0, "method is not implemented");

  if constexpr (method == 0){
    // setup before applying the quadrature rule
    // the, will be, scaled Cholesky decomposition of the covariance matrix
    std::array<double, 3> Sig_chol;

    double const sig1{std::sqrt(Sigma[0])},
                 sig2{std::sqrt(Sigma[3])};
    bool const permuted{-mu[1] / sig2  < -mu[0] / sig1 };
    if(permuted){
      Sig_chol[0] = sig2;
      Sig_chol[1] = Sigma[2] / sig2;
      Sig_chol[2] = std::sqrt(Sigma[0] - Sig_chol[1] * Sig_chol[1]);

    } else {
      Sig_chol[0] = sig1;
      Sig_chol[1] = Sigma[2] / sig1;
      Sig_chol[2] = std::sqrt(Sigma[3] - Sig_chol[1] * Sig_chol[1]);

    }
    if(!std::isfinite(Sig_chol[0]) || !std::isfinite(Sig_chol[2]))
      throw std::invalid_argument("Choleksy decomposition failed");

    // the scaled upper limits to add
    std::array<double, 2> const ubs
      { (permuted ? -mu[1] : -mu[0]) / Sig_chol[0],
        (permuted ? -mu[0] : -mu[1]) / Sig_chol[2] };
    Sig_chol[1] /= Sig_chol[2];

    /* Gauss–Legendre quadrature nodes scale to the interval [0, 1]. I.e.
       gq <- SimSurvNMarker::get_gl_rule(50)
       ord <- order(gq$weight)
       dput((gq$node[ord] + 1) / 2)
       dput(gq$weight[ord] / 2)
     */
    constexpr size_t n_nodes{50};
    constexpr double nodes[]{0.999433202210036, 0.00056679778996449, 0.997015984716045, 0.00298401528395464, 0.992677042024003, 0.00732295797599708, 0.986432192553346, 0.013567807446654, 0.978305477621404, 0.021694522378596, 0.968328309472439, 0.0316716905275611, 0.956539278327896, 0.043460721672104, 0.942983989761807, 0.0570160102381935, 0.927714884714973, 0.072285115285027, 0.910791035429668, 0.0892089645703321, 0.8922779164502, 0.1077220835498, 0.872247151113034, 0.127752848886966, 0.850776234353411, 0.149223765646589, 0.82794823284272, 0.17205176715728, 0.803851463592475, 0.196148536407525, 0.778579152257325, 0.221420847742675, 0.752229072453732, 0.247770927546268, 0.724903167487019, 0.275096832512981, 0.696707155948783, 0.303292844051217, 0.667750122709719, 0.332249877290281, 0.638144096889766, 0.361855903110234, 0.608003618438021, 0.391996381561979, 0.577445294999073, 0.422554705000927, 0.546587350780043, 0.453412649219957, 0.515549169163595, 0.484450830836406},
                   weights[]{0.00145431127657757, 0.00145431127657757, 0.0033798995978727, 0.0033798995978727, 0.00529527419182548, 0.00529527419182548, 0.00719041138074279, 0.00719041138074279, 0.0090577803567447, 0.0090577803567447, 0.0108901215850624, 0.0108901215850624, 0.0126803367850062, 0.0126803367850062, 0.0144214967902676, 0.0144214967902676, 0.016106864111789, 0.016106864111789, 0.0177299178075731, 0.0177299178075731, 0.0192843783062938, 0.0192843783062938, 0.0207642315450738, 0.0207642315450738, 0.0221637521694016, 0.0221637521694016, 0.0234775256519742, 0.0234775256519742, 0.0247004692247332, 0.0247004692247332, 0.0258278515347906, 0.0258278515347906, 0.0268553109444981, 0.0268553109444981, 0.0277788724031063, 0.0277788724031063, 0.0285949628238642, 0.0285949628238642, 0.0293004249066112, 0.0293004249066112, 0.0298925293521327, 0.0298925293521327, 0.0303689854208851, 0.0303689854208851, 0.0307279497951583, 0.0307279497951583, 0.0309680337103416, 0.0309680337103416, 0.0310883083276736, 0.0310883083276736};

    // do the computation
    double out{};
    double const p_outer{Rf_pnorm5(ubs[0], 0, 1, 1, 0)};
    for(size_t i = 0; i < n_nodes; ++i){
      double const z_outer{Rf_qnorm5(nodes[i] * p_outer, 0, 1, 1, 0)},
                   p_inner{Rf_pnorm5(ubs[1] - Sig_chol[1] * z_outer, 0, 1, 1, 0)};
      out += p_inner * weights[i];
    }

    return p_outer * out;
  }

  double const h{mu[0] / std::sqrt(Sigma[0])},
               k{mu[1] / std::sqrt(Sigma[3])};
  double rho{Sigma[1] / std::sqrt(Sigma[0] * Sigma[3])};

  auto pnrm = [](double const x){
    return Rf_pnorm5(x, 0, 1, 1, 0);
  };

  /* Gauss–Legendre quadrature nodes scale to the interval [0, 1]. I.e.
     gq <- SimSurvNMarker::get_gl_rule(12)
     ord <- order(gq$weight)
     dput((gq$node[ord] + 1) / 2)
     dput(gq$weight[ord] / 2)
   */

  constexpr double nodes6[]{0.966234757101576, 0.033765242898424, 0.830604693233132, 0.169395306766868, 0.619309593041598, 0.380690406958402},
                 weights6[]{0.0856622461895852, 0.0856622461895852, 0.180380786524069, 0.180380786524069, 0.233956967286346, 0.233956967286346},
                  nodes12[]{0.99078031712336, 0.00921968287664043, 0.952058628185237, 0.0479413718147626, 0.884951337097152, 0.115048662902848, 0.793658977143309, 0.206341022856691, 0.68391574949909, 0.31608425050091, 0.562616704255734, 0.437383295744266},
                weights12[]{0.0235876681932559, 0.0235876681932559, 0.0534696629976592, 0.0534696629976592, 0.0800391642716731, 0.0800391642716731, 0.101583713361533, 0.101583713361533, 0.116746268269177, 0.116746268269177, 0.124573522906701, 0.124573522906701},
                  nodes20[]{0.996564299592547, 0.00343570040745256, 0.981985963638957, 0.0180140363610431, 0.956117214125663, 0.0438827858743371, 0.919558485911109, 0.0804415140888906, 0.873165953230075, 0.126834046769925, 0.818026840363258, 0.181973159636742, 0.755433500975414, 0.244566499024587, 0.68685304435771, 0.31314695564229, 0.613892925570823, 0.386107074429178, 0.538263260566749, 0.461736739433251},
                weights20[]{0.00880700356957606, 0.00880700356957606, 0.0203007149001935, 0.0203007149001935, 0.0313360241670545, 0.0313360241670545, 0.0416383707883524, 0.0416383707883524, 0.0509650599086202, 0.0509650599086202, 0.0590972659807592, 0.0590972659807592, 0.0658443192245883, 0.0658443192245883, 0.071048054659191, 0.071048054659191, 0.0745864932363019, 0.0745864932363019, 0.0763766935653629, 0.0763766935653629};

  auto wo_border_correction = [&](double const *nodes, double const *weights,
                                  size_t const n_nodes){
    double const offset{h * h + k * k},
                  slope{2 * h * k},
                     ub{std::asin(rho)};

    double out{};
    for(size_t i = 0; i < n_nodes; ++i){
      double const n{ub * nodes[i]},
               sin_n{std::sin(n)};
      out += weights[i] * std::exp
        (-(offset - slope * sin_n) / (2 * (1 - sin_n * sin_n)));
    }
    out *= ub / (2 * M_PI);

    return out + pnrm(-h) * pnrm(-k);
  };

  if(std::abs(rho) <= .3)
    return wo_border_correction(nodes6, weights6, 6);
  else if(std::abs(rho) <= .75)
    return wo_border_correction(nodes12, weights12, 12);
  else if(std::abs(rho) <= .95)
    return wo_border_correction(nodes20, weights20, 20);

  // handle the large absolute correlation

  // computes the indefinite integral
  //   int exp(-b^2/2x^2)(1 + c * x^2 * (1 + d * x^2))
  // TODO: can likely be done a lot smarter
  auto taylor_term = [&](double const x, double const b, double const c,
                         double const d){
    double const x2{x * x},
                 b2{b * b},
                 b4{b2 * b2};
    double out{2 * x * std::exp(-b2 / (2 * x2))};
    out *= (b4 * c * d - b2 * c * (d * x2 + 5) +
      c * x2 * (3 * d * x2 + 5) + 15);

    constexpr double sqrt2pi{2.506628274631};
    out += sqrt2pi * b * (b4 * c * d - 5 * b2 * c + 15) *
      (2 * pnrm(b / x) - 1);

    return out / 30;
  };

  double const s{rho > 0 ? 1. : -1.},
              ub{std::sqrt(1 - rho * rho)},
             shk{s * h * k};

  double const numerator{-(h - s * k) * (h - s * k) / 2},
           exp_m_shk_d_2{std::exp(-shk / 2)};

  double out{};
  for(size_t i = 0; i < 20; ++i){
    double const x{nodes20[i] * ub},
                x2{x * x};
    double tay{1 + (12 - shk) * x2 / 16};
    tay *= (4 - shk) * x2 / 8;
    tay += 1;
    tay *= exp_m_shk_d_2;

    double const sqrt_1_m_x2{std::sqrt(1 - x2)},
                 fn{std::exp(-shk/(1 + sqrt_1_m_x2)) / sqrt_1_m_x2};

    out += weights20[i] * std::exp(numerator / x2) * (fn - tay);
  }
  out *= ub;

  double const b{std::abs(h - s * k)},
               c{(4 - shk) / 8},
               d{(12 - shk) / 16};

  out +=
    exp_m_shk_d_2 * (taylor_term(ub, b, c, d) - taylor_term(0, b, c, d));
  out *= (-s / (2 * M_PI));
  out += s > 0
    ? pnrm(-std::max(h, k))
    : std::max(0., pnrm(-h) - pnrm(k));
  return out;
}

/**
 * computes the derivative of the mean and covariance matrix in of pbvn. For the
 * mean, this is given by
 *
 *   Sigma^(-1)int_(-inf)^0int_(-inf)^0(x - mu)phi(x; mu, Sigma) dx =
 *     Sigma^(-1).int_(-inf)^(-mu_1)int_(-inf)^(-mu_2)x phi(x; 0, Sigma) dx
 *
 * For Sigma, we need to compute
 *
 *   2^(-1)Sigma^(-1)[int_(-inf)^0int_(-inf)^0
 *     ((x - mu).(x - mu)^T - Sigma)phi(x; mu, Sigma) dx]Sigma^(-1) =
 *   2^(-1)Sigma^(-1)[int_(-inf)^(-mu_1)int_(-inf)^(-mu_2)
 *     (x.x^T - Sigma)phi(x; mu, Sigma) dx]Sigma^(-1)
 *
 * the derivatives w.r.t. Sigma are stored as a 2 x 2 matrix ignoring the
 * symmetry. Thus, a 6D array needs to be passed for the gradient.
 */
template<int method = 1, bool comp_d_Sig = true>
double pbvn_grad(double const *mu, double const *Sigma, double *grad){
  static_assert(method == 1 || method == 0, "method is not implemented");
  std::array<double, 3> Sig_chol;

  double const sig1{std::sqrt(Sigma[0])},
               sig2{std::sqrt(Sigma[3])};
  bool const permuted{-mu[1] / sig2  < -mu[0] / sig1 };
  if(permuted){
    Sig_chol[0] = sig2;
    Sig_chol[1] = Sigma[2] / sig2;
    Sig_chol[2] = std::sqrt(Sigma[0] - Sig_chol[1] * Sig_chol[1]);

  } else {
    Sig_chol[0] = sig1;
    Sig_chol[1] = Sigma[2] / sig1;
    Sig_chol[2] = std::sqrt(Sigma[3] - Sig_chol[1] * Sig_chol[1]);

  }
  if(!std::isfinite(Sig_chol[0]) || !std::isfinite(Sig_chol[2]))
    throw std::invalid_argument("Choleksy decomposition failed");

  // the scaled upper limits to add
  std::array<double, 2> const ubs
  { (permuted ? -mu[1] : -mu[0]) / Sig_chol[0],
    (permuted ? -mu[0] : -mu[1]) / Sig_chol[2] };
  double const Sig_12_scaled{Sig_chol[1] / Sig_chol[2]};

  constexpr size_t n_nodes{50};
  constexpr double nodes[]{0.999433202210036, 0.00056679778996449, 0.997015984716045, 0.00298401528395464, 0.992677042024003, 0.00732295797599708, 0.986432192553346, 0.013567807446654, 0.978305477621404, 0.021694522378596, 0.968328309472439, 0.0316716905275611, 0.956539278327896, 0.043460721672104, 0.942983989761807, 0.0570160102381935, 0.927714884714973, 0.072285115285027, 0.910791035429668, 0.0892089645703321, 0.8922779164502, 0.1077220835498, 0.872247151113034, 0.127752848886966, 0.850776234353411, 0.149223765646589, 0.82794823284272, 0.17205176715728, 0.803851463592475, 0.196148536407525, 0.778579152257325, 0.221420847742675, 0.752229072453732, 0.247770927546268, 0.724903167487019, 0.275096832512981, 0.696707155948783, 0.303292844051217, 0.667750122709719, 0.332249877290281, 0.638144096889766, 0.361855903110234, 0.608003618438021, 0.391996381561979, 0.577445294999073, 0.422554705000927, 0.546587350780043, 0.453412649219957, 0.515549169163595, 0.484450830836406},
                 weights[]{0.00145431127657757, 0.00145431127657757, 0.0033798995978727, 0.0033798995978727, 0.00529527419182548, 0.00529527419182548, 0.00719041138074279, 0.00719041138074279, 0.0090577803567447, 0.0090577803567447, 0.0108901215850624, 0.0108901215850624, 0.0126803367850062, 0.0126803367850062, 0.0144214967902676, 0.0144214967902676, 0.016106864111789, 0.016106864111789, 0.0177299178075731, 0.0177299178075731, 0.0192843783062938, 0.0192843783062938, 0.0207642315450738, 0.0207642315450738, 0.0221637521694016, 0.0221637521694016, 0.0234775256519742, 0.0234775256519742, 0.0247004692247332, 0.0247004692247332, 0.0258278515347906, 0.0258278515347906, 0.0268553109444981, 0.0268553109444981, 0.0277788724031063, 0.0277788724031063, 0.0285949628238642, 0.0285949628238642, 0.0293004249066112, 0.0293004249066112, 0.0298925293521327, 0.0298925293521327, 0.0303689854208851, 0.0303689854208851, 0.0307279497951583, 0.0307279497951583, 0.0309680337103416, 0.0309680337103416, 0.0310883083276736, 0.0310883083276736};

  // do the computation
  double out{};
  std::fill(grad, comp_d_Sig ? grad + 6 : grad + 2, 0);
  double * const d_mu{grad},
         * const d_Sig{comp_d_Sig ? grad + 2 : nullptr};
  double const p_outer{Rf_pnorm5(ubs[0], 0, 1, 1, 0)};

  for(size_t i = 0; i < n_nodes; ++i){
    double const z_outer{Rf_qnorm5(nodes[i] * p_outer, 0, 1, 1, 0)},
             u_lim_inner{ubs[1] - Sig_12_scaled * z_outer},
                 p_inner{Rf_pnorm5(u_lim_inner, 0, 1, 1, 0)};
    if(method == 0)
      out += p_inner * weights[i];

    double const g1_fac{z_outer * p_inner},
      dnorm_u_lim_inner{Rf_dnorm4(u_lim_inner, 0, 1, 0)},
      trunc_mean_scaled{-dnorm_u_lim_inner};
    grad[0] += weights[i] * g1_fac;
    grad[1] += weights[i] * trunc_mean_scaled;

    if constexpr (comp_d_Sig){
      d_Sig[0] += weights[i] * g1_fac * z_outer;
      double const off_diag{z_outer * trunc_mean_scaled};
      d_Sig[1] += weights[i] * off_diag;
      double const trunc_sq_moment_scaled
        {p_inner - dnorm_u_lim_inner * u_lim_inner};
      d_Sig[3] += weights[i] * trunc_sq_moment_scaled;
    }
  }

  if(method == 1)
    out = pbvn<method>(mu, Sigma);
  else
    out *= p_outer;

  // handle the derivatives w.r.t. mu
  std::for_each(d_mu, d_mu + 2, [&](double &x){ x *= p_outer; });

  // performs backward substitution
  auto back_sub = [&](double *x){
    x[1] /= Sig_chol[2];
    x[0] = (x[0] - Sig_chol[1] * x[1]) / Sig_chol[0];
  };

  back_sub(d_mu);

  // possibly handle the derivatives w.r.t Sigma
  if constexpr (comp_d_Sig){
    d_Sig[2] = d_Sig[1]; // symmetry
    std::for_each(d_Sig, d_Sig + 4,
                  [&](double &x){ x *= p_outer / 2; });

    // subtract the identity matrix in the diagonal
    d_Sig[0] -= out / 2;
    d_Sig[3] -= out / 2;

    back_sub(d_Sig);
    back_sub(d_Sig + 2);
    std::swap(d_Sig[1], d_Sig[2]); // transpose
    back_sub(d_Sig);
    back_sub(d_Sig + 2);
  }

  if(permuted){
    std::swap(grad[0], grad[1]); // d_mu
    if constexpr (comp_d_Sig)
      std::swap(grad[2], grad[5]); // d_Sigma
  }

  return out;
}

/// computes the Hessian w.r.t. mu. Thus, a 4D array has to be passed
template<int method = 1>
void pbvn_hess(double const *mu, double const *Sigma, double *hess){
  double gr[6];
  pbvn_grad<method, true>(mu, Sigma, gr);

  arma::mat Sig(const_cast<double *>(Sigma), 2, 2, false);
  for(unsigned j = 0; j < 2; ++j)
    for(unsigned i = 0; i < 2; ++i)
      hess[i + j * 2] = 2 * gr[i + j * 2 + 2];
}

} // namespace ghqCpp

#endif
