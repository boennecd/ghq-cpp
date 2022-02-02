#include <testthat.h>
#include "pbvn.h"

using namespace ghqCpp;

namespace {
  /*
   mu <- c(-1, .5)
   Sigma <- matrix(c(2, -.5, -.5, 1), 2)
   dput(mvtnorm::pmvnorm(upper = -mu, sigma = Sigma))

   num_grad <- numDeriv::grad(
   \(par) {
   mu <- head(par, 2)
   S <- matrix(nrow = 2, ncol = 2)
   S[upper.tri(S, TRUE)] <- tail(par, 3)
   S[lower.tri(S)] <- t(S)[lower.tri(S)]

   mvtnorm::pmvnorm(upper = -mu, sigma = S)
   }, c(mu, Sigma[upper.tri(Sigma, TRUE)]), method.args = list(r = 6))

   d_mu <- num_grad[1:2]
   d_Sig <- matrix(0, 2, 2)
   d_Sig[upper.tri(d_Sig, TRUE)] <- tail(num_grad, -2L)
   d_Sig[upper.tri(d_Sig)] <- d_Sig[upper.tri(d_Sig)] / 2
   d_Sig[lower.tri(d_Sig)] <- t(d_Sig)[lower.tri(d_Sig)]
   dput(c(d_mu, d_Sig))

   num_hess <- numDeriv::hessian(\(m) mvtnorm::pmvnorm(upper = -m, sigma = Sigma), mu)
   dput(num_hess)
   */
constexpr double mu[]{-1, .5},
              Sigma[]{2, -.5, -.5, 1},
              truth{0.192983336746525},
          true_grad[]{-0.0866993739345381, -0.251594615825539, -0.010373580459756, 0.0452050520837945, 0.0452050520837945, 0.0855011800114486},
          true_hess[]{-0.0207471609259662, 0.0904101041649396, 0.0904101041649396, 0.1710023599969};
}

context("pbvn functions works as expected") {
  test_that("pbvn works") {
    expect_true(std::abs(pbvn<0>(mu, Sigma) - truth) < truth * 1e-4);
    expect_true(std::abs(pbvn<1>(mu, Sigma) - truth) < truth * 1e-8);
  }
  test_that("pbvn_grad works") {
    {
      double gr[6];
      expect_true(std::abs(pbvn_grad<0>(mu, Sigma, gr) - truth) < truth * 1e-4);
      for(unsigned i = 0; i < 6; ++i)
        expect_true
          (std::abs(gr[i] - true_grad[i]) < std::abs(true_grad[i]) * 1e-4);
    }
    {
       double gr[6];
       expect_true
         (std::abs(pbvn_grad<1>(mu, Sigma, gr) - truth) < truth * 1e-8);
       for(unsigned i = 0; i < 6; ++i)
         expect_true
           (std::abs(gr[i] - true_grad[i]) < std::abs(true_grad[i]) * 1e-4);
    }
    double gr[2];
    expect_true
      (std::abs(pbvn_grad<1, false>(mu, Sigma, gr) - truth) < truth * 1e-8);
    for(unsigned i = 0; i < 2; ++i)
       expect_true
         (std::abs(gr[i] - true_grad[i]) < std::abs(true_grad[i]) * 1e-4);
  }

   test_that("pbvn_hess works") {
      double hess[4];
      {
         pbvn_hess<0>(mu, Sigma, hess);
         for(unsigned i = 0; i < 4; ++i)
            expect_true
            (std::abs(hess[i] - true_hess[i]) < std::abs(true_hess[i]) * 1e-4);
      }
      pbvn_hess<1>(mu, Sigma, hess);
      for(unsigned i = 0; i < 4; ++i)
         expect_true
            (std::abs(hess[i] - true_hess[i]) < std::abs(true_hess[i]) * 1e-4);
   }
}
