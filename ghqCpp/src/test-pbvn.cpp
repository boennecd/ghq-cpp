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
    expect_true(std::abs(pbvn<2>(mu, Sigma) - truth) < truth * 1e-4);
  }
   test_that("pbvn works in an extreme case") {
      /*
       Sigma <- structure(c(10.9402839103739, -5.63628007216318, -5.63628007216318,
       2.93573946546281), .Dim = c(2L, 2L))
       mu <- c(-0.224930379907288, 0.870979595977023)
       dput(mvtnorm::pmvnorm(upper = -mu, sigma = Sigma))
       */
      constexpr double mu_extreme[]{-0.224930379907288, 0.870979595977023},
                       Sigma_extreme[]{10.9402839103739, -5.63628007216318, -5.63628007216318, 2.93573946546281},
                       truth_extreme{1.0991721917919e-07};

      expect_true
         (std::abs(pbvn<0>(mu_extreme, Sigma_extreme) - truth_extreme) <
            truth_extreme * 1e-4);
      expect_true
         (std::abs(pbvn<1>(mu_extreme, Sigma_extreme) - truth_extreme) <
            truth_extreme * 1e-8);
      expect_true
         (std::abs(pbvn<2>(mu_extreme, Sigma_extreme) - truth_extreme) <
            truth_extreme * 1e-4);
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

/*
 .check_input <- \(n, a, b, Sig)
 stopifnot(length(a) == n, length(b) == n, all(dim(Sig) == c(n, n)),
 all(a < b))
 psi <- \(x, a, b, Sig){
 n <- length(x) %/% 2L
 mu <- head(x, n)
 x <- tail(x, n)
 .check_input(n, a, b, Sig)
 ubs <- lbs <- numeric(n)
 C <- chol(Sig)
 lbs[1] <- a[1] / C[1, 1]
 ubs[1] <- b[1] / C[1, 1]
 for(i in seq_len(n - 1L) + 1L){
 lbs[i] <- (a[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 ubs[i] <- (b[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 }
 lbs[is.infinite(a)] <- -Inf
 ubs[is.infinite(b)] <-  Inf
 -sum(mu * x) + sum(mu^2) / 2 + sum(log(pnorm(ubs - mu) - pnorm(lbs - mu)))
 }
 psi_safe <- \(x, a, b, Sig){
 n <- length(x) %/% 2L
 mu <- head(x, n)
 x <- tail(x, n)
 .check_input(n, a, b, Sig)
 ubs <- lbs <- numeric(n)
 C <- chol(Sig)
 lbs[1] <- a[1] / C[1, 1]
 ubs[1] <- b[1] / C[1, 1]
 for(i in seq_len(n - 1L) + 1L){
 lbs[i] <- (a[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 ubs[i] <- (b[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 }
 lbs <- lbs - mu
 ubs <- ubs - mu
 pnrm_terms <- numeric(n)
 for(i in 1:n){
 pnrm_terms[i] <-
 if(lbs[i] > 0){
 pnrm_log_lb <- pnorm(lbs[i], lower.tail = FALSE, log.p = TRUE)
 pnrm_log_ub <- pnorm(ubs[i], lower.tail = FALSE, log.p = TRUE)
 pnrm_log_lb + log1p(-exp(pnrm_log_ub - pnrm_log_lb))
 } else if(ubs[i] < 0){
 pnrm_log_lb <- pnorm(lbs[i], lower.tail = TRUE, log.p = TRUE)
 pnrm_log_ub <- pnorm(ubs[i], lower.tail = TRUE, log.p = TRUE)
 pnrm_log_ub + log1p(-exp(pnrm_log_lb - pnrm_log_ub))
 } else {
 pnrm_lb <- pnorm(lbs[i], lower.tail = TRUE, log.p = FALSE)
 pnrm_ub <- pnorm(ubs[i], lower.tail = FALSE, log.p = FALSE)
 log1p(-pnrm_lb - pnrm_ub)
 }
 }
 -sum(mu * x) + sum(mu^2) / 2 + sum(pnrm_terms)
 }
 d_psi <- \(x, a, b, Sig){
 n <- length(x) %/% 2L
 mu <- head(x, n)
 x <- tail(x, n)
 .check_input(n, a, b, Sig)
 ubs <- lbs <- numeric(n)
 C <- chol(Sig)
 lbs[1] <- a[1] / C[1, 1]
 ubs[1] <- b[1] / C[1, 1]
 for(i in seq_len(n - 1L) + 1L){
 lbs[i] <- (a[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 ubs[i] <- (b[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 }
 lbs <- lbs - mu
 ubs <- ubs - mu
 denoms_log <- numeric(n)
 for(i in 1:n){
 denoms_log[i] <-
 if(lbs[i] > 0){
 pnrm_log_lb <- pnorm(lbs[i], lower.tail = FALSE, log.p = TRUE)
 pnrm_log_ub <- pnorm(ubs[i], lower.tail = FALSE, log.p = TRUE)
 pnrm_log_lb + log1p(-exp(pnrm_log_ub - pnrm_log_lb))
 } else if(ubs[i] < 0){
 pnrm_log_lb <- pnorm(lbs[i], lower.tail = TRUE, log.p = TRUE)
 pnrm_log_ub <- pnorm(ubs[i], lower.tail = TRUE, log.p = TRUE)
 pnrm_log_ub + log1p(-exp(pnrm_log_lb - pnrm_log_ub))
 } else {
 pnrm_lb <- pnorm(lbs[i], lower.tail = TRUE, log.p = FALSE)
 pnrm_ub <- pnorm(ubs[i], lower.tail = FALSE, log.p = FALSE)
 log1p(-pnrm_lb - pnrm_ub)
 }
 }
 dnrms_log_lbs <- dnorm(lbs, log = TRUE)
 dnrms_log_ubs <- dnorm(ubs, log = TRUE)
 ratio_lbs <- exp(dnrms_log_lbs - denoms_log)
 ratio_ubs <- exp(dnrms_log_ubs - denoms_log)
 derivs <- ratio_lbs - ratio_ubs
 C <- C %*% diag(diag(C)^-1)
 c(mu - x + derivs, -mu + (C - diag(n)) %*% derivs)
 }
 dd_psi <- \(x, a, b, Sig){
 n <- length(x) %/% 2L
 mu <- head(x, n)
 x <- tail(x, n)
 .check_input(n, a, b, Sig)
 ubs <- lbs <- numeric(n)
 C <- chol(Sig)
 lbs[1] <- a[1] / C[1, 1]
 ubs[1] <- b[1] / C[1, 1]
 for(i in seq_len(n - 1L) + 1L){
 lbs[i] <- (a[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 ubs[i] <- (b[i] - sum(C[seq_len(i - 1L), i] * x[seq_len(i - 1L)])) / C[i, i]
 }
 lbs <- lbs - mu
 ubs <- ubs - mu
 denoms_log <- numeric(n)
 for(i in 1:n){
 denoms_log[i] <-
 if(lbs[i] > 0){
 pnrm_log_lb <- pnorm(lbs[i], lower.tail = FALSE, log.p = TRUE)
 pnrm_log_ub <- pnorm(ubs[i], lower.tail = FALSE, log.p = TRUE)
 pnrm_log_lb + log1p(-exp(pnrm_log_ub - pnrm_log_lb))
 } else if(ubs[i] < 0){
 pnrm_log_lb <- pnorm(lbs[i], lower.tail = TRUE, log.p = TRUE)
 pnrm_log_ub <- pnorm(ubs[i], lower.tail = TRUE, log.p = TRUE)
 pnrm_log_ub + log1p(-exp(pnrm_log_lb - pnrm_log_ub))
 } else {
 pnrm_lb <- pnorm(lbs[i], lower.tail = TRUE, log.p = FALSE)
 pnrm_ub <- pnorm(ubs[i], lower.tail = FALSE, log.p = FALSE)
 log1p(-pnrm_lb - pnrm_ub)
 }
 }
 dnrms_log_lbs <- dnorm(lbs, log = TRUE)
 dnrms_log_ubs <- dnorm(ubs, log = TRUE)
 ratio_lbs <- exp(dnrms_log_lbs - denoms_log)
 ratio_ubs <- exp(dnrms_log_ubs - denoms_log)
 derivs_gr <- ratio_lbs - ratio_ubs
 derivs <-
 ifelse(is.infinite(lbs), 0, lbs * ratio_lbs) -
 ifelse(is.infinite(ubs), 0, ubs * ratio_ubs) -
 derivs_gr^2
 C <- C %*% diag(diag(C)^-1)
 diff_mat <- C - diag(n)
 out <- matrix(0., 2L * n, 2L * n)
 m1 <- diff_mat %*% diag(derivs)
 out[  1:n ,   1:n ] <- diag(n) + diag(derivs)
 out[-(1:n),   1:n ] <- m1 - diag(n)
 out[-(1:n), -(1:n)] <- tcrossprod(m1, diff_mat)
 out[upper.tri(out)] <- t(out)[upper.tri(out)]
 out
 }
 */

context("find_tilting_param functions works as expected") {
   test_that("find_tilting_param works") {
      /*
       set.seed(111)
       n <- 2L
       Sig <- matrix(1, n, n)
       while (cov2cor(Sig)[lower.tri(Sig)] |> abs() |> max() > .999)
       Sig <- rWishart(1, n, diag(n)) |> drop()
       a <- rep(-Inf, 2)
       b <- runif(n, -2, 2)
       type <- rep(1, n)
       start <- local({
       C <- chol(Sig)
       start_org_scale <- (cbind(a, b) |> rowMeans()) / diag(C)
       start_org_scale[type == 1] <- b[type == 1] / diag(C)[type == 1] - 1
       start_org_scale[type == 2] <- a[type == 2] / diag(C)[type == 2] + 1
       start_org_scale <- start_org_scale * diag(C)
       solve(t(C), start_org_scale)
       })
       par <- local({
       mu <- numeric(n)
       C <- chol(Sig)
       for(i in seq_len(n - 1L) + 1L)
       mu[i] <- -C[seq_len(i - 1L), i] %*% start[seq_len(i - 1L)] / C[i, i]
       c(mu, start)
       })
# do we start of in an interior point?
       ptr <- crossprod(chol(Sig), tail(par, n))
       all(ptr > a)
       all(ptr < b)
       psi(par, a, b, Sig)
       all.equal(psi(par, a, b, Sig), psi_safe(par, a, b, Sig))
       psi <- psi_safe
       stopifnot(all.equal(d_psi(par, a, b, Sig),
       numDeriv::grad(psi, par, a = a, b = b, Sig = Sig)))
       stopifnot(all.equal(dd_psi(par, a, b, Sig),
       numDeriv::jacobian(d_psi, par, a = a, b = b, Sig = Sig)))
# finds a root
       root_finder <- \(x, a, b, Sig, abstol = 1e-2){
       f <- \(x) d_psi(x, a, b, Sig)^2 |> sum()
       d_f <- \(x){
       f_vals <- d_psi(x, a, b, Sig)
       grs <- dd_psi(x, a, b, Sig)
       2 * rowSums(grs %*% diag(f_vals))
       }
# sanity check as this is still experimental
       num_gr <- try(numDeriv::grad(f, x), silent = TRUE)
       if(!inherits(num_gr, "try-error")){
       is_equal <- all.equal(d_f(x), num_gr, tolerance = 1e-5)
       if(!isTRUE(is_equal))
       warning(paste0(capture.output(is_equal), collapse = "\n"))
       }
# find the root
       optim(x, fn = f, gr = d_f, method = "BFGS",
       control = list(reltol = 1e-8, abstol = abstol))
       }
       res <- root_finder(par, a, b, Sig, 0)
       rbind(Estimate = res$par, Start = par)
       d_psi(res$par, a, b, Sig) |> abs() |> sum() # ~ zero
       res$counts
# do we have an interior solution
       ptr <- crossprod(chol(Sig), tail(res$par, n))
       all(ptr > a)
       all(ptr < b)
       head(res$par, n) |> dput()
       C <- chol(Sig)
       dput(a / diag(C))
       dput(b / diag(C))
       dput((C %*% diag(diag(C)^-1))[upper.tri(C, TRUE)])
       */
      constexpr double upper_limits[]{-0.23266007717202, -2.57026138039795},
                       choleksy_off_diag{-3.64100479014587},
                       truth{14.2325094470914};

      auto res = find_tilting_param(upper_limits, choleksy_off_diag, 1e-8);

      expect_true(res.success);
      expect_true(res.is_interior);
      expect_true(std::abs(res.tilting - truth) < std::abs(truth) * 1e-6);
   }

   test_that("find_tilting_param works in an extreme case") {
      /*
       set.seed(111)
       n <- 2L
       Sig <- matrix(c(1, 0, -9.52473675170593, 1), n, n)
       Sig <- t(Sig) %*% Sig
       a <- rep(-Inf, 2)
       b <- c(0.0680039000743836, -4.86836008664038)
       type <- rep(1, n)
       start <- local({
       C <- chol(Sig)
       start_org_scale <- (cbind(a, b) |> rowMeans()) / diag(C)
       start_org_scale[type == 1] <- b[type == 1] / diag(C)[type == 1] - 1
       start_org_scale[type == 2] <- a[type == 2] / diag(C)[type == 2] + 1
       start_org_scale <- start_org_scale * diag(C)
       solve(t(C), start_org_scale)
       })
       par <- local({
       mu <- numeric(n)
       C <- chol(Sig)
       for(i in seq_len(n - 1L) + 1L)
       mu[i] <- -C[seq_len(i - 1L), i] %*% start[seq_len(i - 1L)] / C[i, i]
       c(mu, start)
       })
# do we start of in an interior point?
       ptr <- crossprod(chol(Sig), tail(par, n))
       all(ptr > a)
       all(ptr < b)
       psi(par, a, b, Sig)
       all.equal(psi(par, a, b, Sig), psi_safe(par, a, b, Sig))
       psi <- psi_safe
       stopifnot(all.equal(d_psi(par, a, b, Sig),
       numDeriv::grad(psi, par, a = a, b = b, Sig = Sig)))
       stopifnot(all.equal(dd_psi(par, a, b, Sig),
       numDeriv::jacobian(d_psi, par, a = a, b = b, Sig = Sig)))
# finds a root
       root_finder <- \(x, a, b, Sig, abstol = 1e-2){
       f <- \(x) d_psi(x, a, b, Sig)^2 |> sum()
       d_f <- \(x){
       f_vals <- d_psi(x, a, b, Sig)
       grs <- dd_psi(x, a, b, Sig)
       2 * rowSums(grs %*% diag(f_vals))
       }
# sanity check as this is still experimental
       num_gr <- try(numDeriv::grad(f, x), silent = TRUE)
       if(!inherits(num_gr, "try-error")){
       is_equal <- all.equal(d_f(x), num_gr, tolerance = 1e-5)
       if(!isTRUE(is_equal))
       warning(paste0(capture.output(is_equal), collapse = "\n"))
       }
# find the root
       optim(x, fn = f, gr = d_f, method = "BFGS",
       control = list(reltol = 1e-10, abstol = abstol, maxit = 1000))
       }
       res <- root_finder(par, a, b, Sig, 0)
       rbind(Estimate = res$par, Start = par)
       d_psi(res$par, a, b, Sig) |> abs() |> sum() # ~ zero
       res$counts
# do we have an interior solution
       ptr <- crossprod(chol(Sig), tail(res$par, n))
       all(ptr > a)
       all(ptr < b)
       head(res$par, n) |> dput()
       C <- chol(Sig)
       dput(a / diag(C))
       dput(b / diag(C))
       dput((C %*% diag(diag(C)^-1))[upper.tri(C, TRUE)])
       */

      constexpr double upper_limits[]{0.0680039000743836, -4.86836008664038},
                    choleksy_off_diag{-9.52473675170593},
                                truth{44.2229622170309};

      auto res = find_tilting_param(upper_limits, choleksy_off_diag, 1e-10);

      expect_true(res.success);
      expect_true(res.is_interior);
      expect_true(std::abs(res.tilting - truth) < std::abs(truth) * 1e-6);
   }
}
