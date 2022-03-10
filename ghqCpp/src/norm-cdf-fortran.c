#include <Rmath.h> // Rf_dnorm4, Rf_pnorm5 etc.
#include <R_ext/RS.h> // for F77_NAME and F77_CALL

// TODO: replace with a version that is thread-safe
double F77_SUB(mvphi)(double *x){ return Rf_pnorm5(*x, 0.0, 1.0, 1, 0); }
