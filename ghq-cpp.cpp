// needed as we use Rcpp::sourceCpp
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(psqn)]]
// [[Rcpp::depends(RcppArmadillo)]]

// simple_mem_stack.h

#include <vector>
#include <iterator>
#include <stack>
#include <list>
#include <cstddef>

#ifdef DEBUG_SIMPLE_MEM_STACK
#include <memory>
#include <algorithm>
#endif

/**
 * stack like object used to avoid repeated allocation. In principle,
 * everything goes well if set_mark_raii() is called after all requests in a 
 * function call and before any further function calls and subsequently the 
 * returned object is destroyed at the end of the scope (as typical with RAII). 
 * 
 * If you define DEBUG_SIMPLE_MEM_STACK then just the right memory for each 
 * request is allocated. Thus, you will easily find errors due to too little 
 * memory being requested. 
 */
template<class T> 
class simple_mem_stack {
#ifdef DEBUG_SIMPLE_MEM_STACK
  class block_container {
    size_t const n_ele;
    std::unique_ptr<double[]> mem{new double[n_ele]};
    
  public:
    using iterator = T*;
    
    block_container(size_t const n_ele): n_ele{n_ele} { }
    block_container(const block_container &o):
      n_ele{o.n_ele},
      mem{new double[n_ele]} {
        std::copy(o.begin(), o.end(), begin());
      }
    
    iterator begin() { return mem.get(); }
    iterator end() { return mem.get() + n_ele; }
    size_t size() const { return n_ele; }
  };
  
#else // #ifdef DEBUG_SIMPLE_MEM_STACK
  // TODO: maybe replace with a simpler container?
  using block_container = std::vector<T>; 
  
#endif
  
  using outer_container = std::list<block_container>;
  /// holds the allocated memory in blocks
  outer_container memory;
  
  /**
   * a simple iterator that implements the members we need. It also stores an 
   * iterator to the block which is needed when go back to a mark.
   */
  class iterator {
    using block_it = typename outer_container::iterator;
    using block_cont_it = typename block_container::iterator;
    block_cont_it cur_ptr;
    
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    
    block_it cont;
    
    iterator() = default;
    iterator(block_it cont, block_cont_it cur_ptr):
      cur_ptr{cur_ptr}, cont{cont} { }
    iterator(block_it cont):
      iterator(cont, cont->begin()) { }
    
    reference operator*() const { return *cur_ptr; }
    pointer operator->() { return cur_ptr; }
    
    iterator& operator++() { 
      cur_ptr++; 
      return *this; 
    }
    iterator operator++(int) { 
      iterator res = *this; 
      this->operator++();
      return res; 
    }
    
    iterator& operator+=(const difference_type n){
      cur_ptr += n;
      return *this;
    }
    friend iterator operator+
      (const iterator &it, const difference_type n){
      iterator out{it};
      return out += n;
    }
    friend iterator operator+
      (const difference_type n, const iterator &it){
      return it + n;
    }
    
    friend bool operator==(const iterator& a, const block_cont_it& b) { 
      return a.cur_ptr == b; 
    };
    friend bool operator==(const block_cont_it& b, const iterator& a) { 
      return a == b; 
    };
    friend bool operator==(const iterator& a, const iterator& b) { 
      return a == b.cur_ptr; 
    };
    
    friend bool operator!=(const iterator& a, const block_cont_it& b) { 
      return a.cur_ptr != b; 
    };
    friend bool operator!=(const block_cont_it& b, const iterator& a) { 
      return a != b; 
    };
    friend bool operator!=(const iterator& a, const iterator& b) { 
      return a != b.cur_ptr; 
    };
    
    friend bool operator>=(const iterator& a, const block_cont_it& b) { 
      return a.cur_ptr >= b; 
    };
    friend bool operator>=(const block_cont_it& b, const iterator& a) { 
      return b >= a.cur_ptr; 
    };
    friend bool operator>=(const iterator& a, const iterator& b) { 
      return a >= b.cur_ptr; 
    };
  };
  
  /// markers to jump back to 
  std::stack<iterator> marks;
  
  /// the current head (next free object). May be an end pointer
  iterator cur_head;
  
  /// the minimum size of the blocks
#ifdef DEBUG_SIMPLE_MEM_STACK
  static constexpr size_t min_block_size{0};
#else 
  static constexpr size_t min_block_size{32768};
#endif
  
  /**
   * creates a new block of given minimum size. The new block size will be
   * at least as great as n_ele
   */
  void new_block(size_t const n_ele){
    auto it = cur_head.cont;
    while(++it != memory.end() && it->size() < n_ele) { }
      
    if(it == memory.end()){
      // did not find a block of the appropriate size. Create a new block
#ifdef DEBUG_SIMPLE_MEM_STACK
      // allocate just enough memory
      memory.emplace_back(n_ele);
#else 
      memory.emplace_back(std::max(n_ele, memory.back().size() * 2L));
#endif
      it = std::prev(memory.end());
    }
    
    cur_head = iterator{it};
  }
  
public:
  simple_mem_stack() {
    clear();
  }
  
  /// clears the object deallocating all memory
  void clear(){
    while(!marks.empty())
      marks.pop();
    memory.clear();
    memory.emplace_back(min_block_size);
    cur_head = iterator{memory.begin()};
  }
  
  /// returns a given number of units of memory 
  T* get(size_t const n_ele) {
    if(cur_head + n_ele >= cur_head.cont->end())
      new_block(n_ele);
    
    T* res = &*cur_head;
    cur_head += n_ele;
    return res;
  }
  
  /// sets a mark in the memory that can be returned to in the future
  void set_mark(){
    marks.emplace(cur_head);
  }
  
  /// turn back the memory to the previous mark or the start if there is not any
  void reset_to_mark() {
    if(!marks.empty())
      cur_head = marks.top();
    else 
      cur_head = iterator{memory.begin()};
  }
  
  /// removes the last mark
  void pop_mark() {
    if(!marks.empty())
      marks.pop();
  }
  
  /// turns back the memory to the start without deallocating the memory
  void reset(){
    while(!marks.empty()) 
      marks.pop();
    
    cur_head = iterator{memory.begin()};
  }
  
  /** 
   * class used for RAII. It pops the marker and returns to the previous 
   * marker when the object goes out of scope unless the current marker is not 
   * the one used when the object was created
   */
  class return_memory_handler;
  friend class return_marker;
  class return_memory_handler {
    simple_mem_stack &mem_obj;
    iterator marker;
    
  public:
    return_memory_handler(simple_mem_stack &mem_obj, iterator marker):
      mem_obj{mem_obj}, marker{marker} { }
    
    ~return_memory_handler() {
      if(mem_obj.marks.empty() || mem_obj.marks.top() != marker)
        return;
      mem_obj.pop_mark();
      mem_obj.reset_to_mark();
    }
  };
  
  /** 
   * sets a mark in the memory that can be returned to in the future. When the 
   * returned object is destroyed, the marker is removed and the memory returned 
   * to the previous marker.
   */
  return_memory_handler set_mark_raii(){
    marks.emplace(cur_head);
    return { *this, cur_head };
  }
};

// ghq.h

#include <RcppArmadillo.h>
#include <algorithm>
#include <stdexcept>
#include <psqn-bfgs.h>

/**
 * virtual base class for a Gauss-Hermite quadrature problem. It specifies 
 * the dimension of the random effects and the number of outputs. The integral 
 * we approximate are of the form 
 * 
 *   int phi(x)g(x)dx 
 *   
 * where phi(x) is a given dimensional standard multivariate normal 
 * distribution. The class also computes g(x).
 * 
 * To perform adaptive Gauss-Hermite quadrature for one of the elements of g, 
 * say g_i, the class also has member functions to compute log g_i(x), the 
 * gradient of it, and the Hessian. 
 */
struct ghq_problem {
  /// the number of variables
  virtual size_t n_vars() const = 0;
  /// the number of output
  virtual size_t n_out() const = 0;
  
  /**
   * n_points x n_vars() points at which to evaluate g is passed in points. The 
   * output of evaluating g at each point is then written to n_points x n_out() 
   * matrix outs. The mem object can be used for working memory. 
   */
  virtual void eval
    (double const *points, size_t const n_points, double * __restrict__ outs, 
     simple_mem_stack<double> &mem) const = 0;
  
  /// evaluates log g_i for the element chosen for the adaptive method
  virtual double log_integrand
    (double const *point, simple_mem_stack<double> &mem) const {
    throw std::runtime_error("not implemented");
    return 0;
  }
  
  /** 
   * evaluates log g_i and the gradient for the element chosen for the adaptive 
   * method
   */
  virtual double log_integrand_grad
    (double const *point, double * __restrict__ grad,
     simple_mem_stack<double> &mem) const {
    throw std::runtime_error("not implemented");
    return 0;
  }
  
  /// evaluates the Hessian of log g_i
  virtual void log_integrand_hess
    (double const *point, double *hess, 
     simple_mem_stack<double> &mem) const {
    throw std::runtime_error("not implemented");
  }
  
  virtual ~ghq_problem() = default;
};

/// Gauss-Hermite quadrature nodes and weights
struct ghq_data {
  double const * nodes, * weights;
  size_t n_nodes;
};

/** 
 * performs Gauss-Hermite quadrature. The target_size is the maximum number of 
 * integrands to simultaneously process. However, at least the number of 
 * quadrature nodes is simultaneously processed.
 */
std::vector<double> ghq
  (ghq_data const &ghq_data_in, ghq_problem const &problem, 
   simple_mem_stack<double> &mem, size_t const target_size = 128);
   
/**
 * Takes the integral
 * 
 *   int phi(x)g(x)dx 
 *   
 * and makes is adaptive by working with 
 * 
 *   int phi(x, mu, Psi)phi(x, mu, Psi)^(-1)phi(x)g(x)dx 
 *   
 * where mu is the mode of the log of the integrand and Psi is the inverse of 
 * the negative Hessian of the log of the integrand. This is then transformed to 
 * 
 *   |Psi|^(1/2)int phi(x)[phi(x)^(-1)phi(mu + C.x)g(mu + C.x)]dx 
 *                        |-----------------------------------|
 *                                        h(x)
 *                             
 * where C.C^T = Psi is the Cholesky decomposition of Psi.
 */
class adaptive_problem final : public ghq_problem  {
  ghq_problem const &problem;
  size_t const v_n_vars{problem.n_vars()},  
               v_n_out{problem.n_out()};
  
  /**
   * the Cholesky decomposition of the -H^(-1) where H is the Hessian at the 
   * mode
   */
  arma::mat C;
  /// the mode
  arma::vec mu;
  /// the square root of the determinant of C^T.C
  double sq_C_deter;
  
public:
  /// class used to find the mode
  class mode_problem final : public PSQN::problem {
    ghq_problem const &problem;
    simple_mem_stack<double> &mem;
    PSQN::psqn_uint const v_n_vars = problem.n_vars();
    
  public:
    mode_problem(ghq_problem const &problem, simple_mem_stack<double> &mem);
    
    PSQN::psqn_uint size() const { return v_n_vars; }
    double func(double const *val);
    
    double grad(double const * __restrict__ val,
                double       * __restrict__ gr);
  };
  
  adaptive_problem(ghq_problem const &problem, simple_mem_stack<double> &mem);
  
  size_t n_vars() const { return v_n_vars; }
  size_t n_out() const { return v_n_out; }
  
  void eval
  (double const *points, size_t const n_points, double * __restrict__ outs, 
   simple_mem_stack<double> &mem) const;
};

/**
 * The class takes multiple problems g_1(x), g_2(x), ..., g_l(x) and assumes 
 * that
 * 
 *  1. the first entries, g_(11)(x), g_(21)(x), ..., g_(l1)(x), are for the 
 *     integral 
 *     
 *       A = int phi(x) prod_(i = 1)^l g_(i1)(x) dx
 *       
 *  2. the remaining entries are derivatives in the form of 
 *       
 *       g_i' = d/dz_i g_i(x; z_i). 
 *  
 * for some fixed vector z_i. The function then returns the estimator of A in 
 * the first element and the derivatives of A for each z_i. The latter 
 * derivatives are stored in the order the problems are passed and can be 
 * computed with 
 * 
 *   int phi(x) g_j'(x) / g_(j1)(x)prod_(i = 1)^l g_(i1)(x) dx
 */
class combined_problem final : public ghq_problem  {
  std::vector<ghq_problem const *> problems;
  
  // to avoid the virtual dispatch
  std::vector<size_t> const n_outs
  {
    ([&]{
      std::vector<size_t> out;
      out.reserve(problems.size());
      for(auto p : problems)
        out.emplace_back(p->n_out());
      return out;
    })()
  };
  
  size_t const v_n_vars{ problems.size() == 0 ? 0 : problems[0]->n_vars() };
  size_t const n_out_inner
    {std::accumulate(
        n_outs.begin(), n_outs.end(), static_cast<size_t>(0))};
  size_t const v_n_out{n_out_inner - problems.size() + 1};
  
public:
  combined_problem(std::vector<ghq_problem const *> const &problems);
  
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

/**
 * computes g(x) = (1, x.x^T) where only the upper triangle is stored for the 
 * outer product. This is useful when used with combined_problem for problems
 * of the form 
 * 
 *   int phi(x; 0, Sigma)g_1(x)...g_l(x) dx 
 *     = int phi(x)g_1(C.x)...g_l(C.x) dx
 *     
 * where C.C^T = Sigma is the Cholesky decomposition. In the framework defined 
 * here, we then redfine the g functions as h_i(x) = g_i(C.x). 
 * 
 * The derivatives w.r.t. Sigma in this case is 
 * 
 *   1/2 Sigma^(-1)
 *     [int (x.x^T - Sigma) phi(x; 0, Sigma)g_1(x)...g_l(x) dx]Sigma^(-1)
 *    = 1/2 C^(-T)
 *      [int (x.x^T - I) phi(x; 0, Sigma)h_1(x)...h_l(x) dx]C^(-1)
 *      
 * so we need to compute 
 * 
 *   int x.x^T phi(x; 0, Sigma)h_1(x)...h_l(x) dx
 *   
 * to compute the gradient w.r.t. Sigma.
 */
class outer_prod_problem final : public ghq_problem  {
  size_t const v_n_vars,
               v_n_out{1 + (v_n_vars * (v_n_vars + 1)) / 2};
  
public:
  outer_prod_problem(size_t const n_vars);
  
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
  
  /**
   * computes the derivatives w.r.t. Sigma given the last 
   * (n_vars() * (n_vars() + 1)) / 2 elements of the final output, the value of
   *  the entire integral, and the covariance matrix Sigma. 
   *  
   *  The result is stored in a n_vars() x n_vars() matrix. 
   */
  void d_Sig
    (double * __restrict__ res, double const *out, double const integral, 
     arma::mat const &Sigma) const ;
};

// ghq-lp-utils.h

#include <R_ext/RS.h> // for F77_NAME and F77_CALL

extern "C" {
  void F77_NAME(dtrmm)
    (const char *side, const char *uplo, const char *transa, 
     const char *diag, const int *m, const int *n, const double *alpha, 
     const double *A, const int *lda, double *B, 
     const int *ldb, 
     size_t, size_t, size_t, size_t);
  
  // TODO: possibly use DTPMV for the matrix-vector calls
}

// integrand-moment-test.h

/**
 * simple test function that computes the moment generating function 
 * f(t) = E(exp(t^T.X)) of X ~ N(mu, Sigma). 
 */
class moment_test final : public ghq_problem {
  double offset;
  arma::vec t_Sigma_chol;
  size_t const v_n_vars = t_Sigma_chol.n_elem;
  
public:
  moment_test(arma::vec const &t, arma::vec const &mu, arma::mat const &Sigma):
    offset{arma::dot(t, mu)}, 
    t_Sigma_chol{(t.t() * arma::chol(Sigma, "lower")).t()} { }

  size_t n_vars() const {
    return v_n_vars;
  }

  size_t n_out() const {
    return 1;
  }

  void eval
    (double const *points, size_t const n_points, double * __restrict__ outs, 
     simple_mem_stack<double> &mem) const {
    double * const __restrict__ lp{mem.get(n_points)};
    std::fill(lp, lp + n_points, offset);
    
    for(size_t j = 0; j < n_vars(); ++j)
      for(size_t i = 0; i < n_points; ++i, ++points)
        lp[i] += t_Sigma_chol[j] * *points;
    
    for(size_t i = 0; i < n_points; ++i)
      outs[i] = std::exp(lp[i]);
  }
};

// integrand-mixed-mult-logit-term.h

/**
 * computes the likelihood of a mixed multinomial term for a cluster. That is,
 * if there K groups, the probability that outcome i is in level k > 0 given the 
 * random effects is 
 * 
 *   exp(eta[i, k - 1] + u[k - 1]) / 
 *     (1 + sum_(j = 1)^K exp(eta[i, j - 1] + u[j - 1]))
 *     
 *  and the probability of the reference level, k = 0, is 
 *  
 *    1 / (1 + sum_(j = 1)^K exp(eta[i, j - 1] + u[j - 1]))
 *    
 *  The random effect is assumed to be N(0, Sigma). 
 *  
 *  If the gradient is required, the first output is the integral and the next 
 *  elements are the gradients w.r.t. the eta matrix.
 */
template<bool comp_grad = false>
class mixed_mult_logit_term final : public ghq_problem {
  /**
   * the fixed offsets in the linear predictor stored 
   * (K - 1) x <number of outcomes> matrix
   */
  arma::mat const &eta;
  arma::mat const Sigma_chol; 
  // the category of each outcome. It is zero indexed
  arma::uvec const &which_category;
  
  size_t const v_n_vars = eta.n_rows, 
               v_n_out{comp_grad ? 1 + eta.n_rows * eta.n_cols: 1};
  
public:
  mixed_mult_logit_term(arma::mat const &eta, arma::mat const &Sigma, 
                   arma::uvec const &which_category);
  
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

// integrand-mixed-probit-term.h

/**
 * computes the likelihood of single mixed probit term. That is the probability 
 * 
 *   g(u) = Phi(eta + z^Tu)
 *   
 * where u ~ N(0, Sigma). The function can easily be extended to more than one 
 * probit factor if needed. The gradient is with respect to the scalar eta and 
 * the vector z.
 */
template<bool comp_grad = false>
class mixed_probit_term final : public ghq_problem {
  double const eta;
  arma::mat const Sigma_chol;
  arma::vec const Sigma_chol_z;
  
  size_t const v_n_vars = Sigma_chol_z.n_elem, 
                v_n_out{comp_grad ? 2 + Sigma_chol_z.n_elem: 1};
  
public:
  mixed_probit_term
    (double const eta, arma::mat const &Sigma, arma::vec const &z);
  
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

// integrand-expected-survival.h

/**
 * computes the expected survival. That is 
 * 
 *   E(exp(-sum w_i[i] * exp(eta[i] + M.U)))
 *   
 * given n weights and offsets w and eta and a matrix M x R. U is assumed to be 
 * a R dimensional random variable which is ~ N(0, Sigma).
 * 
 * TODO: something about the derivatives.
 */
template<bool comp_grad = false>
class expected_survival_term final : public ghq_problem {
  arma::vec const &eta, &weights;
  arma::mat const M_Sigma_chol_t;
  
  size_t const v_n_vars = M_Sigma_chol_t.n_cols, 
               v_n_out{comp_grad ? 1 : 1};
  
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

// ghq.cpp

adaptive_problem::mode_problem::mode_problem
  (ghq_problem const &problem, simple_mem_stack<double> &mem):
  problem{problem}, mem{mem} { }

double adaptive_problem::mode_problem::func(double const *val){
  double out{};
  for(PSQN::psqn_uint i = 0; i < size(); ++i)
    out += val[i] * val[i];
  out /= 2;
  out -= problem.log_integrand(val, mem);
  return out;
}

double adaptive_problem::mode_problem::grad
  (double const * __restrict__ val, double * __restrict__ gr){
  double const out{-problem.log_integrand_grad(val, gr, mem)};
  std::for_each(gr, gr + size(), [](double &res){ res *= -1; });
  
  double extra_term{};
  for(PSQN::psqn_uint i = 0; i < size(); ++i){
    extra_term += val[i] * val[i];
    gr[i] += val[i];
  }
  extra_term /= 2;
  
  return out + extra_term;
}

adaptive_problem::adaptive_problem
  (ghq_problem const &problem, simple_mem_stack<double> &mem):
  problem{problem} {
    // attempt to find the mode
    mode_problem my_mode_problem(problem, mem);
    mu.zeros(n_vars());
    
    // TODO: let the caller set the thresholds etc. 
    // TODO: I can avoid the allocation in PSQN::bfgs with minor changes in the 
    //       package
    auto res = PSQN::bfgs
      (my_mode_problem, mu.memptr(), 1e-4, 1000L, 1e-4, .9, 0L, -1);
    
    bool succeeded = res.info == PSQN::info_code::converged;
    if(succeeded){
      // we compute the Hessian
      arma::mat hess(mem.get(2 * n_vars() * n_vars()), n_vars(), n_vars(), 
                     false),
            hess_inv(hess.end(), n_vars(), n_vars(), false);
      problem.log_integrand_hess(mu.memptr(), hess.memptr(), mem);
      hess.for_each([](double &res){ res *= -1; });
      for(size_t i = 0; i < n_vars(); ++i)
        hess(i, i) += 1;
      
      if((succeeded = arma::inv_sympd(hess_inv, hess))){
        succeeded = arma::chol(C, hess_inv);
        
        sq_C_deter = 1;
        for(arma::uword i = 0; i < C.n_cols; ++i)
          sq_C_deter *= C(i, i);
      }
    }
    
    if(!succeeded){
      // perform the computation with a non-adaptive version
      mu.zeros(n_vars());
      C.zeros(n_vars(), n_vars());
      C.diag() += 1;
      sq_C_deter = 1;
    }
    mem.reset_to_mark();
  }

void adaptive_problem::eval
  (double const *points, size_t const n_points, double * __restrict__ outs, 
   simple_mem_stack<double> &mem) const {
  /// transform the points
  double * const __restrict__ 
    points_trans{mem.get(n_vars() * n_points + n_points)};
  double * const __restrict__ fac{points_trans + n_vars() * n_points};
  
  // do the matrix product points.C
  std::copy(points, points + n_points * n_vars(), points_trans);
  {
    int const m = n_points, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, C.memptr(), &n, 
       points_trans, &m, 1, 1, 1, 1);
  }
  
  // add the mode
  for(size_t j = 0; j < n_vars(); ++j)
    std::for_each
      (points_trans + j * n_points, points_trans + (j + 1) * n_points, 
       [&](double &lhs){ lhs += mu[j]; });
    
  // evaluate the inner part
  auto mem_marker = mem.set_mark_raii(); // problem may turn back mem
  problem.eval(points_trans, n_points, outs, mem);
  
  // add the additional weight
  std::fill(fac, fac + n_points, 0);
  for(size_t j = 0; j < n_vars(); ++j){
    size_t const offset{j * n_points};
    for(size_t i = 0; i < n_points; ++i)
      fac[i] += 
        points[i + offset] * points[i + offset]
        - points_trans[i + offset] * points_trans[i + offset];
  }
  
  std::for_each
    (fac, fac + n_points, 
     [&](double &res) { res = std::exp(res / 2) * sq_C_deter; });
  
  for(size_t j = 0; j < n_out(); ++j)
    for(size_t i = 0; i < n_points; ++i)
      outs[i + j * n_points] *= fac[i];
}

// recursive functions needed for quadrature implementation
namespace {
void ghq_fill_fixed
  (size_t const lvl, double * const points, double * const weights, 
   size_t const n_points, ghq_data const &dat){
  // how many times should we repeat each node?
  size_t const n_nodes{dat.n_nodes};
  size_t n_rep{1};
  for(size_t i = 1; i < lvl; ++i)
    n_rep *= n_nodes;
  
  // fill the weights and points
  double *p{points}, *w{weights};
  for(size_t j = 0; j < n_points;)
      for(size_t n = 0; n < n_nodes; ++n)
        for(size_t i = 0; i < n_rep; ++i, ++j){
          *p++  = dat.nodes[n];
          *w++ *= dat.weights[n];
        }
        
  if(lvl > 1)
    ghq_fill_fixed(lvl - 1, points + n_points, weights, n_points, dat);
}

void ghq_inner
  (std::vector<double> &res, double * const outs, size_t const lvl, 
   size_t const idx_fix, size_t const n_points, size_t const n_vars,
   double * const points, double const * weights, 
   ghq_problem const &problem, ghq_data const &dat, 
   simple_mem_stack<double> &mem){
  if(lvl == idx_fix){
    // evaluate the integrand and add the result
    problem.eval(points, n_points, outs, mem);
    mem.reset_to_mark();
    
    size_t const n_res{res.size()};
    for(size_t i = 0; i < n_res; ++i)
      for(size_t j = 0; j < n_points; ++j)
        res[i] += weights[j] * outs[j + i * n_points];
    
    return;
  }
  
  // we have to go through all the configurations recursively
  double * const __restrict__ weights_scaled{mem.get(n_points)};
  auto mem_marker = mem.set_mark_raii();
  
  size_t const n_nodes{dat.n_nodes};
  for(size_t j  = 0; j < n_nodes; ++j){
    double * const p{points + (n_vars - lvl) * n_points};
    for(size_t i = 0; i < n_points; ++i){
      weights_scaled[i] = dat.weights[j] * weights[i];
      p[i] = dat.nodes[j];
    }
    
    // run the next level
    ghq_inner(res, outs, lvl - 1, idx_fix, n_points, n_vars, points, 
              weights_scaled, problem, dat, mem);
  }
}
} // namespace

std::vector<double> ghq
  (ghq_data const &ghq_data_in, ghq_problem const &problem, 
   simple_mem_stack<double> &mem, size_t const target_size){
  size_t const n_nodes{ghq_data_in.n_nodes},
               n_vars{problem.n_vars()},
               n_out{problem.n_out()};
  
  // checks
  if(n_out < 1)
    return {};
  else if(n_nodes < 1)
    throw std::invalid_argument("n_nodes < 1");
  else if(n_vars < 1)
    throw std::invalid_argument("n_vars < 1");
  
  // determine the maximum number of points we will use and the "fixed" level
  size_t idx_fix{1};
  size_t n_points{n_nodes};
  for(; n_points * n_nodes < target_size && idx_fix < n_vars; ++idx_fix)
    n_points *= n_nodes;
  
  // get the memory we need 
  double * const points
    {mem.get(2 * n_nodes + n_points * (1 + n_vars + n_out))},
         * const outs{points + n_points * n_vars},
         * const weights{outs + n_points * n_out}, 
         * const ghq_nodes{weights + n_points}, 
         * const ghq_weigths{ghq_nodes + n_nodes};
  
  auto mem_marker = mem.set_mark_raii();
  
  // initialize the objects before the computation
  std::fill(weights, weights + n_points, 1);
  std::vector<double> res(n_out, 0);
  
  for(size_t i = 0; i < n_nodes; ++i){
    ghq_nodes[i] = ghq_data_in.nodes[i] * 1.4142135623731;  // sqrt(2)
    ghq_weigths[i] = ghq_data_in.weights[i] * 0.564189583547756; // 1 / sqrt(pi)
  }
  
  ghq_data const ghq_data_use{ghq_nodes, ghq_weigths, n_nodes};
  
  // the points matrix has a "fixed part" that never changes and set the 
  // corresponding weights 
  ghq_fill_fixed
    (idx_fix, points + n_points * (n_vars - idx_fix), weights, n_points, 
     ghq_data_use);
  
  ghq_inner(res, outs, n_vars, idx_fix, n_points, n_vars, points, weights, 
            problem, ghq_data_use, mem);
  
  return res;
}

combined_problem::combined_problem
  (std::vector<ghq_problem const *> const &problems):
  problems{problems} {
    if(problems.size() > 0){
      size_t const n_vars_first{problems[0]->n_vars()};
      for(ghq_problem const * p : problems){
        if(p->n_vars() != n_vars_first)
          throw std::invalid_argument("p->n_vars() != n_vars_first");
        else if(p->n_out() < 1)
          throw std::invalid_argument("p->n_out() < 1");
      }
    }
  }

void combined_problem::eval
  (double const *points, size_t const n_points, double * __restrict__ outs, 
   simple_mem_stack<double> &mem) const {
  double * const __restrict__ scales{mem.get(n_points * (1 + n_out_inner))},
         * const __restrict__ outs_inner{scales + n_points};
  auto mem_marker = mem.set_mark_raii();
  
  // call eval on each of the problems while setting the value of the integrand
  double * const integrands{outs};
  outs += n_points; // outs are now the derivatives
  std::fill(integrands, integrands + n_points, 1);
  {
    double * outs_inner_p{outs_inner};
    size_t pi{};
    for(auto p : problems){
      p->eval(points, n_points, outs_inner_p, mem);
      
      for(size_t i = 0; i < n_points; ++i)
        integrands[i] *= outs_inner_p[i];
      outs_inner_p += n_outs[pi] * n_points;
      ++pi;
    }
  }
  
  // compute the derivatives
  double const * outs_inner_p{outs_inner};
  for(size_t const n_outs_p : n_outs){
    if(n_outs_p > 1){
      // compute the scales to use for the derivatives
      for(size_t i = 0; i < n_points; ++i, ++outs_inner_p)
        scales[i] = integrands[i] / *outs_inner_p;
      
      // set the derivatives
      for(size_t j = 0; j < n_outs_p - 1; ++j)
        for(size_t i = 0; i < n_points; ++i, ++outs_inner_p, ++outs)
          *outs = *outs_inner_p * scales[i];
      
    } else
      outs_inner_p += n_points;
  }
}

double combined_problem::log_integrand
  (double const *point, simple_mem_stack<double> &mem) const {
  double out{};
  for(auto p : problems)
    out += p->log_integrand(point, mem);
  return out;
}

double combined_problem::log_integrand_grad
  (double const *point, double * __restrict__ grad,
   simple_mem_stack<double> &mem) const {
  double * const grad_inner{mem.get(n_vars())};
  auto mem_marker = mem.set_mark_raii();
  
  std::fill(grad, grad + n_vars(), 0);
  double out{};
  for(auto p : problems){
    out += p->log_integrand_grad(point, grad_inner, mem);
    for(size_t i = 0; i < n_vars(); ++i)
      grad[i] += grad_inner[i];
  }
  return out;
}

void combined_problem::log_integrand_hess
  (double const *point, double *hess, 
   simple_mem_stack<double> &mem) const {
  size_t const n_vars_sq{n_vars() * n_vars()};
  double * const hess_inner{mem.get(n_vars_sq)};
  auto mem_marker = mem.set_mark_raii();
  
  std::fill(hess, hess + n_vars_sq, 0);
  for(auto p : problems){
    p->log_integrand_hess(point, hess_inner, mem);
    for(size_t i = 0; i < n_vars_sq; ++i)
      hess[i] += hess_inner[i];
  }
}

outer_prod_problem::outer_prod_problem(size_t const n_vars):
  v_n_vars{n_vars} { }

void outer_prod_problem::eval
  (double const *points, size_t const n_points, double * __restrict__ outs, 
   simple_mem_stack<double> &mem) const {
  // have till fill in the ones
  std::fill(outs, outs + n_points, 1);
  outs += n_points;

  // compute the outer products
  size_t out_offset{};
  for(size_t j = 0; j < n_vars(); ++j)
    for(size_t k = 0; k <= j; ++k, ++out_offset)
        for(size_t i = 0; i < n_points; ++i)
          outs[i + out_offset * n_points] = 
            points[i + k * n_points] * points[i + j * n_points];
}

double outer_prod_problem::log_integrand
  (double const*, simple_mem_stack<double>&) const {
  return 0;
}

double outer_prod_problem::log_integrand_grad
  (double const*, double * __restrict__ gr, simple_mem_stack<double>&) const {
  std::fill(gr, gr + n_vars(), 0);
  return 0;
}

void outer_prod_problem::log_integrand_hess
  (double const*, double * __restrict__ hess, simple_mem_stack<double>&) const {
  std::fill(hess, hess + n_vars() * n_vars(), 0);
}

void outer_prod_problem::d_Sig
  (double * __restrict__ res, double const *out, double const integral, 
   arma::mat const &Sigma) const {
  arma::mat outer_int(n_vars(), n_vars());
  for(arma::uword j = 0; j < outer_int.n_cols; ++j, ++out){
    for(arma::uword i = 0; i < j; ++i, ++out){
      outer_int(i, j) = *out / 2;
      outer_int(j, i) = *out / 2;
    }
    outer_int(j, j) = (*out - integral) / 2;
  }
  
  arma::mat const sig_chol = arma::chol(Sigma);  
  arma::mat lhs(res, n_vars(), n_vars(), false);
  
  lhs = arma::solve
    (arma::trimatu(sig_chol),
     arma::solve(arma::trimatu(sig_chol), outer_int).t());
}

// integrand-mixed-mult-logit-term.cpp

template<bool comp_grad>
mixed_mult_logit_term<comp_grad>::mixed_mult_logit_term
  (arma::mat const &eta, arma::mat const &Sigma, 
   arma::uvec const &which_category):
  eta{eta}, Sigma_chol{arma::chol(Sigma, "upper")}, 
  which_category{which_category} {
  if(which_category.n_elem != eta.n_cols)
    throw std::invalid_argument("which_category.n_elem != eta.n_cols");
  else if(Sigma.n_cols != Sigma.n_rows)
    throw std::invalid_argument("Sigma.n_cols != Sigma.n_rows");
  else if(Sigma.n_cols != eta.n_rows)
    throw std::invalid_argument("Sigma.n_cols != eta.n_rows");
  for(arma::uword i : which_category)
    if(i > eta.n_rows)
      throw std::invalid_argument
        ("which_category has entries with i > eta.n_rows");
}
 
// TODO: test this function
template<bool comp_grad>
void mixed_mult_logit_term<comp_grad>::eval
  (double const *points, size_t const n_points, double * __restrict__ outs, 
   simple_mem_stack<double> &mem) const {
  double * const __restrict__ us{mem.get(n_points * n_vars() + n_vars())}, 
         * const __restrict__ us_j{us + n_points * n_vars()};
  
  // do the matrix product points.chol(Sigma)
  std::copy(points, points + n_points * n_vars(), us);
  {
    int const m = n_points, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n, 
       us, &m, 1, 1, 1, 1);
  }
  
  if constexpr (comp_grad){
    double * const terms{mem.get(2 * eta.n_cols + eta.n_cols * eta.n_rows)}, 
           * const denoms{terms + eta.n_cols},
           * const lps{denoms + eta.n_cols};
    
    // do the computations point by point
    for(size_t j = 0; j < n_points; ++j){
      // copy the random effect
      for(size_t i = 0; i < n_vars(); ++i)
        us_j[i] = us[j + i * n_points];
      
      // compute the integrand
      outs[j] = 1;
      for(arma::uword k = 0; k < eta.n_cols; ++k){
        size_t const offset = k * n_vars();
        denoms[k] = 1;
        double const * eta_k{eta.colptr(k)};
        for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
          lps[i + offset] = std::exp(*eta_k + us_j[i]);
          denoms[k] += lps[i + offset];
        }
        
        double const numerator
          {which_category[k] < 1 ? 1 : lps[which_category[k] - 1 + offset]};
        terms[k] = numerator / denoms[k];
        outs[j] *= terms[k]; 
      }
      
      // compute the gradient
      double * d_eta_j{outs + j + n_points};
      for(arma::uword k = 0; k < eta.n_cols; ++k)
        for(size_t i = 0; i < n_vars(); ++i, d_eta_j += n_points){
          *d_eta_j = i + 1 == which_category[k]
            ? (denoms[k] - lps[i + k * n_vars()])  
            : -lps[i + k * n_vars()];
          *d_eta_j *= outs[j] / denoms[k];
        }
    }
    
  } else {
    double * const __restrict__ lp{mem.get(n_vars())};
    
    // do the computations point by point
    for(size_t j = 0; j < n_points; ++j){
      // copy the random effect
      for(size_t i = 0; i < n_vars(); ++i)
        us_j[i] = us[j + i * n_points];
          
      outs[j] = 1;
      for(arma::uword k = 0; k < eta.n_cols; ++k){
        double denom{1};
        double const * eta_k{eta.colptr(k)};
        for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
          lp[i] = std::exp(*eta_k + us_j[i]);
          denom += lp[i];
        }
        
        double const numerator
          {which_category[k] < 1 ? 1 : lp[which_category[k] - 1]};
        outs[j] *= numerator / denom;
      }
    }
  }
}
  
// TODO: test this function
template<bool comp_grad>
double mixed_mult_logit_term<comp_grad>::log_integrand
  (double const *point, simple_mem_stack<double> &mem) const {
  double * const __restrict__ u{mem.get(2 * n_vars())}, 
         * const __restrict__ lp{u + n_vars()};
  
  // do the matrix product point.chol(Sigma)
  std::copy(point, point + n_vars(), u);
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n, 
       u, &m, 1, 1, 1, 1);
  }
 
  double out{};
  for(arma::uword k = 0; k < eta.n_cols; ++k){
    double denom{1};
    double const * eta_k{eta.colptr(k)};
    for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
      lp[i] = *eta_k + u[i];
      denom += std::exp(lp[i]);
    }
    
    if(which_category[k] < 1)
      out -= log(denom);
    else 
      out += lp[which_category[k] - 1] - log(denom);
  }
  
  return out;
}
  
// TODO: test this function
template<bool comp_grad>
double mixed_mult_logit_term<comp_grad>::log_integrand_grad
  (double const *point, double * __restrict__ grad,
   simple_mem_stack<double> &mem) const {
  double * const __restrict__ u{mem.get(3 * n_vars())}, 
         * const __restrict__ lp{u + n_vars()},
         * const __restrict__ lp_exp{lp + n_vars()};
  
  std::copy(point, point + n_vars(), u);
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n, 
       u, &m, 1, 1, 1, 1);
  }
  
  double out{};
  std::fill(grad, grad + n_vars(), 0);
  for(arma::uword k = 0; k < eta.n_cols; ++k){
    double denom{1};
    double const * eta_k{eta.colptr(k)};
    for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
      lp[i] = *eta_k + u[i];
      lp_exp[i] = exp(lp[i]);
      denom += lp_exp[i];
    }
    
    // handle the denominator term of the derivative
    for(size_t i = 0; i < n_vars(); ++i)
      grad[i] -= lp_exp[i] / denom;
    
    if(which_category[k] < 1)
      out -= log(denom);
    else {
      out += lp[which_category[k] - 1] - log(denom);
      grad[which_category[k] - 1] += 1;
    }
  }
  
  // compute grad <- Sigma_chol.grad
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_L{'L'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_L, &c_U, &c_N, &c_N, &n, &m, &alpha, Sigma_chol.memptr(), &n,
       grad, &n, 1, 1, 1, 1);
  }
  
  return out;
}
  
// TODO: test this function
template<bool comp_grad>
void mixed_mult_logit_term<comp_grad>::log_integrand_hess
  (double const *point, double *hess, 
 simple_mem_stack<double> &mem) const {
  double * const __restrict__ u{mem.get(2 * n_vars())},
         * const __restrict__ lp_exp{u + n_vars()};
  
  std::copy(point, point + n_vars(), u);
  {
    int const m = 1, n = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n, 
       u, &m, 1, 1, 1, 1);
  }
  
  std::fill(hess, hess + n_vars() * n_vars(), 0);
  for(arma::uword k = 0; k < eta.n_cols; ++k){
    double denom{1};
    double const * eta_k{eta.colptr(k)};
    for(size_t i = 0; i < n_vars(); ++i, ++eta_k){
      lp_exp[i] = exp(*eta_k + u[i]);
      denom += lp_exp[i];
    }
    
    double const denom_sq{denom * denom};
    for(size_t j = 0; j < n_vars(); ++j){
      for(size_t i = 0; i < j; ++i){
        double entry{lp_exp[i] * lp_exp[j] / denom_sq};
        hess[i + j * n_vars()] += entry;
        hess[j + i * n_vars()] += entry;
      }
      hess[j + j * n_vars()] -= 
        (denom - lp_exp[j]) * lp_exp[j] / denom_sq;
    }
  }
  
  // compute hess <- Sigma_chol.hess.Sigma_chol^T
  {
    int const m = n_vars();
    constexpr double const alpha{1};
    constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'}, c_T{'T'}, c_L{'L'};
    F77_CALL(dtrmm)
      (&c_L, &c_U, &c_N, &c_N, &m, &m, &alpha, Sigma_chol.memptr(), &m, 
       hess, &m, 1, 1, 1, 1);
    F77_CALL(dtrmm)
      (&c_R, &c_U, &c_T, &c_N, &m, &m, &alpha, Sigma_chol.memptr(), &m, 
       hess, &m, 1, 1, 1, 1);
  }
}

template class mixed_mult_logit_term<false>;
template class mixed_mult_logit_term<true>;

// integrand-mixed-probit-term.cpp

#include <Rmath.h> // Rf_dnorm4 and Rf_pnorm5

template<bool comp_grad>
mixed_probit_term<comp_grad>::mixed_probit_term
  (double const eta, arma::mat const &Sigma, arma::vec const &z):
  eta{eta}, Sigma_chol{arma::chol(Sigma, "upper")}, 
  Sigma_chol_z{Sigma_chol * z} {
    if(z.n_elem != Sigma.n_cols)
      throw std::invalid_argument("z.n_elem != Sigma_chol.n_cols");
  }

// TODO: test this function
template<bool comp_grad>
void mixed_probit_term<comp_grad>::eval
  (double const *points, size_t const n_points, double * __restrict__ outs, 
   simple_mem_stack<double> &mem) const {
  // compute the linear predictor
  double * const __restrict__ lps{mem.get(n_points)};
  std::fill(lps, lps + n_points, eta);
  for(size_t j = 0; j < n_vars(); ++j)
    for(size_t i = 0; i < n_points; ++i)
      lps[i] += points[i + j * n_points] * Sigma_chol_z[j];
  
  // set the output 
  for(size_t i = 0; i < n_points; ++i, ++outs)
    *outs = Rf_pnorm5(lps[i], 0, 1, 1, 0);
  
  if constexpr (comp_grad){
    // the derivative w.r.t. eta
    for(size_t i = 0; i < n_points; ++i, ++outs){
      lps[i] = Rf_dnorm4(lps[i], 0, 1, 0);
      *outs = lps[i]; 
    }
    
    // the derivatives w.r.t. z
    double * const us{mem.get(n_points * n_vars())};
    std::copy(points, points + n_points * n_vars(), us);
    {
      int const m = n_points, n = n_vars();
      constexpr double const alpha{1};
      constexpr char const c_R{'R'}, c_U{'U'}, c_N{'N'};
      F77_CALL(dtrmm)
        (&c_R, &c_U, &c_N, &c_N, &m, &n, &alpha, Sigma_chol.memptr(), &n, 
         us, &m, 1, 1, 1, 1);
    }
    
    double const *us_ij{us};
    for(size_t j = 0; j < n_vars(); ++j)
      for(size_t i = 0; i < n_points; ++i, ++outs, ++us_ij)
        *outs = lps[i] * *us_ij;
  }
}

// TODO: test this function
template<bool comp_grad>
double mixed_probit_term<comp_grad>::log_integrand
  (double const *point, simple_mem_stack<double> &mem) const {
  double lp{eta};
  for(size_t i = 0; i < n_vars(); ++i)
    lp += point[i] * Sigma_chol_z[i];
  return Rf_pnorm5(lp, 0, 1, 1, 1);
}

// TODO: test this function
template<bool comp_grad>
double mixed_probit_term<comp_grad>::log_integrand_grad
  (double const *point, double * __restrict__ grad, 
   simple_mem_stack<double> &mem) const {
  double lp{eta};
  for(size_t i = 0; i < n_vars(); ++i)
    lp += point[i] * Sigma_chol_z[i];
  
  double const log_pnrm{Rf_pnorm5(lp, 0, 1, 1, 1)}, 
               log_dnrm{Rf_dnorm4(lp, 0, 1, 1)},
               d_lp{std::exp(log_dnrm - log_pnrm)};
  
  for(size_t i = 0; i < n_vars(); ++i)
    grad[i] = Sigma_chol_z[i] * d_lp;
  
  return log_pnrm;
}

// TODO: test this function
template<bool comp_grad>
void mixed_probit_term<comp_grad>::log_integrand_hess
  (double const *point, double *hess, 
   simple_mem_stack<double> &mem) const {
  double lp{eta};
  for(size_t i = 0; i < n_vars(); ++i)
    lp += point[i] * Sigma_chol_z[i];
  
  double const log_pnrm{Rf_pnorm5(lp, 0, 1, 1, 1)}, 
               log_dnrm{Rf_dnorm4(lp, 0, 1, 1)},
                  ratio{std::exp(log_dnrm - log_pnrm)},
                d_lp_sq{-(lp * ratio + ratio * ratio)};
  
  for(size_t j = 0; j < n_vars(); ++j)
    for(size_t i = 0; i < n_vars(); ++i)
      hess[i + j * n_vars()] = Sigma_chol_z[i] * Sigma_chol_z[j] * d_lp_sq;
}

template class mixed_probit_term<false>;
template class mixed_probit_term<true>;

// integrand-expected-survival.cpp

template<bool comp_grad>
expected_survival_term<comp_grad>::expected_survival_term
  (arma::vec const &eta, arma::vec const &weights, arma::mat const &M, 
   arma::mat const &Sigma):
  eta{eta}, weights{weights}, M_Sigma_chol_t{M * arma::chol(Sigma).t()} {
    if(eta.n_elem != weights.n_elem)
      throw std::invalid_argument("eta.n_elem != weights.n_elem");
    else if(eta.n_elem != M_Sigma_chol_t.n_rows)
      throw std::invalid_argument("eta.n_elem != M_Sigma_chol_t.n_rows");
  }

// TODO: test
template<bool comp_grad>
void expected_survival_term<comp_grad>::eval
  (double const *points, size_t const n_points, double * __restrict__ outs, 
   simple_mem_stack<double> &mem) const {
  size_t const n_lps = M_Sigma_chol_t.n_rows;
  double * const __restrict__ lps{mem.get(n_lps * n_points)};
  
  // add the offset
  for(size_t j = 0; j < n_points; ++j)
    std::copy(eta.begin(), eta.end(), lps + j * n_lps);
  
  // add the terms from the random effects
  {
    double const * m{M_Sigma_chol_t.begin()};
    for(size_t k = 0; k < n_vars(); ++k)
      for(size_t j = 0; j < n_points; ++j)
        for(size_t i = 0; i < n_lps; ++i)
          lps[i + j * n_lps] += m[i + k * n_lps] * points[j + k * n_points];
  }
  
  // compute the weighted sum
  for(size_t j = 0; j < n_points; ++j){
    outs[j] = 0;
    for(size_t i = 0; i < n_lps; ++i)
      outs[j] -= weights[i] * std::exp(lps[i + j * n_lps]);
    
    outs[j] = std::exp(outs[j]);
  }
  
  // TODO: implement the gradient
}

// TODO: test
template<bool comp_grad>
double expected_survival_term<comp_grad>::log_integrand
  (double const *point, simple_mem_stack<double> &mem) const {
  size_t const n_lps = M_Sigma_chol_t.n_rows;
  double * const __restrict__ lp{mem.get(n_lps)};
  std::copy(eta.begin(), eta.end(), lp);
  
  {
    double const * m{M_Sigma_chol_t.begin()};
    for(size_t k = 0; k < n_vars(); ++k)
      for(size_t i = 0; i < n_lps; ++i)
        lp[i] += m[i + k * n_lps] * point[k];
  }
  
  double out{};
  for(size_t i = 0; i < n_lps; ++i)
    out -= weights[i] * std::exp(lp[i]);
  
  return out;
}

// TODO: test
template<bool comp_grad>
double expected_survival_term<comp_grad>::log_integrand_grad
  (double const *point, double * __restrict__ grad,
   simple_mem_stack<double> &mem) const {
  size_t const n_lps = M_Sigma_chol_t.n_rows;
  double * const __restrict__ lp{mem.get(n_lps)};
  std::copy(eta.begin(), eta.end(), lp);
  
  {
    double const * m{M_Sigma_chol_t.begin()};
    for(size_t k = 0; k < n_vars(); ++k)
      for(size_t i = 0; i < n_lps; ++i)
        lp[i] += m[i + k * n_lps] * point[k];
  }
  
  double out{};
  for(size_t i = 0; i < n_lps; ++i){
    lp[i] = -weights[i] * std::exp(lp[i]); // the derivative w.r.t. lp
    out += lp[i];
  }
  
  std::fill(grad, grad + n_vars(), 0);
  double const * m{M_Sigma_chol_t.begin()};
  for(size_t k = 0; k < n_vars(); ++k)
    for(size_t i = 0; i < n_lps; ++i)
      grad[k] += m[i + k * n_lps] * lp[i];
  
  return out;
}

// TODO: test
template<bool comp_grad>
void expected_survival_term<comp_grad>::log_integrand_hess
  (double const *point, double *hess, simple_mem_stack<double> &mem) const {
  size_t const n_lps = M_Sigma_chol_t.n_rows;
  double * const __restrict__ lp{mem.get(n_lps)};
  std::copy(eta.begin(), eta.end(), lp);
  
  {
    double const * m{M_Sigma_chol_t.begin()};
    for(size_t k = 0; k < n_vars(); ++k)
      for(size_t i = 0; i < n_lps; ++i)
        lp[i] += m[i + k * n_lps] * point[k];
  }
  
  double out{};
  for(size_t i = 0; i < n_lps; ++i){
    lp[i] = -weights[i] * std::exp(lp[i]); // the derivative w.r.t. lp
    out += lp[i];
  }
  
  // TODO: do this smarter
  arma::mat H(hess, n_vars(), n_vars(), false);
  arma::mat dum(n_lps, n_lps, arma::fill::zeros);
  for(arma::uword i = 0; i < n_lps; ++i)
    dum(i, i) = lp[i];
  
  H = M_Sigma_chol_t.t() * dum * M_Sigma_chol_t;
}

template class expected_survival_term<false>;
template class expected_survival_term<true>;

// R-interface.cpp

ghq_data vecs_to_ghq_data(arma::vec const &weights, arma::vec const &nodes){
  if(nodes.size() != weights.size())
    throw std::invalid_argument("nodes.size() != weights.size()");
  return { &nodes[0], &weights[0], nodes.size() };
}

// [[Rcpp::export(rng = false)]]
double mult_norm_moment
  (arma::vec const &t, arma::vec const &mu, arma::mat const &Sigma,
   arma::vec const &weights, arma::vec const &nodes, 
   size_t const target_size = 128, size_t const n_rep = 1){
  simple_mem_stack<double> mem;
  moment_test prob{t, mu, Sigma};
  
  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);
  
  std::vector<double> res;
  for(size_t i = 0; i < n_rep; ++i){
    mem.reset();
    res = ghq(ghq_data_pass, prob, mem, target_size);
  }
  return res[0];
}

// [[Rcpp::export("mixed_mult_logit_term", rng = false)]]
double mixed_mult_logit_term_to_R
  (arma::mat const &eta, arma::mat const &Sigma, 
   arma::uvec const &which_category, arma::vec const &weights, 
   arma::vec const &nodes, size_t const target_size = 128, 
   size_t const n_rep = 1, bool const use_adaptive = false){
  simple_mem_stack<double> mem;
  mixed_mult_logit_term prob(eta, Sigma, which_category);
  
  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);

  std::vector<double> res;  
  if(use_adaptive){
    adaptive_problem prob_adap(prob, mem);
    
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob_adap, mem, target_size);
    }
    
  } else 
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob, mem, target_size);
    }
    
  return res[0];
}

// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector mixed_mult_logit_term_grad
  (arma::mat const &eta, arma::mat const &Sigma, 
   arma::uvec const &which_category, arma::vec const &weights, 
   arma::vec const &nodes, size_t const target_size = 128, 
   size_t const n_rep = 1, bool const use_adaptive = false){
  simple_mem_stack<double> mem;
  mixed_mult_logit_term<true> logit_term(eta, Sigma, which_category);
  outer_prod_problem outer_term(Sigma.n_cols);
  
  std::vector<ghq_problem const *> const prob_dat{ &logit_term, &outer_term };
  combined_problem prob(prob_dat);
  
  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);
  
  std::vector<double> res;  
  if(use_adaptive){
    adaptive_problem prob_adap(prob, mem);
    
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob_adap, mem, target_size);
    }
    
  } else 
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob, mem, target_size);
    }
    
  // handle the derivatives w.r.t. Sigma
  size_t const fixef_shift{logit_term.n_out()};
  Rcpp::NumericVector out(fixef_shift + Sigma.n_cols * Sigma.n_cols);
  std::copy(res.begin(), res.begin() + fixef_shift, &out[0]);
  outer_term.d_Sig
    (&out[fixef_shift], res.data() + fixef_shift, res[0], Sigma);
    
  return out;
}

/// for the combination of two types of outcomes
// [[Rcpp::export(rng = false)]]
double mixed_mult_logit_n_probit
  (arma::mat const &eta, arma::mat const &Sigma, 
   arma::uvec const &which_category, double const probit_eta, 
   arma::vec const &z, arma::vec const &weights, 
   arma::vec const &nodes, size_t const target_size = 128, 
   size_t const n_rep = 1, bool const use_adaptive = false){
  simple_mem_stack<double> mem;
  mixed_mult_logit_term<false> logit_term(eta, Sigma, which_category);
  mixed_probit_term<false> probit_term(probit_eta, Sigma, z);
  outer_prod_problem outer_term(Sigma.n_cols);
  
  std::vector<ghq_problem const *> const prob_dat{ &logit_term, &probit_term };
  combined_problem prob(prob_dat);
  
  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);
  
  std::vector<double> res;  
  if(use_adaptive){
    adaptive_problem prob_adap(prob, mem);
    
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob_adap, mem, target_size);
    }
    
  } else 
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob, mem, target_size);
    }
  
  return res[0];
}

/// to test the member functions for the adaptive method
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector mixed_mult_logit_n_probit_grad
  (arma::mat const &eta, arma::mat const &Sigma, 
   arma::uvec const &which_category, double const probit_eta, 
   arma::vec const &z, arma::vec const &weights, 
   arma::vec const &nodes, size_t const target_size = 128, 
   size_t const n_rep = 1, bool const use_adaptive = false){
  simple_mem_stack<double> mem;
  mixed_mult_logit_term<true> logit_term(eta, Sigma, which_category);
  mixed_probit_term<true> probit_term(probit_eta, Sigma, z);
  outer_prod_problem outer_term(Sigma.n_cols);
  
  std::vector<ghq_problem const *> const prob_dat
    { &logit_term, &probit_term, &outer_term };
  combined_problem prob(prob_dat);
  
  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);
  
  std::vector<double> res;  
  if(use_adaptive){
    adaptive_problem prob_adap(prob, mem);
    
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob_adap, mem, target_size);
    }
    
  } else 
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob, mem, target_size);
    }
    
  // handle the derivatives w.r.t. Sigma
  size_t const fixef_shift{logit_term.n_out() + probit_term.n_out() - 1};
  Rcpp::NumericVector out(fixef_shift + Sigma.n_cols * Sigma.n_cols);
  std::copy(res.begin(), res.begin() + fixef_shift, &out[0]);
  outer_term.d_Sig
    (&out[fixef_shift], res.data() + fixef_shift, res[0], Sigma);
  
  return out;
}

// [[Rcpp::export("expected_survival_term", rng = false)]]
double expected_survival_term_to_R
  (arma::vec const &eta, arma::vec const &ws, 
   arma::mat const &M, arma::mat const &Sigma, arma::vec const &weights, 
   arma::vec const &nodes, size_t const target_size = 128, 
   size_t const n_rep = 1, bool const use_adaptive = false){
  simple_mem_stack<double> mem;
  expected_survival_term prob(eta, ws, M, Sigma);
  
  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);
  
  std::vector<double> res;  
  if(use_adaptive){
    adaptive_problem prob_adap(prob, mem);
    
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob_adap, mem, target_size);
    }
    
  } else 
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob, mem, target_size);
    }
    
  return res[0];
}

/// to test the member functions for the adaptive method
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector mixed_probit_log_integrand
  (arma::vec const &point, double const probit_eta, 
   arma::vec const &z, arma::mat const &Sigma, unsigned const ders){
  simple_mem_stack<double> mem;
  mixed_probit_term prob(probit_eta, Sigma, z);
  
  if(ders == 0){
    return { prob.log_integrand(point.memptr(), mem) };
  } else if(ders == 1){
    Rcpp::NumericVector out(prob.n_vars() + 1);
    out[0] = prob.log_integrand_grad(point.memptr(), &out[1], mem);
    return out;
  } 
  
  Rcpp::NumericVector out(prob.n_vars() * prob.n_vars());
  prob.log_integrand_hess(point.memptr(), &out[0], mem);
  return out;
}

/// to test the member functions for the adaptive method
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector expected_survival_term_log_integrand
  (arma::vec const &point, arma::vec const &eta, arma::vec const &weights, 
   arma::mat const &M, arma::mat const &Sigma, unsigned const ders){
  simple_mem_stack<double> mem;
  expected_survival_term prob(eta, weights, M, Sigma);
  
  if(ders == 0){
    return { prob.log_integrand(point.memptr(), mem) };
  } else if(ders == 1){
    Rcpp::NumericVector out(prob.n_vars() + 1);
    out[0] = prob.log_integrand_grad(point.memptr(), &out[1], mem);
    return out;
  }
  
  Rcpp::NumericVector out(prob.n_vars() * prob.n_vars());
  prob.log_integrand_hess(point.memptr(), &out[0], mem);
  return out;
}

/// to test the member functions for the adaptive method
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector mixed_mult_logit_term_log_integrand
  (arma::vec const &point, arma::mat const &eta, arma::mat const &Sigma, 
   arma::uvec const &which_category, unsigned const ders){
  simple_mem_stack<double> mem;
  mixed_mult_logit_term prob(eta, Sigma, which_category);
  
  if(ders == 0){
    return { prob.log_integrand(point.memptr(), mem) };
  } else if(ders == 1){
    Rcpp::NumericVector out(prob.n_vars() + 1);
    out[0] = prob.log_integrand_grad(point.memptr(), &out[1], mem);
    return out;
  } 
  
  Rcpp::NumericVector out(prob.n_vars() * prob.n_vars());
  prob.log_integrand_hess(point.memptr(), &out[0], mem);
  return out;
}

/// to test that the combine class works
// [[Rcpp::export(rng = false)]]
Rcpp::NumericVector mixed_mult_logit_term_grad_comb_test
  (arma::mat const &eta_one, arma::mat const &eta_two, arma::mat const &Sigma, 
   arma::uvec const &which_category_one, arma::uvec const &which_category_two, 
   arma::vec const &weights, arma::vec const &nodes, 
   size_t const target_size = 128, size_t const n_rep = 1, 
   bool const use_adaptive = false){
  simple_mem_stack<double> mem;
  mixed_mult_logit_term<true> p1(eta_one, Sigma, which_category_one), 
                              p2(eta_two, Sigma, which_category_two);
  outer_prod_problem outer_term(Sigma.n_cols);
  
  std::vector<ghq_problem const *> const prob_dat{ &p1, &p2, &outer_term };
  combined_problem prob(prob_dat);
  
  auto ghq_data_pass = vecs_to_ghq_data(weights, nodes);
  
  std::vector<double> res;  
  if(use_adaptive){
    adaptive_problem prob_adap(prob, mem);
    
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob_adap, mem, target_size);
    }
    
  } else 
    for(size_t i = 0; i < n_rep; ++i){
      mem.reset();
      res = ghq(ghq_data_pass, prob, mem, target_size);
    }
    
  // handle the derivatives w.r.t. Sigma
  size_t const fixef_shift{p1.n_out() + p2.n_out() - 1};
  Rcpp::NumericVector out(fixef_shift + Sigma.n_cols * Sigma.n_cols);
  std::copy(res.begin(), res.begin() + fixef_shift, &out[0]);
  outer_term.d_Sig
    (&out[fixef_shift], res.data() + fixef_shift, res[0], Sigma);
    
  return out;
}

/***R
# check with the simple moment generating function of the multivariate normal 
# distribution as it serves as an OK example as the integrand is cheap to 
# evaluate. Thus, we can look at the computation time mainly of the quadrature 
# code. We also know the true value
set.seed(1)
n <- 3L
Sigma <- drop(rWishart(1, 2 * n, diag(1/n, n))) |> as.matrix()
mu <- rnorm(n)
tval <- runif(n, -1)

n_points <- 10L
ghq_dat <- fastGHQuad::gaussHermiteData(n_points)
truth <- exp(tval %*% mu + tval %*% Sigma %*% tval / 2) |> drop()

do_comp <- \(target_size, n_rep = 1L)
  mult_norm_moment(
    t = tval, mu = mu, Sigma = Sigma, weights = ghq_dat$w, nodes = ghq_dat$x,
    target_size = target_size, n_rep = n_rep)

all.equal(do_comp(1L), truth)
all.equal(do_comp(n_points), truth)
all.equal(do_comp(n_points^2), truth)

# time it takes to do the computation 1000 times
bench::mark(
  GHQ1 = do_comp(target_size = n_points, n_rep = 1000L),
  GHQ2 = do_comp(target_size = n_points^2, n_rep = 1000L),
  GHQ2 = do_comp(target_size = n_points^3, n_rep = 1000L))

# mixed multinomial logit example
set.seed(1)
n_obs <- 3L
n <- 3L
Sigma <- drop(rWishart(1, 2 * n, diag(1/n, n))) |> as.matrix()
eta <- runif(n_obs * n, -1) |> matrix(nrow = n)

# sample the outcome
u <- mvtnorm::rmvnorm(1, sigma = Sigma) |> drop()
lp <- eta + u
p_hats <- rbind(1, exp(lp)) / rep(1 + colSums(exp(lp)), each = n + 1L)
which_cat <- apply(p_hats, 2L, \(x) sample.int(n + 1L, 1L, prob = x))
rm(u, lp, p_hats)

# make a brute force Monte Carlo estimate
brute_ests <- apply(mvtnorm::rmvnorm(1e5, sigma = Sigma), 1L, \(u){
  exp_lp <- exp(eta + u)
  denom <- 1 + colSums(exp_lp)
  num <- mapply(
    \(i, j) if(i == 1L) 1 else exp_lp[i - 1L, j], i = which_cat,
    j = 1:NCOL(eta))
  prod(num / denom)
})
se <- sd(brute_ests) / sqrt(length(brute_ests))
brute_est <- mean(brute_ests)

# use the C++ function
n_points <- 5L
ghq_dat <- fastGHQuad::gaussHermiteData(n_points)

do_comp <- \(target_size, n_rep = 1L, use_adaptive = FALSE)
  mixed_mult_logit_term(
    eta = eta, Sigma = Sigma, 
    which_category = which_cat - 1L, # zero indexed
    weights = ghq_dat$w, nodes = ghq_dat$x, target_size = target_size, 
    n_rep = n_rep, use_adaptive = use_adaptive)

se / abs(brute_est) # ~ what we expect
all.equal(do_comp(n_points), brute_est)
all.equal(do_comp(n_points^2), brute_est)
all.equal(do_comp(n_points^3), brute_est)

# we can check that the member functions for adaptive GHQ are correct
log_integrand <- \(x){
  x <- crossprod(chol(Sigma), x) |> drop()
  exp_lp <- exp(eta + x)
  denom <- 1 + colSums(exp_lp)
  num <- mapply(
    \(i, j) if(i == 1L) 1 else exp_lp[i - 1L, j], i = which_cat,
    j = 1:NCOL(eta))
  sum(log(num / denom))
}
log_integrand_cpp <- \(ders)
  mixed_mult_logit_term_log_integrand(
    point = point, eta = eta, Sigma = Sigma, 
    which_category = which_cat - 1L, ders = ders)

point <- runif(n, -1)
all.equal(log_integrand(point), log_integrand_cpp(0))

f <- log_integrand_cpp(1)[1]
g <- log_integrand_cpp(1)[-1]
all.equal(log_integrand(point), f)
all.equal(numDeriv::grad(log_integrand, point), g)

all.equal(numDeriv::hessian(log_integrand, point) |> c(), log_integrand_cpp(2L))

# use the adaptive version
se / abs(brute_est) # ~ what we expect
all.equal(do_comp(n_points, use_adaptive = TRUE), brute_est)
all.equal(do_comp(n_points^2, use_adaptive = TRUE), brute_est)
all.equal(do_comp(n_points^3, use_adaptive = TRUE), 
          brute_est)

# time it takes to do the computation 1000 times
bench::mark(
  GHQ1 = do_comp(
    target_size = n_points, use_adaptive = FALSE, n_rep = 1000L),
  GHQ2 = do_comp(
    target_size = n_points^2, use_adaptive = FALSE, n_rep = 1000L),
  GHQ3 = do_comp(
    target_size = n_points^3, use_adaptive = FALSE ,n_rep = 1000L),
  `GHQ1 adaptive` = do_comp(
    target_size = n_points, use_adaptive = TRUE, n_rep = 1000L),
  `GHQ2 adaptive` = do_comp(
    target_size = n_points^2, use_adaptive = TRUE, n_rep = 1000L),
  `GHQ3 adaptive` = do_comp(
    target_size = n_points^3, use_adaptive = TRUE ,n_rep = 1000L),
  check = FALSE)

# compute the gradient
num_grad <- numDeriv::grad(
  \(par){
    e <- par[seq_along(eta)] |> matrix(nrow = NROW(eta))
    S <- matrix(nrow = NROW(Sigma), ncol = NCOL(Sigma))
    S[upper.tri(S, TRUE)] <- par[-seq_along(eta)]
    S[lower.tri(S)] <- t(S)[lower.tri(S)]
    
    mixed_mult_logit_term(
      eta = e, Sigma = S, 
      which_category = which_cat - 1L, # zero indexed
      weights = ghq_dat$w, nodes = ghq_dat$x, target_size = n_points^2, 
      n_rep = 1L, use_adaptive = TRUE)
  }, c(eta, Sigma[upper.tri(Sigma, TRUE)]))

do_comp_grad <- \(target_size, n_rep = 1L, use_adaptive = FALSE, 
                  use_comb = FALSE){
  res <- if(use_comb)
    mixed_mult_logit_term_grad_comb_test(
      eta_one = eta[,  1, drop = FALSE], which_category_one = which_cat[ 1] - 1L,
      eta_two = eta[, -1, drop = FALSE], which_category_two = which_cat[-1] - 1L,
      Sigma = Sigma, weights = ghq_dat$w, nodes = ghq_dat$x, 
      target_size = target_size, n_rep = n_rep, use_adaptive = use_adaptive)
  else
    mixed_mult_logit_term_grad(
      eta = eta, Sigma = Sigma, 
      which_category = which_cat - 1L, # zero indexed
      weights = ghq_dat$w, nodes = ghq_dat$x, target_size = target_size, 
      n_rep = n_rep, use_adaptive = use_adaptive)
  
  d_Sig <- tail(res, NCOL(Sigma)^2) |> matrix(nrow = NCOL(Sigma))
  d_Sig[upper.tri(d_Sig)] <- 2 * d_Sig[upper.tri(d_Sig)]
  c(head(res, -NCOL(Sigma)^2), d_Sig[upper.tri(d_Sig, TRUE)])
}
  
grad_cpp <- do_comp_grad(n_points, use_adaptive = TRUE)

# the first term is the integral
all.equal(do_comp(n_points, use_adaptive = TRUE), grad_cpp[1])
# the last are the derivatives w.r.t. eta
all.equal(num_grad, grad_cpp[-1])

# also work if we use the combine method
grad_cpp_comb <- do_comp_grad(n_points, use_adaptive = TRUE, use_comb = TRUE)
all.equal(grad_cpp, grad_cpp_comb)

# check the computation time of 1000 evaluations
bench::mark(
 `GHQ1 adaptive` = do_comp_grad(
   target_size = n_points, use_adaptive = TRUE, n_rep = 1000L),
`GHQ2 adaptive` = do_comp_grad(
  target_size = n_points^2, use_adaptive = TRUE, n_rep = 1000L), 
`GHQ2 adaptive combined` = 
  do_comp_grad(n_points^2, use_adaptive = TRUE, use_comb = TRUE, 
               n_rep = 1000L))

# mixed multinomial logit with the probit factor example
set.seed(1)
n_obs <- 2L
n <- 3L
Sigma <- drop(rWishart(1, 2 * n, diag(1/n, n))) |> as.matrix()
eta <- runif(n_obs * n, -1) |> matrix(nrow = n)
z <- runif(n, -1)
probit_eta <- runif(1, -1)

# sample the outcome
u <- mvtnorm::rmvnorm(1, sigma = Sigma) |> drop()
lp <- eta + u
p_hats <- rbind(1, exp(lp)) / rep(1 + colSums(exp(lp)), each = n + 1L)
which_cat <- apply(p_hats, 2L, \(x) sample.int(n + 1L, 1L, prob = x))

sign_probit <- (pnorm(probit_eta + z %*% u) > runif(1)) |> drop()
if(!sign_probit){
  z <- -z
  probit_eta <- -probit_eta
}
rm(u, lp, p_hats, sign_probit)

# make a brute force Monte Carlo estimate
brute_ests <- apply(mvtnorm::rmvnorm(1e5, sigma = Sigma), 1L, \(u){
  exp_lp <- exp(eta + u)
  denom <- 1 + colSums(exp_lp)
  num <- mapply(
    \(i, j) if(i == 1L) 1 else exp_lp[i - 1L, j], i = which_cat,
    j = 1:NCOL(eta))

  prod(num / denom) * pnorm(probit_eta + z %*% u)
})
se <- sd(brute_ests) / sqrt(length(brute_ests))
brute_est <- mean(brute_ests)

# use the C++ function
n_points <- 5L
ghq_dat <- fastGHQuad::gaussHermiteData(n_points)

do_comp <- \(target_size, n_rep = 1L, use_adaptive = FALSE)
  mixed_mult_logit_n_probit(
    eta = eta, Sigma = Sigma, 
    which_category = which_cat - 1L, # zero indexed
    probit_eta = probit_eta, z = z,
    weights = ghq_dat$w, nodes = ghq_dat$x, target_size = target_size, 
    n_rep = n_rep, use_adaptive = use_adaptive)

se / abs(brute_est) # ~ what we expect
all.equal(do_comp(n_points), brute_est)

# we can check that the member functions for adaptive GHQ are correct
log_integrand <- \(x){
  x <- crossprod(chol(Sigma), x) |> drop()
  pnorm(probit_eta + z %*% x, log.p = TRUE) |> drop()
}
log_integrand_cpp <- \(ders)
  mixed_probit_log_integrand(
    point = point, probit_eta = probit_eta, Sigma = Sigma, 
    z = z, ders = ders)

point <- runif(n, -1)
all.equal(log_integrand(point), log_integrand_cpp(0))

f <- log_integrand_cpp(1)[1]
g <- log_integrand_cpp(1)[-1]
all.equal(log_integrand(point), f)
all.equal(numDeriv::grad(log_integrand, point), g)

all.equal(numDeriv::hessian(log_integrand, point) |> c(), log_integrand_cpp(2L))

# check the quadrature precision with the adaptive version
se / abs(brute_est) # ~ what we expect
all.equal(do_comp(n_points, use_adaptive = TRUE), brute_est)

# compute the gradient numerically
num_grad <- numDeriv::grad(
  \(par){
    e <- par[seq_along(eta)] |> matrix(nrow = NROW(eta))
    par <- par[-seq_along(eta)]
    e_prob <- par[1]
    par <- par[-1]
    z_pass <- par[seq_along(z)]

    S <- matrix(nrow = NROW(Sigma), ncol = NCOL(Sigma))
    S[upper.tri(S, TRUE)] <- par[-seq_along(z_pass)]
    S[lower.tri(S)] <- t(S)[lower.tri(S)]

    mixed_mult_logit_n_probit(
      eta = e, Sigma = S,
      which_category = which_cat - 1L, # zero indexed
      probit_eta = e_prob, z = z_pass,
      weights = ghq_dat$w, nodes = ghq_dat$x, target_size = n_points^2,
      n_rep = 1L, use_adaptive = TRUE)[1]
  }, c(eta, probit_eta, z, Sigma[upper.tri(Sigma, TRUE)]))

# use the C++ implementation
do_comp_grad <- \(target_size, n_rep = 1L, use_adaptive = FALSE){
  res <- mixed_mult_logit_n_probit_grad(
    eta = eta, Sigma = Sigma,
    which_category = which_cat - 1L, # zero indexed
    probit_eta = probit_eta, z = z,
    weights = ghq_dat$w, nodes = ghq_dat$x, target_size = target_size,
    n_rep = n_rep, use_adaptive = use_adaptive)

  d_Sig <- tail(res, NCOL(Sigma)^2) |> matrix(nrow = NCOL(Sigma))
  d_Sig[upper.tri(d_Sig)] <- 2 * d_Sig[upper.tri(d_Sig)]
  c(head(res, -NCOL(Sigma)^2), d_Sig[upper.tri(d_Sig, TRUE)])
}

est <- do_comp_grad(n_points, use_adaptive = TRUE)
all.equal(brute_est, est[1])
all.equal(num_grad, est[-1])

# perform a benchmark on both the integral it self and the gradient for 1000 
# evaluations
bench::mark(
  Func = do_comp(target_size = n_points^2, n_rep = 1000), 
  `Func adaptive` = do_comp(target_size = n_points^2, n_rep = 1000, 
                            use_adaptive = TRUE), 
  Grad = do_comp_grad(target_size = n_points^2, n_rep = 1000), 
  `Grad adaptive` = do_comp_grad(
    target_size = n_points^2, n_rep = 1000, use_adaptive = TRUE), 
  check = FALSE)

# example for the expected survival function
etas <- c(-0.6381, -0.6343, -0.6285, -0.6227, -0.6193, -0.6212, -0.6313, -0.6515, -0.6827, -0.724, -0.7723, -0.8229, -0.8699, -0.9073, -0.93)
ws <- c(0.0473878636519986, 0.10842748431263, 0.165122315533235, 0.215065333801771, 0.256205263034596, 0.286856653590245, 0.305764321437458, 0.312154084712178, 0.305764321437458, 0.286856653590245, 0.256205263034596, 0.215065333801771, 0.165122315533235, 0.10842748431263, 0.0473878636519986)
M <- structure(c(0.3115, 0.312, 0.3128, 0.3141, 0.3156, 0.3173, 0.3192, 0.3212, 0.3232, 0.3251, 0.3269, 0.3284, 0.3296, 0.3305, 0.331, -0.1315, -0.1378, -0.1488, -0.1641, -0.1831, -0.205, -0.2288, -0.2537, -0.2786, -0.3024, -0.3243, -0.3433, -0.3586, -0.3696, -0.3758, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -1.2253, -1.1941, -1.1392, -1.0629, -0.9683, -0.8593, -0.7404, -0.6164, -0.4924, -0.3734, -0.2644, -0.1699, -0.0936, -0.0387, -0.0074, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), .Dim = c(15L, 5L))
V <- structure(c(0.35, 0.08, -0.05, 0.01, 0, 0.08, 1.92, -0.24, -0.04, 0, -0.05, -0.24, 0.32, 0.09, 0, 0.01, -0.04, 0.09, 0.12, 0, 0, 0, 0, 0, 0.04), .Dim = c(5L, 5L))
n <- NCOL(V)

# make a brute force Monte Carlo estimate
brute_ests <- apply(mvtnorm::rmvnorm(1e6, sigma = V), 1L, \(u){
  x <- M %*% u |> drop()
  -sum(ws * exp(x + etas)) |> exp()
})
se <- sd(brute_ests) / sqrt(length(brute_ests))
brute_est <- mean(brute_ests)

# use the C++ function
n_points <- 2L
ghq_dat <- fastGHQuad::gaussHermiteData(n_points)

do_comp <- \(target_size, n_rep = 1L, use_adaptive = FALSE)
  expected_survival_term(
    eta = etas, ws = ws, M = M, Sigma = V,
    weights = ghq_dat$w, nodes = ghq_dat$x, target_size = target_size, 
    n_rep = n_rep, use_adaptive = use_adaptive)

se / abs(brute_est) # ~ what we expect
all.equal(do_comp(n_points), brute_est)

# we can check that the member functions for adaptive GHQ are correct
log_integrand <- \(x){
  x <- crossprod(chol(V), x)
  x <- M %*% x |> drop()
  -sum(ws * exp(x + etas))
}
log_integrand_cpp <- \(ders)
  expected_survival_term_log_integrand(
    point = point, eta = etas, weights = ws, M = M, Sigma = V, ders = ders)

point <- runif(n, -1)
all.equal(log_integrand(point), log_integrand_cpp(0))

f <- log_integrand_cpp(1)[1]
g <- log_integrand_cpp(1)[-1]
all.equal(log_integrand(point), f)
all.equal(numDeriv::grad(log_integrand, point), g)

all.equal(numDeriv::hessian(log_integrand, point) |> c(), log_integrand_cpp(2L))

# check the quadrature precision with the adaptive version
se / abs(brute_est) # ~ what we expect
all.equal(do_comp(n_points, use_adaptive = TRUE), brute_est)

# time it takes to do the computation 1000 times
bench::mark(
  GHQ1 = do_comp(
    target_size = n_points, use_adaptive = FALSE, n_rep = 1000L),
  GHQ5 = do_comp(
    target_size = n_points^5, use_adaptive = FALSE ,n_rep = 1000L),
  `GHQ1 adaptive` = do_comp(
    target_size = n_points, use_adaptive = TRUE, n_rep = 1000L),
  `GHQ5 adaptive` = do_comp(
    target_size = n_points^5, use_adaptive = TRUE ,n_rep = 1000L),
  check = FALSE)
*/
