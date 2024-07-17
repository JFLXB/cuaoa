/* Copyright 2024 Jonas Blenninger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuaoa/optimize.hpp"
#include "cuaoa/cuaoa.hpp"
#include "cuaoa/handle.hpp"
#include <cstdint>
#include <cstdio>
#include <lbfgs.h>

typedef struct {
  CUAOAHandle *handle;
  size_t numberOfNodes;
  size_t depth;
  size_t *polynomialKeys;
  double *polynomialValues;
  size_t polynomialSize;
  size_t blockSize;
  const char *rxmethod;
  double *fx_log;
  double *x_log;
  bool should_log;
  int64_t iteration;
  int64_t n_evaluations;
} cuaoa_instance_t;

lbfgsfloatval_t evaluate(void *instance, const lbfgsfloatval_t *x,
                         lbfgsfloatval_t *g, const int n,
                         const lbfgsfloatval_t step) {
  double expectationValue;
  cuaoa_instance_t *cuaoa_instance = (cuaoa_instance_t *)instance;
  const int stride = n >> 1;

  inner_gradients_cuaoa(cuaoa_instance->handle, cuaoa_instance->numberOfNodes,
                        cuaoa_instance->depth, cuaoa_instance->polynomialKeys,
                        cuaoa_instance->polynomialValues,
                        cuaoa_instance->polynomialSize, x, x + stride,
                        cuaoa_instance->blockSize, g, g + stride,
                        &expectationValue, cuaoa_instance->rxmethod);
  return (lbfgsfloatval_t)expectationValue;
}

static int progress(void *instance, const lbfgsfloatval_t *x,
                    const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                    const lbfgsfloatval_t step, int n, int k, int ls) {
  cuaoa_instance_t *cuaoa_instance = (cuaoa_instance_t *)instance;
  cuaoa_instance->iteration = k;
  cuaoa_instance->n_evaluations += ls;

  if (cuaoa_instance->should_log) {
    cuaoa_instance->fx_log[k - 1] = (double)fx;
    for (int i = 0; i < n; ++i) {
      cuaoa_instance->x_log[(k - 1) * n + i] = (double)x[i];
    }
  }
  return 0;
}

void initparams(lbfgs_parameter_t *param, int m, double epsilon, int past,
                double delta, int max_iterations, int linesearch,
                int max_linesearch, double min_step, double max_step,
                double ftol, double wolfe, double gtol, double xtol,
                double orthantwise_c, int orthantwise_start,
                int orthantwise_end) {
  // int 	m
  //  	The number of corrections to approximate the inverse hessian matrix.
  // lbfgsfloatval_t 	epsilon
  //  	Epsilon for convergence test.
  // int 	past
  //  	Distance for delta-based convergence test.
  // lbfgsfloatval_t 	delta
  //  	Delta for convergence test.
  // int 	max_iterations
  //  	The maximum number of iterations.
  // int 	linesearch
  //  	The line search algorithm.
  // int 	max_linesearch
  //  	The maximum number of trials for the line search.
  // lbfgsfloatval_t 	min_step
  //  	The minimum step of the line search routine.
  // lbfgsfloatval_t 	max_step
  //  	The maximum step of the line search.
  // lbfgsfloatval_t 	ftol
  //  	A parameter to control the accuracy of the line search routine.
  // lbfgsfloatval_t 	wolfe
  //  	A coefficient for the Wolfe condition.
  // lbfgsfloatval_t 	gtol
  //  	A parameter to control the accuracy of the line search routine.
  // lbfgsfloatval_t 	xtol
  //  	The machine precision for floating-point values.
  // lbfgsfloatval_t 	orthantwise_c
  //  	Coeefficient for the L1 norm of variables.
  // int 	orthantwise_start
  //  	Start index for computing L1 norm of the variables.
  // int 	orthantwise_end
  //  	End index for computing L1 norm of the variables.
  lbfgs_parameter_init(param);
  param->m = m;
  param->epsilon = (lbfgsfloatval_t)epsilon;
  param->past = past;
  param->delta = (lbfgsfloatval_t)delta;
  param->max_iterations = max_iterations;
  param->linesearch = linesearch;
  param->max_linesearch = max_linesearch;
  param->min_step = (lbfgsfloatval_t)min_step;
  param->max_step = (lbfgsfloatval_t)max_step;
  param->ftol = (lbfgsfloatval_t)ftol;
  param->wolfe = (lbfgsfloatval_t)wolfe;
  param->gtol = (lbfgsfloatval_t)gtol;
  param->xtol = (lbfgsfloatval_t)xtol;
  param->orthantwise_c = (lbfgsfloatval_t)orthantwise_c;
  param->orthantwise_start = orthantwise_start;
  param->orthantwise_end = orthantwise_end;
}

CuaoaOptimized optimize_cuaoa(
    CUAOAHandle *handle, size_t numberOfNodes, size_t depth,
    size_t *polynomialKeys, double *polynomialValues, size_t polynomialSize,
    double *betas, double *gammas, int m, double epsilon, int past,
    double delta, int max_iterations, int linesearch, int max_linesearch,
    double min_step, double max_step, double ftol, double wolfe, double gtol,
    double xtol, double orthantwise_c, int orthantwise_start,
    int orthantwise_end, bool log, size_t blockSize, const char *rxmethod) {
  CuaoaOptimized result;
  result.betas = new double[depth];
  result.gammas = new double[depth];

  double *fx_log;
  double *x_log;
  bool should_log;
  if (log && (max_iterations != 0)) {
    fx_log = new double[max_iterations];
    x_log = new double[max_iterations * 2 * depth];
    should_log = true;
  } else {
    fx_log = nullptr;
    x_log = nullptr;
    should_log = false;
  }
  cuaoa_instance_t instance = {handle,         numberOfNodes,    depth,
                               polynomialKeys, polynomialValues, polynomialSize,
                               blockSize,      rxmethod,         fx_log,
                               x_log,          should_log};

  lbfgsfloatval_t fx;
  const size_t N = depth << 1;

  lbfgsfloatval_t *x = lbfgs_malloc(N);
  for (size_t i = 0; i < depth; i++) {
    x[i] = betas[i];
    x[i + depth] = gammas[i];
  }

  lbfgs_parameter_t param;
  initparams(&param, m, epsilon, past, delta, max_iterations, linesearch,
             max_linesearch, min_step, max_step, ftol, wolfe, gtol, xtol,
             orthantwise_c, orthantwise_start, orthantwise_end);

  int ret;
  ret = lbfgs(N, x, &fx, evaluate, progress, &instance, &param);
  result.status = ret;

  for (size_t i = 0; i < depth; i++) {
    result.betas[i] = x[i];
    result.gammas[i] = x[i + depth];
  }

  result.iteration = instance.iteration;
  result.n_evals = instance.n_evaluations;
  result.fx_log = instance.fx_log;
  result.x_log = instance.x_log;
  result.has_log = should_log;

  lbfgs_free(x);
  return result;
}
