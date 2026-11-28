--
-- luatorch / lib.lua
--
-- FFI declarations for the native shim and the loader for libluatorch.so.
-- This module is intentionally low-level: it exposes raw pointers and
-- status codes.  Use `torch` (lua/torch/init.lua) for the friendly API.
--
local ffi = require("ffi")

ffi.cdef[[
typedef struct lt_tensor lt_tensor;

/* lifecycle */
const char* lt_version(void);
const char* lt_last_error(void);
void        lt_free(lt_tensor* t);
int64_t     lt_free_counted(lt_tensor* t);

/* creation */
lt_tensor* lt_zeros(int64_t ndim, const int64_t* sizes, int dtype);
lt_tensor* lt_ones(int64_t ndim, const int64_t* sizes, int dtype);
lt_tensor* lt_empty(int64_t ndim, const int64_t* sizes, int dtype);
lt_tensor* lt_full(int64_t ndim, const int64_t* sizes, double value, int dtype);
lt_tensor* lt_rand(int64_t ndim, const int64_t* sizes, int dtype);
lt_tensor* lt_randn(int64_t ndim, const int64_t* sizes, int dtype);
lt_tensor* lt_eye(int64_t n, int dtype);
lt_tensor* lt_arange(double start, double stop, double step, int dtype);
lt_tensor* lt_linspace(double start, double stop, int64_t steps, int dtype);
lt_tensor* lt_randperm(int64_t n, int dtype);
lt_tensor* lt_from_data(const void* data, int64_t ndim, const int64_t* sizes, int dtype);
lt_tensor* lt_from_blob(const void* data, int64_t ndim, const int64_t* sizes, int dtype);
void       lt_manual_seed(uint64_t seed);

/* metadata */
int64_t     lt_dim(const lt_tensor* t);
int64_t     lt_numel(const lt_tensor* t);
int         lt_dtype(const lt_tensor* t);
int         lt_device(const lt_tensor* t);
int64_t     lt_size(const lt_tensor* t, int64_t dim);
int         lt_sizes(const lt_tensor* t, int64_t* out);
int         lt_is_contiguous(const lt_tensor* t);
int         lt_is_floating_point(const lt_tensor* t);
int         lt_requires_grad(const lt_tensor* t);
int64_t     lt_element_size(const lt_tensor* t);
int64_t     lt_nbytes(const lt_tensor* t);
const void* lt_data_ptr(const lt_tensor* t);

/* copies / dtypes / autograd */
lt_tensor* lt_contiguous(const lt_tensor* t);
lt_tensor* lt_clone(const lt_tensor* t);
lt_tensor* lt_to_dtype(const lt_tensor* t, int dtype);
lt_tensor* lt_detach(const lt_tensor* t);
int        lt_copy_(lt_tensor* dst, const lt_tensor* src);
int        lt_set_requires_grad(lt_tensor* t, int requires_grad);
int        lt_has_grad(const lt_tensor* t);
lt_tensor* lt_grad(const lt_tensor* t);
int        lt_backward(lt_tensor* t, const lt_tensor* gradient);

/* scalar access */
int lt_item_f64(const lt_tensor* t, int64_t ndim, const int64_t* index, double* out);
int lt_item_i64(const lt_tensor* t, int64_t ndim, const int64_t* index, int64_t* out);
int lt_item_bool(const lt_tensor* t, int64_t ndim, const int64_t* index, int* out);
int lt_set_item_f64(lt_tensor* t, int64_t ndim, const int64_t* index, double value);
int lt_set_item_i64(lt_tensor* t, int64_t ndim, const int64_t* index, int64_t value);

/* binary */
lt_tensor* lt_add(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_sub(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_mul(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_div(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_pow(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_remainder(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_fmod(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_atan2(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_copysign(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_floor_divide(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_heaviside(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_igamma(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_igammac(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_maximum(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_minimum(lt_tensor* a, lt_tensor* b, int inplace);
lt_tensor* lt_add_scalar(lt_tensor* a, double v, int inplace);
lt_tensor* lt_sub_scalar(lt_tensor* a, double v, int inplace);
lt_tensor* lt_mul_scalar(lt_tensor* a, double v, int inplace);
lt_tensor* lt_div_scalar(lt_tensor* a, double v, int inplace);
lt_tensor* lt_pow_scalar(lt_tensor* a, double v, int inplace);
lt_tensor* lt_remainder_scalar(lt_tensor* a, double v, int inplace);

/* comparisons */
lt_tensor* lt_eq(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_ne(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_lt(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_le(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_gt(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_ge(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_eq_scalar(lt_tensor* a, double v);
lt_tensor* lt_ne_scalar(lt_tensor* a, double v);
lt_tensor* lt_lt_scalar(lt_tensor* a, double v);
lt_tensor* lt_le_scalar(lt_tensor* a, double v);
lt_tensor* lt_gt_scalar(lt_tensor* a, double v);
lt_tensor* lt_ge_scalar(lt_tensor* a, double v);
lt_tensor* lt_logical_and(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_logical_or(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_logical_not(lt_tensor* a);

/* unary */
lt_tensor* lt_neg(lt_tensor* a, int inplace);
lt_tensor* lt_abs(lt_tensor* a, int inplace);
lt_tensor* lt_exp(lt_tensor* a, int inplace);
lt_tensor* lt_expm1(lt_tensor* a, int inplace);
lt_tensor* lt_log(lt_tensor* a, int inplace);
lt_tensor* lt_log1p(lt_tensor* a, int inplace);
lt_tensor* lt_log2(lt_tensor* a, int inplace);
lt_tensor* lt_log10(lt_tensor* a, int inplace);
lt_tensor* lt_erf(lt_tensor* a, int inplace);
lt_tensor* lt_erfc(lt_tensor* a, int inplace);
lt_tensor* lt_erfinv(lt_tensor* a, int inplace);
lt_tensor* lt_i0(lt_tensor* a, int inplace);
lt_tensor* lt_lgamma(lt_tensor* a, int inplace);
lt_tensor* lt_digamma(lt_tensor* a, int inplace);
lt_tensor* lt_ndtr(lt_tensor* a, int inplace);
lt_tensor* lt_log_ndtr(lt_tensor* a, int inplace);
lt_tensor* lt_ndtri(lt_tensor* a, int inplace);
lt_tensor* lt_i0e(lt_tensor* a, int inplace);
lt_tensor* lt_i1(lt_tensor* a, int inplace);
lt_tensor* lt_i1e(lt_tensor* a, int inplace);
lt_tensor* lt_bessel_j0(lt_tensor* a, int inplace);
lt_tensor* lt_polygamma(lt_tensor* a, int64_t order, int inplace);
lt_tensor* lt_mvlgamma(lt_tensor* a, int64_t order, int inplace);
lt_tensor* lt_sinc(lt_tensor* a, int inplace);
lt_tensor* lt_sqrt(lt_tensor* a, int inplace);
lt_tensor* lt_rsqrt(lt_tensor* a, int inplace);
lt_tensor* lt_sin(lt_tensor* a, int inplace);
lt_tensor* lt_cos(lt_tensor* a, int inplace);
lt_tensor* lt_tan(lt_tensor* a, int inplace);
lt_tensor* lt_asin(lt_tensor* a, int inplace);
lt_tensor* lt_acos(lt_tensor* a, int inplace);
lt_tensor* lt_atan(lt_tensor* a, int inplace);
lt_tensor* lt_sinh(lt_tensor* a, int inplace);
lt_tensor* lt_cosh(lt_tensor* a, int inplace);
lt_tensor* lt_tanh(lt_tensor* a, int inplace);
lt_tensor* lt_sigmoid(lt_tensor* a, int inplace);
lt_tensor* lt_relu(lt_tensor* a, int inplace);
lt_tensor* lt_floor(lt_tensor* a, int inplace);
lt_tensor* lt_ceil(lt_tensor* a, int inplace);
lt_tensor* lt_round(lt_tensor* a, int inplace);
lt_tensor* lt_trunc(lt_tensor* a, int inplace);
lt_tensor* lt_sign(lt_tensor* a, int inplace);
lt_tensor* lt_square(lt_tensor* a, int inplace);
lt_tensor* lt_reciprocal(lt_tensor* a, int inplace);
lt_tensor* lt_clamp(lt_tensor* a, double lo, double hi, int inplace);
lt_tensor* lt_fill(lt_tensor* a, double value, int inplace);

/* NaN / Inf predicates */
lt_tensor* lt_isnan(const lt_tensor* t);
lt_tensor* lt_isinf(const lt_tensor* t);
lt_tensor* lt_isfinite(const lt_tensor* t);

/* linalg */
lt_tensor* lt_matmul(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_mm(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_bmm(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_dot(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_einsum(lt_tensor* const* tensors, int64_t n, const char* equation);

/* neural-network primitives */
lt_tensor* lt_softmax(lt_tensor* a, int64_t dim);
lt_tensor* lt_fused_softmax(lt_tensor* a);
lt_tensor* lt_log_softmax(lt_tensor* a, int64_t dim);
lt_tensor* lt_fused_log_softmax(lt_tensor* a);
lt_tensor* lt_linalg_inv(lt_tensor* a);
int lt_linalg_inv_ex(lt_tensor* a, int check_errors,
                     lt_tensor** inverse, lt_tensor** info);
lt_tensor* lt_linalg_det(lt_tensor* a);
int lt_linalg_slogdet(lt_tensor* a, lt_tensor** sign, lt_tensor** logabsdet);
int lt_linalg_lstsq(lt_tensor* a, lt_tensor* b, const char* driver,
                    double rcond, int has_rcond,
                    lt_tensor** solution, lt_tensor** residuals,
                    lt_tensor** rank, lt_tensor** singular_values);
lt_tensor* lt_linalg_solve(lt_tensor* a, lt_tensor* b);
int lt_linalg_solve_ex(lt_tensor* a, lt_tensor* b, int left, int check_errors,
                       lt_tensor** result, lt_tensor** info);
lt_tensor* lt_linalg_solve_triangular(lt_tensor* a, lt_tensor* b,
                                      int upper, int left,
                                      int unitriangular);
lt_tensor* lt_linalg_cholesky(lt_tensor* a, int upper);
lt_tensor* lt_linalg_cholesky_inverse(lt_tensor* a, int upper);
lt_tensor* lt_linalg_cholesky_solve(lt_tensor* b, lt_tensor* factor,
                                    int upper);
int lt_linalg_cholesky_ex(lt_tensor* a, int upper, int check_errors,
                          lt_tensor** L, lt_tensor** info);
int lt_linalg_eigh(lt_tensor* a, const char* uplo,
                   lt_tensor** eigenvalues, lt_tensor** eigenvectors);
lt_tensor* lt_linalg_eigvalsh(lt_tensor* a, const char* uplo);
int lt_linalg_svd(lt_tensor* a, int full_matrices,
                  lt_tensor** U, lt_tensor** S, lt_tensor** Vh);
int lt_linalg_qr(lt_tensor* a, const char* mode, lt_tensor** Q, lt_tensor** R);
int lt_linalg_lu(lt_tensor* a, lt_tensor** P, lt_tensor** L, lt_tensor** U);
int lt_linalg_lu_factor(lt_tensor* a, lt_tensor** LU, lt_tensor** pivots);
int lt_linalg_lu_factor_ex(lt_tensor* a, int pivot, int check_errors,
                           lt_tensor** LU, lt_tensor** pivots,
                           lt_tensor** info);
lt_tensor* lt_linalg_lu_solve(lt_tensor* LU, lt_tensor* pivots,
                              lt_tensor* b, int left, int adjoint);
int lt_linalg_ldl_factor(lt_tensor* a, int hermitian,
                         lt_tensor** LD, lt_tensor** pivots);
int lt_linalg_ldl_factor_ex(lt_tensor* a, int hermitian, int check_errors,
                            lt_tensor** LD, lt_tensor** pivots,
                            lt_tensor** info);
lt_tensor* lt_linalg_ldl_solve(lt_tensor* LD, lt_tensor* pivots,
                               lt_tensor* b, int hermitian);
lt_tensor* lt_linalg_pinv(lt_tensor* a, double atol, double rtol,
                          int has_atol, int has_rtol);
lt_tensor* lt_linalg_matrix_exp(lt_tensor* a);
lt_tensor* lt_linalg_householder_product(lt_tensor* a, lt_tensor* tau);
lt_tensor* lt_mv(lt_tensor* a, lt_tensor* v);
lt_tensor* lt_outer(lt_tensor* a, lt_tensor* b);
lt_tensor* lt_addmm(lt_tensor* self, lt_tensor* a, lt_tensor* b,
                    double beta, double alpha);
lt_tensor* lt_baddbmm(lt_tensor* self, lt_tensor* a, lt_tensor* b,
                      double beta, double alpha);
lt_tensor* lt_logsumexp(lt_tensor* a, int64_t dim, int keepdim);
int lt_normal_(lt_tensor* a, double mean, double std);
int lt_exponential_(lt_tensor* a, double lambd);
int lt_bernoulli_(lt_tensor* a, double p);
lt_tensor* lt_multinomial(lt_tensor* a, int64_t num_samples, int replacement);
void lt_set_num_threads(int num_threads);
int lt_get_num_threads(void);
lt_tensor* lt_conv2d(
    lt_tensor* input, lt_tensor* weight, const lt_tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t groups);
lt_tensor* lt_conv_transpose2d(
    lt_tensor* input, lt_tensor* weight, const lt_tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t output_pad_h, int64_t output_pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t groups);
lt_tensor* lt_max_pool2d(
    lt_tensor* input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int ceil_mode);
lt_tensor* lt_avg_pool2d(
    lt_tensor* input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int ceil_mode, int count_include_pad);
lt_tensor* lt_embedding(lt_tensor* weight, lt_tensor* indices);
lt_tensor* lt_layer_norm(
    lt_tensor* input, const lt_tensor* weight, const lt_tensor* bias,
    int64_t ndim, const int64_t* normalized_shape, double eps);
lt_tensor* lt_cross_entropy(lt_tensor* input, lt_tensor* target, int64_t reduction);
lt_tensor* lt_fused_cross_entropy(lt_tensor* input, lt_tensor* target, int64_t reduction);
lt_tensor* lt_nll_loss(lt_tensor* input, lt_tensor* target, int64_t reduction);
lt_tensor* lt_binary_cross_entropy_with_logits(lt_tensor* input, lt_tensor* target, int64_t reduction);
int        lt_uniform_(lt_tensor* t, double low, double high);

/* shape */
lt_tensor* lt_reshape(lt_tensor* a, int64_t ndim, const int64_t* sizes);
lt_tensor* lt_view(lt_tensor* a, int64_t ndim, const int64_t* sizes);
lt_tensor* lt_transpose(lt_tensor* a, int64_t d0, int64_t d1);
lt_tensor* lt_t(lt_tensor* a);
lt_tensor* lt_permute(lt_tensor* a, int64_t ndim, const int64_t* dims);
lt_tensor* lt_squeeze(lt_tensor* a, int64_t dim);
lt_tensor* lt_squeeze_all(lt_tensor* a);
lt_tensor* lt_unsqueeze(lt_tensor* a, int64_t dim);
lt_tensor* lt_flatten(lt_tensor* a, int64_t start, int64_t end);
lt_tensor* lt_expand(lt_tensor* a, int64_t ndim, const int64_t* sizes);

/* indexing */
lt_tensor* lt_select(lt_tensor* a, int64_t dim, int64_t index);
lt_tensor* lt_narrow(lt_tensor* a, int64_t dim, int64_t start, int64_t length);
lt_tensor* lt_slice(lt_tensor* a, int64_t dim, int64_t start, int64_t stop, int64_t step);
lt_tensor* lt_flip(lt_tensor* a, int64_t dim);
lt_tensor* lt_cat(lt_tensor* const* tensors, int64_t n, int64_t dim);
lt_tensor* lt_stack(lt_tensor* const* tensors, int64_t n, int64_t dim);
lt_tensor* lt_where(lt_tensor* cond, lt_tensor* a, lt_tensor* b);
lt_tensor* lt_masked_fill(lt_tensor* a, lt_tensor* mask, double value);
lt_tensor* lt_masked_select(lt_tensor* a, lt_tensor* mask);
lt_tensor* lt_index_select(lt_tensor* a, int64_t dim, lt_tensor* indices);
int lt_index_select_batch(lt_tensor* a, int64_t dim,
                          lt_tensor* const* indices, int64_t count,
                          lt_tensor** outputs);
lt_tensor* lt_nonzero(lt_tensor* a);
lt_tensor* lt_gather(lt_tensor* a, int64_t dim, lt_tensor* index);
lt_tensor* lt_scatter(lt_tensor* a, int64_t dim, lt_tensor* index, lt_tensor* src);
int lt_sort(lt_tensor* a, int64_t dim, int descending,
            lt_tensor** values, lt_tensor** indices);
int lt_topk(lt_tensor* a, int64_t k, int64_t dim, int largest, int sorted,
            lt_tensor** values, lt_tensor** indices);
int lt_unique(lt_tensor* a, int sorted, int return_inverse, int return_counts,
              lt_tensor** values, lt_tensor** inverse, lt_tensor** counts);

/* reductions */
lt_tensor* lt_sum(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_mean(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_prod(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_max(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_min(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_argmax(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_argmin(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_var(lt_tensor* a, int64_t dim, int keepdim, int unbiased);
lt_tensor* lt_std(lt_tensor* a, int64_t dim, int keepdim, int unbiased);
lt_tensor* lt_norm(lt_tensor* a, double p, int64_t dim, int keepdim);
lt_tensor* lt_cumsum(lt_tensor* a, int64_t dim);
lt_tensor* lt_cumprod(lt_tensor* a, int64_t dim);
int lt_cummax(lt_tensor* a, int64_t dim, lt_tensor** values,
              lt_tensor** indices);
int lt_cummin(lt_tensor* a, int64_t dim, lt_tensor** values,
              lt_tensor** indices);
int lt_mode(lt_tensor* a, int64_t dim, int keepdim, lt_tensor** values,
            lt_tensor** indices);
int lt_nanmedian(lt_tensor* a, int64_t dim, int has_dim, int keepdim,
                 lt_tensor** values, lt_tensor** indices);
lt_tensor* lt_nanquantile(lt_tensor* a, double q, int64_t dim, int has_dim,
                          int keepdim);
lt_tensor* lt_all(lt_tensor* a, int64_t dim, int keepdim);
lt_tensor* lt_any(lt_tensor* a, int64_t dim, int keepdim);

/* fused optimizer updates */
int lt_adam_amsgrad_step(
    lt_tensor* parameter, const lt_tensor* gradient,
    lt_tensor* exp_avg, lt_tensor* exp_avg_sq, lt_tensor* max_exp_avg_sq,
    double lr, double beta1, double beta2, double eps,
    double weight_decay, int64_t step, int decoupled);
int lt_adam_step(
    lt_tensor* parameter, const lt_tensor* gradient,
    lt_tensor* exp_avg, lt_tensor* exp_avg_sq,
    double lr, double beta1, double beta2, double eps,
    double weight_decay, int64_t step, int decoupled);
int lt_sgd_step(
    lt_tensor* parameter, const lt_tensor* gradient,
    lt_tensor* momentum_buffer,
    double lr, double momentum, double dampening,
    double weight_decay, int nesterov, int first_step);

int lt_rmsprop_step(
    lt_tensor* parameter, const lt_tensor* gradient,
    lt_tensor* square_avg, lt_tensor* grad_avg, lt_tensor* momentum_buffer,
    double lr, double alpha, double eps, double weight_decay,
    double momentum, int centered);
int lt_adagrad_step(
    lt_tensor* parameter, const lt_tensor* gradient,
    lt_tensor* sum,
    double lr, double lr_decay, double weight_decay,
    double eps, int64_t step);

int lt_clip_grad_norm_(
    lt_tensor* const* gradients, int64_t count,
    double max_norm, double norm_type, double* out_norm);
int lt_clip_grad_value_(
    lt_tensor* const* gradients, int64_t count, double clip_value);

lt_tensor* lt_fused_apply(
    lt_tensor* const* inputs, int64_t n_inputs,
    void* forward_kernel, void* backward_kernel,
    const double* scalars, int64_t n_scalars);

lt_tensor* lt_fused_reduce_apply(
    lt_tensor* const* inputs, int64_t n_inputs,
    void* forward_kernel, void* backward_kernel,
    const double* scalars, int64_t n_scalars);

/* grad mode */
void       lt_set_grad_enabled(int enabled);
int        lt_is_grad_enabled(void);

/* serialization */
int        lt_save(const lt_tensor* t, const char* path);
lt_tensor* lt_load(const char* path);
]]

-- Locate the shared library.  Resolution order:
--   1. $LUATORCH_LIB
--   2. <repo>/build/libluatorch.so (normal development layout)
--   3. <repo>/libluatorch.so
--   4. system loader path
local source = debug.getinfo(1, "S").source
local file = source:sub(1, 1) == "@" and source:sub(2) or source
local here = file:match("^(.*)[/\\][^/\\]+$") or "."
local candidates = {}
local env = os.getenv("LUATORCH_LIB")
if env and #env > 0 then
  candidates[#candidates + 1] = env
end
candidates[#candidates + 1] = here .. "/../../build/libluatorch.so"
candidates[#candidates + 1] = here .. "/../../libluatorch.so"
candidates[#candidates + 1] = here .. "/libluatorch.so"
candidates[#candidates + 1] = "libluatorch.so"

local loaded, handle = nil, nil
local failures = {}
for _, path in ipairs(candidates) do
  local ok, result = pcall(ffi.load, path)
  if ok then
    loaded, handle = path, result
    break
  end
  failures[#failures + 1] = string.format("  %s: %s", path, tostring(result))
end

if not handle then
  error(
    "luatorch: could not load the native shim (libluatorch.so).\n" ..
    "Tried:\n" .. table.concat(failures, "\n") .. "\n" ..
    "Build it with scripts/build.sh or set LUATORCH_LIB to its path.",
    0)
end

local M = {
  ffi = ffi,
  lib = handle,
  path = loaded,
  version = ffi.string(handle.lt_version()),
  dtype_names = {
    [0] = "float32",
    [1] = "float64",
    [2] = "int64",
    [3] = "int32",
    [4] = "int16",
    [5] = "int8",
    [6] = "uint8",
    [7] = "bool",
    [8] = "float16",
  },
}

return M
