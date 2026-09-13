#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    W4A4,
};

bool is_27b_residual(std::int32_t output_rows, std::int32_t input_rows) {
    return output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
           (input_rows == Nvfp4Residual6144Geometry::kInputRows ||
            input_rows == Nvfp4Residual17408Geometry::kInputRows);
}

bool is_ornith_down(std::int32_t output_rows, std::int32_t input_rows) {
    return output_rows == Nvfp4MlpDown12288Geometry::kOutputRows &&
           input_rows == Nvfp4MlpDown12288Geometry::kInputRows;
}

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_add: unsupported shape"); }
    if (is_ornith_down(output_rows, input_rows)) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("nvfp4 linear_add: Ornith down admits only A16");
        }
        return use_ornith_dynamic_w4a4(tokens) ? Nvfp4LinearAddRoute::W4A4
                                               : Nvfp4LinearAddRoute::A16;
    }
    if (!is_27b_residual(output_rows, input_rows)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) { return Nvfp4LinearAddRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported policy");
    }
    const std::int32_t first_w4a4 = input_rows == 6144 ? 7 : 8;
    return tokens >= first_w4a4 ? Nvfp4LinearAddRoute::W4A4 : Nvfp4LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    if (is_ornith_down(weight.n, weight.k) && x.ne[1] >= kNvfp4OrnithMmaMinT) {
        launch_nvfp4_w4a16_mma_add(x, weight, residual, stream);
        return;
    }
    const std::int32_t kChunk = is_ornith_down(weight.n, weight.k) ? kNvfp4OrnithLastSmallT
                                                                  : kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            nvfp4_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            nvfp4_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearAddRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
    auto scope                       = workspace.scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], weight.k);
    if (is_ornith_down(weight.n, weight.k)) {
        launch_nvfp4_dynamic_w4a4_add(x, weight, residual, scratch, stream);
        return;
    }
    nvfp4_linear_add_w4a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
