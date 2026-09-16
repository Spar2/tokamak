#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8GdnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8GdnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    return tokens >= 8 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

enum class Fp8GdnInputParent : std::uint8_t {
    Wide5120,
    Narrow4096,
};

Fp8GdnInputParent resolve_parent(const Weight& weight) {
    if (weight.n == Fp8GdnInputGeometry::kOutputRows &&
        weight.k == Fp8GdnInputGeometry::kInputRows) {
        return Fp8GdnInputParent::Wide5120;
    }
    // Ornith-1.5-9B fused parent (qk 4096 + v 4096 + z 4096).
    if (weight.n == Fp8GdnInput4096Geometry::kOutputRows &&
        weight.k == Fp8GdnInput4096Geometry::kInputRows) {
        return Fp8GdnInputParent::Narrow4096;
    }
    throw std::invalid_argument("fp8 gdn_input_proj: unsupported parent shape");
}

} // namespace

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8GdnInputGeometry::kInputRows)
               : 0;
}

// 4096-wide parent workspace (A8 scratch scales with K).
std::size_t fp8_gdn_input_4096_workspace_capacity_bytes(LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8GdnInput4096Geometry::kInputRows)
               : 0;
}

void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    if (resolve_parent(weight) == Fp8GdnInputParent::Narrow4096) {
        if (x.ne[1] == 1) {
            fp8_gdn_input_4096_decode_launch(x, weight, qkv, z, stream);
        } else {
            fp8_gdn_input_4096_matrix_launch(x, weight, qkv, z, stream);
        }
        return;
    }
    if (x.ne[1] == 1) {
        fp8_gdn_input_decode_launch(x, weight, qkv, z, stream);
    } else {
        fp8_gdn_input_matrix_launch(x, weight, qkv, z, stream);
    }
}

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    if (resolve_parent(weight) == Fp8GdnInputParent::Narrow4096) {
        fp8_gdn_input_4096_a8_launch(x, weight, qkv, z, scratch, stream);
        return;
    }
    fp8_gdn_input_a8_launch(x, weight, qkv, z, scratch, stream);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8GdnInputRoute::A16) {
        fp8_gdn_input_a16_dispatch(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    fp8_gdn_input_a8_dispatch(x, weight, qkv, z, *workspace, stream);
}

} // namespace ninfer::ops::detail
