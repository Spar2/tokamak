#include "targets/qwen3_5_9b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_5_9b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_5_9b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_5_9b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

ops::LinearPolicy text_policy(const Weight&) {
    return ops::LinearPolicy::A16Only;
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

// Hybrid GDN threshold: token widths at or above this use the native FP8
// fused parent; narrower widths stay on the groupwise split parents. The
// threshold is a static property of each captured graph (width x batch),
// so CUDA-graph capture bakes in the selected route.
constexpr std::int32_t kGdnHybridFp8MinTokens = 8;

bool use_hybrid_fp8_parent(std::int32_t tokens) {
    return tokens >= kGdnHybridFp8MinTokens;
}

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    if (const auto* hybrid =
            std::get_if<HybridGdnInputProjectionPayload>(&weights.input_projection)) {
        const std::size_t split = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch, width, width);
        const Weight& parent = hybrid->query_key_value_z;
        const std::size_t fused = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width);
        return std::max({kMinimumLeafWorkspaceBytes, split, fused});
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    if (const auto* hybrid =
            std::get_if<HybridGdnInputProjectionPayload>(&weights.input_projection)) {
        const std::size_t split = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch, width, width);
        const Weight& parent = hybrid->query_key_value_z;
        const std::size_t fused = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width);
        return std::max({kMinimumLeafWorkspaceBytes, split, fused});
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value,
                                   qwen3_6::TextPhase, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused),
                         workspace, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            stream);
        return;
    }
    if (const auto* hybrid =
            std::get_if<HybridGdnInputProjectionPayload>(&weights.input_projection)) {
        if (use_hybrid_fp8_parent(hidden.ne[1])) {
            ops::gdn_input_proj(hidden, hybrid->query_key_value_z, qkv, output_gate_flat,
                                text_policy(hybrid->query_key_value_z), workspace, stream);
            return;
        }
        ops::gdn_input_proj(hidden, hybrid->split.query_key, hybrid->split.value_z, qkv,
                            output_gate_flat, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    if (const auto* hybrid =
            std::get_if<HybridGdnInputProjectionPayload>(&weights.input_projection)) {
        if (use_hybrid_fp8_parent(hidden.ne[1] * hidden.ne[2])) {
            ops::gdn_input_proj_conv_snapshot(
                hidden, hybrid->query_key_value_z, conv_weight, conv_states, valid_columns,
                initial_slot, snapshot_base_slot, query, key, value, output_gate_view,
                text_policy(hybrid->query_key_value_z), leaf_workspace, stream);
            return;
        }
        ops::gdn_input_proj_conv_snapshot(hidden, hybrid->split.query_key, hybrid->split.value_z,
                                          conv_weight, conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    if (const auto* hybrid =
            std::get_if<HybridGdnInputProjectionPayload>(&weights.input_projection)) {
        if (use_hybrid_fp8_parent(hidden.ne[1] * hidden.ne[2])) {
            ops::gdn_input_proj_conv_record(hidden, hybrid->query_key_value_z, conv_weight,
                                            conv_states, valid_columns, initial_slots, conv_record,
                                            query, key, value, output_gate_view,
                                            text_policy(hybrid->query_key_value_z),
                                            leaf_workspace, stream);
            return;
        }
        ops::gdn_input_proj_conv_record(hidden, hybrid->split.query_key, hybrid->split.value_z,
                                        conv_weight, conv_states, valid_columns, initial_slots,
                                        conv_record, query, key, value, output_gate_view,
                                        leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace,
                                          DeviceExecutionView execution) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, execution);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, execution);
}

// Experimental Phase-2 capture hook (defined at end of file). No-op unless
// NINFER_DUMP_MLP_DIR is set; see the contract comment at the definition.
void debug_mlp_dump(const Tensor& hidden, const Tensor& activation, cudaStream_t stream);

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, const ::ninfer::ops::SparseMoeHints&,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up),
                       workspace, stream);
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
    debug_mlp_dump(hidden, activation, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::Nvfp4Mlp:
        return 0;
    }
    throw std::logic_error("invalid 9B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::Nvfp4Mlp:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    }
    throw std::logic_error("invalid 9B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::Nvfp4Mlp:
        return 0;
    }
    throw std::logic_error("invalid 9B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size,
    std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::Nvfp4Mlp:
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size, first,
            last);
    }
    throw std::logic_error("invalid 9B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size,
    std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::Nvfp4Mlp:
        return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size, first,
            last);
    }
    throw std::logic_error("invalid 9B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::Nvfp4Mlp:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    }
    throw std::logic_error("invalid 9B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    validate_token_interval(first, last);
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    const ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    QType gate_up_qtype            = QType::Q4G64_F16S;
    QType down_qtype               = QType::Q5G64_F16S;
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        break;
    case WeightsProfile::Nvfp4Mlp:
        gate_up_qtype = QType::NVFP4;
        down_qtype    = QType::NVFP4;
        break;
    default:
        throw std::invalid_argument("qwen3_5_9b: invalid weights profile");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

// ---------------------------------------------------------------------------
// Experimental MLP activation capture (Phase-2 quality tooling, w4a4 branch).
//
// Env contract (all optional; unset NINFER_DUMP_MLP_DIR = fully off):
//   NINFER_DUMP_MLP_DIR     destination directory (must exist; dumps stay local)
//   NINFER_DUMP_MLP_LAYERS  csv layer list, default "0,15,30"
//   NINFER_DUMP_MLP_TOKENS  first-N tokens per tensor, default 512
//   NINFER_DUMP_MLP_MIN_T   ignore invocations with T below this, default 64
//                           (startup warmup passes must not latch the dump)
//
// Layer identity comes from an invocation counter mod 32, which is valid only
// for single-stream ordered passes (MTP OFF, max-concurrency 1) WITHOUT CUDA
// graphs: graph replay does not re-execute host code, so captured graphs would
// dump capture-time data. Serve with --no-cuda-graph for capture runs.
// Dumps the first invocation per selected layer (gate_up input = post-norm
// hidden state [4096,T]; down input = SwiGLU output [12288,T]) as raw LE BF16
// plus a meta.json manifest. Never enabled in production.
// ---------------------------------------------------------------------------
namespace debug_mlp_dump_detail {
std::atomic<std::uint64_t> g_calls{0};
bool g_dumped[32] = {};

struct DumpConfig {
    bool enabled = false;
    std::string dir;
    bool layers[32] = {};
    int tokens      = 512;
    int min_t       = 64;
};

const DumpConfig& config() {
    static const DumpConfig cfg = [] {
        DumpConfig c;
        const char* dir = std::getenv("NINFER_DUMP_MLP_DIR");
        if (dir == nullptr || dir[0] == '\0') { return c; }
        c.dir     = dir;
        c.enabled = true;
        const char* layers = std::getenv("NINFER_DUMP_MLP_LAYERS");
        const std::string list = (layers != nullptr && layers[0] != '\0') ? layers : "0,15,30";
        std::size_t begin      = 0;
        while (begin <= list.size()) {
            const std::size_t end = list.find(',', begin);
            const std::string tok =
                (end == std::string::npos) ? list.substr(begin) : list.substr(begin, end - begin);
            try {
                const int layer = std::stoi(tok);
                if (layer >= 0 && layer < 32) { c.layers[layer] = true; }
            } catch (...) {}
            if (end == std::string::npos) { break; }
            begin = end + 1;
        }
        const char* tok = std::getenv("NINFER_DUMP_MLP_TOKENS");
        if (tok != nullptr && tok[0] != '\0') {
            try {
                c.tokens = std::max(1, std::stoi(tok));
            } catch (...) {}
        }
        const char* mint = std::getenv("NINFER_DUMP_MLP_MIN_T");
        if (mint != nullptr && mint[0] != '\0') {
            try {
                c.min_t = std::max(1, std::stoi(mint));
            } catch (...) {}
        }
        return c;
    }();
    return cfg;
}

void write_raw(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { return; }
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}
} // namespace debug_mlp_dump_detail

void debug_mlp_dump(const Tensor& hidden, const Tensor& activation, cudaStream_t stream) {
    using namespace debug_mlp_dump_detail;
    const DumpConfig& cfg = config();
    if (!cfg.enabled) { return; }
    const int layer = static_cast<int>(g_calls.fetch_add(1) % 32);
    if (!cfg.layers[layer] || g_dumped[layer]) { return; }
    // Skip warmup/decode trickles: first invocation per layer with T >= min_t.
    // Startup warmup passes (small synthetic T) must not latch the dump.
    if (hidden.ne[1] < cfg.min_t) { return; }
    g_dumped[layer] = true;
    const int take = std::min(cfg.tokens, hidden.ne[1]);
    if (take <= 0 || hidden.ne[0] != 4096 || activation.ne[0] != 12288 ||
        activation.ne[1] != hidden.ne[1]) {
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }
    char name[256];
    std::snprintf(name, sizeof(name), "%s/l%02d_gatein_K4096_T%d.bf16le", cfg.dir.c_str(), layer,
                  take);
    std::vector<std::uint8_t> host(static_cast<std::size_t>(4096) * take * 2);
    if (cudaMemcpy(host.data(), hidden.data, host.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return;
    }
    // Column-major [K,T] layout: first `take` columns are a contiguous prefix.
    write_raw(name, host.data(), host.size());
    std::snprintf(name, sizeof(name), "%s/l%02d_downin_K12288_T%d.bf16le", cfg.dir.c_str(), layer,
                  take);
    host.assign(static_cast<std::size_t>(12288) * take * 2, 0);
    if (cudaMemcpy(host.data(), activation.data, host.size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        return;
    }
    write_raw(name, host.data(), host.size());
    std::snprintf(name, sizeof(name),
                  "{\"layer\":%d,\"take\":%d,\"full_t\":%d,\"gatein\":\"l%02d_gatein_K4096_T%d."
                  "bf16le\",\"downin\":\"l%02d_downin_K12288_T%d.bf16le\"}\n",
                  layer, take, hidden.ne[1], layer, take, layer, take);
    std::ofstream meta(cfg.dir + "/meta.json", std::ios::app);
    if (meta) { meta << name; }
}

} // namespace ninfer::targets::qwen3_5_9b::detail
