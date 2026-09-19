#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace ninfer::artifact {

class MaterializedArtifact;

[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);

[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

struct RotationPlan {
    ObjectHandle signs;
    ObjectHandle spec;
};

// Rotation resources stay host-resident (RawBytesV1): F32LE sign vector of
// sign_width entries plus a JSON descriptor. Device upload happens once at
// program/test creation; kernels receive an explicit const float* argument.
// Weight is never widened with rotation fields.
[[nodiscard]] inline RotationPlan bind_rotation(Binder& binder, std::string_view signs_name,
                                                std::string_view spec_name) {
    return RotationPlan{bind_raw_resource(binder, signs_name), bind_raw_resource(binder, spec_name)};
}

struct RotationHost {
    std::span<const float> signs;
    std::span<const std::byte> spec;
};

[[nodiscard]] RotationHost
materialized_rotation(const MaterializedArtifact& materialized, const RotationPlan& plan,
                      std::uint32_t expect_sign_width);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

[[nodiscard]] Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns);

} // namespace ninfer::artifact
