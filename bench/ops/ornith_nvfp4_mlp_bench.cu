// Ornith NVFP4 MLP W4A16 validation benchmark (standalone, no Engine/runtime).
//
// Strict A/B for Ornith-1.5-9B MLP GEMMs via ops::linear (bias-free projection)
// so both paths share one op. --matrix selects:
//   gate_up: logical N=24576, K=4096 (gate+up fused); A=Q4G64 (registered
//      (24576,4096) Q4 launch), B=native NVFP4 (Nvfp4MlpGateUp4096Geometry)
//   down:    logical N=4096, K=12288 (single down_proj); A=Q5G64 (registered
//      (4096,12288) Q5 launch: q5_dispatch.cpp case 12288/4096, all T),
//      B=native NVFP4 (Nvfp4MlpDown12288Geometry)
//
// W4A16 isolates "are NVFP4 weights useful?" from W4A4 ("are FP4 Tensor Cores
// useful?"). W4A4 needs a calibrated input_scale_divisor the Ornith source does
// not ship (MLP input_quantizer is disabled: W4A16_NVFP4); it is out of scope
// (nvfp4_w4a4.cu throws for the 4096-K problem).
//
// Bench note: an earlier draft targeted ops::linear_swiglu, whose wrapper gates
// NVFP4 to 34816x5120. The pilot runs at ops::linear level instead, where the
// MlpGateUp4096 problem is now registered for W4A16.
//
// Weight handling: native U8/F8/F32 words are preserved end to end (packed
// codes concatenated gate+up, natural scales concatenated then swizzled to
// blockscale-k16-m128x4-v1 per tools/artifact/layouts.py). No BF16 round-trip.
// Divisor equality between gate and up is enforced (mirrors
// recipe_nvfp4._same_divisor); E4M3 scale words are validated (mirrors
// validate_nvfp4_words: no sign bit, no 0x7F).

#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "quantized_weight.cuh"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
namespace fs = std::filesystem;

namespace {

// Ornith-1.5-9B MLP parents.
struct MatrixSpec {
    const char* name;      // "gate_up" | "down"
    std::int32_t n, k;     // logical GEMM shape
    QType base_qtype;       // production groupwise-int format
    const char* base_label; // "Q4-prod" | "Q5-prod"
};

constexpr MatrixSpec kGateUp{ "gate_up", 24576, 4096, QType::Q4G64_F16S, "Q4-prod" };
constexpr MatrixSpec kDown{ "down", 4096, 12288, QType::Q5G64_F16S, "Q5-prod" };

constexpr std::int32_t kGateRows = 12288;  // gate/up part rows (gate_up only)
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    std::string nvfp4_dir = R"(O:\LLM\models\Ornith-1.5-9B-NVFP4)";
    std::string matrix    = "gate_up";
    int layer = 0;
    std::vector<std::int32_t> t_sweep{1, 2, 4, 8, 16, 32};
    int warmup = 5;
    int repeat = 30;
    std::string activations;  // optional raw BF16 [4096 x Tmax] replay file
    std::string csv_out;
    bool profile = false;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> out;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string tok(raw.substr(begin, end == std::string_view::npos
                                                      ? raw.size() - begin
                                                      : end - begin));
        const long v = std::stol(tok);
        if (v <= 0 || v > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("--t-sweep values must be positive int32");
        out.push_back(static_cast<std::int32_t>(v));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    if (out.empty()) throw std::invalid_argument("--t-sweep must not be empty");
    return out;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) throw std::invalid_argument(std::string("missing ") + label);
            return argv[i];
        };
        if (a == "--nvfp4-dir") {
            o.nvfp4_dir = std::string(next("--nvfp4-dir"));
        } else if (a == "--matrix") {
            o.matrix = std::string(next("--matrix"));
            if (o.matrix != "gate_up" && o.matrix != "down")
                throw std::invalid_argument("--matrix must be gate_up or down");
        } else if (a == "--layer") {
            o.layer = std::stoi(std::string(next("--layer")));
            if (o.layer < 0 || o.layer > 31) throw std::invalid_argument("--layer must be 0..31");
        } else if (a == "--t-sweep") {
            o.t_sweep = parse_t_sweep(next("--t-sweep"));
        } else if (a == "--warmup") {
            o.warmup = std::stoi(std::string(next("--warmup")));
        } else if (a == "--repeat") {
            o.repeat = std::stoi(std::string(next("--repeat")));
        } else if (a == "--activations") {
            o.activations = std::string(next("--activations"));
        } else if (a == "--csv-out") {
            o.csv_out = std::string(next("--csv-out"));
        } else if (a == "--profile") {
            o.profile = true;
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "Usage: %s [--nvfp4-dir PATH] [--matrix gate_up|down] [--layer L] "
                "[--t-sweep 1,2,4,8,16,32]\n"
                "       [--warmup N] [--repeat N] [--activations RAW_BF16] [--csv-out PATH] "
                "[--profile]\n"
                "\n"
                "  --nvfp4-dir   dir with model.safetensors (default: O:\\LLM\\models\\\n"
                "                Ornith-1.5-9B-NVFP4)\n"
                "  --activations raw LE BF16 [K x Tmax] replay capture; absent =>\n"
                "                deterministic ramp, labeled SYNTHETIC\n"
                "  --profile     single-T NVTX/profile run (requires one T)\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(a));
        }
    }
    if (o.warmup < 0 || o.repeat <= 0)
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    if (o.profile && o.t_sweep.size() != 1)
        throw std::invalid_argument("--profile requires exactly one T");
    return o;
}

// ---- safetensors slicing (header JSON via nlohmann, payload via pread) ----

struct Slice {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct SafetensorsFile {
    std::ifstream f;
    std::uint64_t data_base = 0;
    nlohmann::json header;

    explicit SafetensorsFile(const fs::path& path) : f(path, std::ios::binary) {
        if (!f) throw std::runtime_error("cannot open " + path.string());
        std::uint64_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 8);
        if (!f) throw std::runtime_error("cannot read safetensors header length");
        std::string raw(static_cast<std::size_t>(hlen), '\0');
        f.read(raw.data(), static_cast<std::streamsize>(hlen));
        if (!f) throw std::runtime_error("cannot read safetensors header JSON");
        header    = nlohmann::json::parse(raw);
        data_base = 8 + hlen;
    }

    void entry(const std::string& name, std::string& dtype, std::vector<std::int64_t>& shape,
               Slice& slice) const {
        const auto it = header.find(name);
        if (it == header.end()) throw std::runtime_error("missing tensor: " + name);
        dtype = it->at("dtype").get<std::string>();
        for (const auto& d : it->at("shape")) shape.push_back(d.get<std::int64_t>());
        const auto& off = it->at("data_offsets");
        slice.begin     = off.at(0).get<std::uint64_t>();
        slice.end       = off.at(1).get<std::uint64_t>();
    }

    std::vector<std::uint8_t> read_bytes(const Slice& s) {
        std::vector<std::uint8_t> out(static_cast<std::size_t>(s.end - s.begin));
        f.seekg(static_cast<std::streamoff>(data_base + s.begin), std::ios::beg);
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
        if (!f) throw std::runtime_error("safetensors payload read failed");
        return out;
    }
};

void require_entry(SafetensorsFile& sf, const std::string& name, const std::string& want_dtype,
                   std::vector<std::int64_t> want_shape, std::vector<std::uint8_t>& out) {
    std::string dtype;
    std::vector<std::int64_t> shape;
    Slice s;
    sf.entry(name, dtype, shape, s);
    if (dtype != want_dtype || shape != want_shape)
        throw std::runtime_error(name + ": signature mismatch (got " + dtype + " [...], want " +
                                 want_dtype + ")");
    out = sf.read_bytes(s);
}

// Mirrors tools/convert/qwen3_6_27b/recipe_nvfp4.py::validate_nvfp4_words (scales)
// and _word (divisor must be finite positive FP32).
float checked_divisor(const std::vector<std::uint8_t>& raw, const std::string& name) {
    if (raw.size() != 4) throw std::runtime_error(name + ": divisor must be 4 bytes");
    float v = 0.0F;
    std::memcpy(&v, raw.data(), 4);
    if (!std::isfinite(v) || v <= 0.0F)
        throw std::runtime_error(name + ": divisor must be finite and positive");
    return v;
}

void checked_scales(const std::vector<std::uint8_t>& scales, const std::string& name) {
    for (std::size_t i = 0; i < scales.size(); ++i) {
        const std::uint8_t b = scales[i];
        if ((b & 0x80) != 0 || b == 0x7F)
            throw std::runtime_error(name + ": invalid E4M3FN scale word at " +
                                     std::to_string(i));
    }
}

// Swizzle natural [N,K/16] scales to blockscale-k16-m128x4-v1 stored order.
// Mirrors tools/artifact/layouts.py::swizzle_nvfp4_scales:
//   source.reshape(N/128,4,32,K_tiles,4).permute(0,3,2,1,4)
std::vector<std::uint8_t> swizzle_scales(const std::vector<std::uint8_t>& natural, int n,
                                         int groups_per_row, int k_tiles) {
    const int n128 = n / 128;
    std::vector<std::uint8_t> stored(natural.size());
    for (int b = 0; b < n128; ++b)
        for (int kt = 0; kt < k_tiles; ++kt)
            for (int r = 0; r < 32; ++r)
                for (int q = 0; q < 4; ++q)
                    for (int e = 0; e < 4; ++e) {
                        const int src_row = b * 128 + q * 32 + r;
                        const int src_col = kt * 4 + e;
                        const std::size_t si =
                            static_cast<std::size_t>(src_row) * groups_per_row + src_col;
                        const std::size_t di = ((((static_cast<std::size_t>(b) * k_tiles + kt) * 32 +
                                                  r) * 4 +
                                                 q) * 4) +
                                               e;
                        stored[di] = natural[si];
                    }
    return stored;
}

struct NativeMatrix {
    std::vector<std::uint8_t> packed;   // native U8 codes, fused row order
    std::vector<std::uint8_t> swizzled; // stored-order scales
    float divisor = 0.0F;
};

NativeMatrix load_native_matrix(const Options& o, const MatrixSpec& spec) {
    const fs::path dir(o.nvfp4_dir);
    SafetensorsFile sf(dir / "model.safetensors");
    const std::string p =
        "model.language_model.layers." + std::to_string(o.layer) + ".mlp.";
    const int groups_per_row = spec.k / 16;
    const int k_tiles        = spec.k / 64;

    std::vector<std::uint8_t> packed_all, natural_all;
    float divisor = 0.0F;
    bool first    = true;
    // gate_up fuses gate+up row blocks (divisors must match); down is one block.
    const char* parts[2]      = {"gate_proj", "up_proj"};
    const int part_rows[2]    = {kGateRows, kGateRows};
    const int nparts          = (spec.n == 24576) ? 2 : 1;
    const char* single[1]     = {"down_proj"};
    const int single_rows[1]  = {4096};
    const char** names        = (nparts == 2) ? parts : single;
    const int* rows           = (nparts == 2) ? part_rows : single_rows;
    for (int i = 0; i < nparts; ++i) {
        std::vector<std::uint8_t> pk, sc, dv;
        const std::string base = p + names[i];
        require_entry(sf, base + ".weight", "U8", {rows[i], spec.k / 2}, pk);
        require_entry(sf, base + ".weight_scale", "F8_E4M3",
                      {rows[i], spec.k / 16}, sc);
        require_entry(sf, base + ".weight_scale_2", "F32", {}, dv);
        checked_scales(sc, base + ".weight_scale");
        const float d = checked_divisor(dv, base + ".weight_scale_2");
        if (first) {
            divisor = d;
            first   = false;
        } else if (dv !=
                   std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(&divisor),
                                             reinterpret_cast<const std::uint8_t*>(&divisor) + 4)) {
            throw std::runtime_error("gate/up weight_scale_2 words differ: fused parent invalid");
        }
        packed_all.insert(packed_all.end(), pk.begin(), pk.end());
        natural_all.insert(natural_all.end(), sc.begin(), sc.end());
    }
    if (static_cast<std::int64_t>(packed_all.size()) !=
        static_cast<std::int64_t>(spec.n) * spec.k / 2)
        throw std::runtime_error("native packed size does not match matrix shape");

    NativeMatrix out;
    out.packed   = std::move(packed_all);
    out.swizzled = swizzle_scales(natural_all, spec.n, groups_per_row, k_tiles);
    out.divisor  = divisor;
    // Word-level round-trip: swizzle is a permutation; unswizzle must recover input.
    std::uint64_t mism = 0;
    for (int b = 0; b < spec.n / 128; ++b)
        for (int kt = 0; kt < k_tiles; ++kt)
            for (int r = 0; r < 32; ++r)
                for (int q = 0; q < 4; ++q)
                    for (int e = 0; e < 4; ++e) {
                        const int row = b * 128 + q * 32 + r;
                        const int col = kt * 4 + e;
                        const std::size_t si =
                            static_cast<std::size_t>(row) * groups_per_row + col;
                        const std::size_t di =
                            ((((static_cast<std::size_t>(b) * k_tiles + kt) * 32 + r) * 4 + q) *
                                 4) +
                            e;
                        mism += (natural_all[si] != out.swizzled[di]);
                    }
    if (mism != 0) throw std::runtime_error("NVFP4 scale swizzle round-trip mismatch");
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const MatrixSpec& spec =
            (options.matrix == "down") ? kDown : kGateUp;
        const std::int32_t kN = spec.n, kK = spec.k;
        const std::uint64_t kCodeBytes  = static_cast<std::uint64_t>(kN) * kK / 2;
        const std::uint64_t kScaleBytes = static_cast<std::uint64_t>(kN) * kK / 16;
        const auto mm         = std::minmax_element(options.t_sweep.begin(), options.t_sweep.end());
        const std::int32_t min_t = *mm.first, max_t = *mm.second;

        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        std::printf("device: %s (cc %d.%d, %d SMs) matrix=%s N=%d K=%d\n", props.name, props.major,
                    props.minor, props.multiProcessorCount, spec.name, kN, kK);
        std::printf("run nvidia-smi dmon -s puc alongside for power/clocks (no NVML dep).\n");

        // ---- native NVFP4 verification (host-side, no kernels) ----
        const NativeMatrix native = load_native_matrix(options, spec);
        std::printf("native NVFP4 %s layer %d: packed=%zu scales=%zu divisor=%g\n", spec.name,
                    options.layer, native.packed.size(), native.swizzled.size(),
                    static_cast<double>(native.divisor));
        std::printf("row layout: VALID  divisor match: VALID  E4M3 scales: VALID  "
                    "swizzle round-trip: VALID\n");

        // Device image of native words (BlockScale payload without re-layout games).
        const std::uint64_t code_bytes   = kCodeBytes;
        const std::uint64_t scale_offset = (code_bytes + 255) / 256 * 256;
        const std::uint64_t payload      = scale_offset + kScaleBytes + 4;
        DeviceBuffer nvfp4_dev(static_cast<std::size_t>(payload));
        CUDA_CHECK(cudaMemset(nvfp4_dev.p, 0, nvfp4_dev.bytes));
        CUDA_CHECK(cudaMemcpy(nvfp4_dev.p, native.packed.data(), native.packed.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(nvfp4_dev.p) + scale_offset,
                              native.swizzled.data(), native.swizzled.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(nvfp4_dev.p) + scale_offset + kScaleBytes,
                              &native.divisor, 4, cudaMemcpyHostToDevice));
        Weight nvfp4_w{};
        nvfp4_w.payload              = nvfp4_dev.p;
        nvfp4_w.payload_bytes        = payload;
        nvfp4_w.qtype                = QType::NVFP4;
        nvfp4_w.layout               = QuantLayout::BlockScaleK16M128x4;
        nvfp4_w.scale_dtype          = DType::FP8_E4M3FN;
        nvfp4_w.group_size           = 16;
        nvfp4_w.group                = 16;
        nvfp4_w.ndim                 = 2;
        nvfp4_w.shape[0] = nvfp4_w.padded_shape[0] = kN;
        nvfp4_w.shape[1] = nvfp4_w.padded_shape[1] = kK;
        nvfp4_w.qdata                = nvfp4_dev.p;
        nvfp4_w.qhigh                = nullptr;
        nvfp4_w.scales               = static_cast<const std::uint8_t*>(nvfp4_dev.p) + scale_offset;
        nvfp4_w.n                    = kN;
        nvfp4_w.k                    = kK;
        nvfp4_w.weight_scale_divisor = native.divisor;
        nvfp4_w.input_scale_divisor  = 1.0F;  // W4A16 placeholder: A16 kernels ignore it;
                                              // W4A4 calibration is out of scope.

        // ---- activations: replay capture or labeled synthetic ramp ----
        DeviceBuffer input(static_cast<std::size_t>(kK) * max_t * 2);
        std::string act_label = "SYNTHETIC-RAMP";
        if (!options.activations.empty()) {
            std::ifstream af(options.activations, std::ios::binary | std::ios::ate);
            if (!af) throw std::runtime_error("cannot open --activations file");
            const std::streamsize sz = af.tellg();
            const std::streamsize want =
                static_cast<std::streamsize>(kK) * max_t * 2;
            if (sz != want)
                throw std::runtime_error("--activations size must be K*Tmax*2 bytes");
            std::vector<char> hb(static_cast<std::size_t>(sz));
            af.seekg(0, std::ios::beg);
            af.read(hb.data(), sz);
            input.copy_from_host(hb.data(), hb.size());
            act_label = "REPLAY:" + options.activations;
        } else {
            DeviceBuffer tmp = bench::make_bf16(static_cast<std::size_t>(kK) * max_t);
            CUDA_CHECK(cudaMemcpy(input.p, tmp.p, tmp.bytes, cudaMemcpyDeviceToDevice));
        }
        DeviceBuffer output(static_cast<std::size_t>(kN) * max_t * 2);
        DeviceBuffer flush(kFlushBytes);

        // ---- side A: production groupwise-int baseline (registered 9B shape) ----
        bench::PackedQuantizedWeight base =
            bench::make_row_split_weight(spec.base_qtype, kN, kK, kK);
        const std::size_t q4_ws = ops::linear_workspace_capacity_bytes(
            spec.base_qtype, kN, kK, ops::LinearPolicy::A16Only, min_t, max_t);
        WorkspaceArena q4_arena(std::max<std::size_t>(q4_ws, 256));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::printf("%-6s %-14s %11s %11s %11s %10s %10s %10s\n", "T", "path", "median_us",
                    "min_us", "p95_us", "eff_GB/s", "TFLOP/s", "tok/s");
        struct Row {
            std::int32_t t;
            const char* path;
            bench::ColdTiming tm;
            double gbs, tflops, toks;
        };
        std::vector<Row> rows;
        const double weight_q4 = static_cast<double>(base.model_weight_bytes());
        for (const std::int32_t t : options.t_sweep) {
            auto launch = [&](cudaStream_t s) {
                Tensor x(input.p, DType::BF16, {kK, t});
                Tensor out(output.p, DType::BF16, {kN, t});
                ops::linear(x, base.weight, out, ops::LinearPolicy::A16Only, q4_arena, s);
            };
            const bench::ColdTiming tm =
                bench::measure_cold_launch(launch, flush, stream, options.warmup, options.repeat);
            const double sec   = tm.median_us * 1.0e-6;
            const double flops = 2.0 * kN * kK * t;
            const double bytes = weight_q4 + 2.0 * (kK + kN) * t;
            std::printf("%-6d %-14s %11.3f %11.3f %11.3f %10.1f %10.2f %10.1f\n", t,
                        spec.base_label, tm.median_us, tm.min_us, tm.p95_us,
                        bytes / sec / 1e9, flops / sec / 1e12, t / sec);
            rows.push_back({t, spec.base_label, tm, bytes / sec / 1e9, flops / sec / 1e12, t / sec});
        }

        // ---- side B: native NVFP4 W4A16 ----
        bool nvfp4_ran = false;
        try {
            const std::size_t nv_ws = ops::linear_workspace_capacity_bytes(
                QType::NVFP4, kN, kK, ops::LinearPolicy::A16Only, min_t, max_t);
            WorkspaceArena nv_arena(std::max<std::size_t>(nv_ws, 256));
            for (const std::int32_t t : options.t_sweep) {
                auto launch = [&](cudaStream_t s) {
                    Tensor x(input.p, DType::BF16, {kK, t});
                    Tensor out(output.p, DType::BF16, {kN, t});
                    ops::linear(x, nvfp4_w, out, ops::LinearPolicy::A16Only, nv_arena, s);
                };
                const bench::ColdTiming tm = bench::measure_cold_launch(launch, flush, stream,
                                                                        options.warmup,
                                                                        options.repeat);
                const double sec   = tm.median_us * 1.0e-6;
                const double flops = 2.0 * kN * kK * t;
                const double bytes =
                    static_cast<double>(payload) + 2.0 * (kK + kN) * t;
                std::printf("%-6d %-14s %11.3f %11.3f %11.3f %10.1f %10.2f %10.1f\n", t,
                            "NVFP4-W4A16", tm.median_us, tm.min_us, tm.p95_us,
                            bytes / sec / 1e9, flops / sec / 1e12, t / sec);
                rows.push_back(
                    {t, "NVFP4-W4A16", tm, bytes / sec / 1e9, flops / sec / 1e12, t / sec});
            }
            nvfp4_ran = true;
        } catch (const std::exception& e) {
            std::printf("NVFP4-W4A16 FAILED: %s\n", e.what());
            std::printf("gate: src/ops/linear/nvfp4/nvfp4_config.h (MlpGateUp4096/MlpDown12288)\n");
        }

        // ---- side C: dynamic W4A4 (runtime per-K16 quant + FP4 MMA) ----
        try {
            const std::size_t w4a4_ws =
                ninfer::ops::detail::nvfp4_w4a4_workspace_capacity_bytes(max_t, kK);
            WorkspaceArena w4a4_arena(std::max<std::size_t>(w4a4_ws, 256));
            const auto scratch =
                ninfer::ops::detail::allocate_nvfp4_w4a4_workspace(w4a4_arena, max_t, kK);
            cudaEvent_t eq0 = nullptr, eq1 = nullptr, em1 = nullptr;
            CUDA_CHECK(cudaEventCreate(&eq0));
            CUDA_CHECK(cudaEventCreate(&eq1));
            CUDA_CHECK(cudaEventCreate(&em1));
            for (const std::int32_t t : options.t_sweep) {
                if (t < 32) { continue; }
                std::vector<double> total_us, quant_us, mma_us;
                total_us.reserve(static_cast<std::size_t>(options.repeat));
                quant_us.reserve(static_cast<std::size_t>(options.repeat));
                mma_us.reserve(static_cast<std::size_t>(options.repeat));
                for (int i = 0; i < options.warmup; ++i) {
                    Tensor x(input.p, DType::BF16, {kK, t});
                    Tensor out(output.p, DType::BF16, {kN, t});
                    ninfer::ops::detail::launch_nvfp4_dynamic_w4a4(x, nvfp4_w, out, scratch,
                                                                   stream);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int i = 0; i < options.repeat; ++i) {
                    bench::flush_l2(flush, stream);
                    Tensor x(input.p, DType::BF16, {kK, t});
                    Tensor out(output.p, DType::BF16, {kN, t});
                    CUDA_CHECK(cudaEventRecord(eq0, stream));
                    ninfer::ops::detail::launch_nvfp4_dynamic_w4a4_quantize(x, scratch, stream);
                    CUDA_CHECK(cudaEventRecord(eq1, stream));
                    ninfer::ops::detail::launch_nvfp4_dynamic_w4a4_mma(x, nvfp4_w, out, scratch,
                                                                      stream);
                    CUDA_CHECK(cudaEventRecord(em1, stream));
                    CUDA_CHECK(cudaEventSynchronize(em1));
                    float q_ms = 0.0F, t_ms = 0.0F, full_ms = 0.0F;
                    CUDA_CHECK(cudaEventElapsedTime(&q_ms, eq0, eq1));
                    CUDA_CHECK(cudaEventElapsedTime(&t_ms, eq1, em1));
                    CUDA_CHECK(cudaEventElapsedTime(&full_ms, eq0, em1));
                    quant_us.push_back(static_cast<double>(q_ms) * 1000.0);
                    mma_us.push_back(static_cast<double>(t_ms) * 1000.0);
                    total_us.push_back(static_cast<double>(full_ms) * 1000.0);
                }
                const bench::ColdTiming tm = bench::summarize_timings(total_us);
                const bench::ColdTiming tq = bench::summarize_timings(quant_us);
                const bench::ColdTiming tmma = bench::summarize_timings(mma_us);
                const double sec   = tm.median_us * 1.0e-6;
                const double flops = 2.0 * kN * kK * t;
                const double bytes =
                    static_cast<double>(payload) + 2.0 * (kK + kN) * t +
                    static_cast<double>(t) * kK / 2.0 + static_cast<double>(t) * kK / 16.0;
                std::printf("%-6d %-14s %11.3f %11.3f %11.3f %10.1f %10.2f %10.1f\n", t,
                            "dyn-W4A4", tm.median_us, tm.min_us, tm.p95_us, bytes / sec / 1e9,
                            flops / sec / 1e12, t / sec);
                std::printf("%-6d %-14s %11.3f  (quant median)  mma_median=%11.3f  ws=%zu\n", t,
                            "dyn-W4A4-split", tq.median_us, tmma.median_us, w4a4_ws);
                rows.push_back({t, "dyn-W4A4", tm, bytes / sec / 1e9, flops / sec / 1e12, t / sec});
            }
            CUDA_CHECK(cudaEventDestroy(eq0));
            CUDA_CHECK(cudaEventDestroy(eq1));
            CUDA_CHECK(cudaEventDestroy(em1));
        } catch (const std::exception& e) {
            std::printf("dyn-W4A4 FAILED: %s\n", e.what());
        }

        if (!options.csv_out.empty()) {
            const fs::path p(options.csv_out);
            if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
            std::ofstream out(options.csv_out);
            if (!out) throw std::runtime_error("failed to open CSV");
            out << "matrix,path,N,K,T,median_us,min_us,p95_us,eff_gbs,tflops,toks,"
                   "activations,warmup,repeat,flush_bytes\n";
            char buf[512];
            for (const auto& r : rows) {
                std::snprintf(buf, sizeof(buf), "%s,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.3f,%.1f,%s,"
                                                 "%d,%d,%zu\n",
                              spec.name, r.path, kN, kK, r.t, r.tm.median_us, r.tm.min_us,
                              r.tm.p95_us, r.gbs, r.tflops, r.toks, act_label.c_str(),
                              options.warmup, options.repeat, kFlushBytes);
                out << buf;
            }
        }

        std::printf("activations=%s\n", act_label.c_str());
        if (!nvfp4_ran)
            std::printf("decision: Q4 baseline above; NVFP4 side failed, see error.\n");
        else
            std::printf("decision: BOTH SIDES RAN. Apply >30%% / 5-15%% / lose rule per T.\n");
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_ornith_nvfp4_mlp_bench: %s\n", e.what());
        return 1;
    }
}
