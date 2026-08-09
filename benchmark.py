import time
import torch

# ============================================================
# CONFIG
# ============================================================

M = 10000
K = 10000
N = 10000

WARMUP = 30
ITERATIONS = 100

device = "cuda"

torch.set_grad_enabled(False)

print("=" * 75)
print("L40 PyTorch / TorchAO Precision Benchmark")
print("=" * 75)

print("GPU       :", torch.cuda.get_device_name(0))
print("CUDA      :", torch.version.cuda)
print("PyTorch   :", torch.__version__)
print("M,K,N     :", M, K, N)
print("=" * 75)


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(name, fn, ops):

    print(f"\n{name}")

    # Warmup
    for _ in range(WARMUP):
        y = fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(ITERATIONS):
        y = fn()

    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / ITERATIONS

    tflops = ops / (avg_ms / 1000) / 1e12

    print(f"Time        : {avg_ms:.4f} ms")
    print(f"Throughput  : {tflops:.2f} TFLOPS/TOPS")

    return avg_ms, tflops


# GEMM operations
OPS = 2 * M * K * N


# ============================================================
# FP32
# ============================================================

x = torch.randn(
    M, K,
    device=device,
    dtype=torch.float32
)

linear_fp32 = torch.nn.Linear(
    K, N,
    bias=False,
    device=device,
    dtype=torch.float32
)

benchmark(
    "FP32",
    lambda: linear_fp32(x),
    OPS
)


# ============================================================
# FP16
# ============================================================

x = torch.randn(
    M, K,
    device=device,
    dtype=torch.float16
)

linear_fp16 = torch.nn.Linear(
    K, N,
    bias=False,
    device=device,
    dtype=torch.float16
)

benchmark(
    "FP16",
    lambda: linear_fp16(x),
    OPS
)


# ============================================================
# BF16
# ============================================================

x = torch.randn(
    M, K,
    device=device,
    dtype=torch.bfloat16
)

linear_bf16 = torch.nn.Linear(
    K, N,
    bias=False,
    device=device,
    dtype=torch.bfloat16
)

benchmark(
    "BF16",
    lambda: linear_bf16(x),
    OPS
)


# ============================================================
# TORCHAO
# ============================================================

try:

    from torchao.quantization import (
        quantize_,
        Int8DynamicActivationInt8WeightConfig,
        Int4WeightOnlyConfig,
        Float8DynamicActivationFloat8WeightConfig,
    )

    print("\nTorchAO detected.")

except Exception as e:

    print("\nTorchAO unavailable:")
    print(e)
    print("\nInstall with:")
    print("pip install torchao")

    raise SystemExit


# ============================================================
# FP8
# ============================================================

try:

    print("\nCreating FP8 model...")

    model_fp8 = torch.nn.Linear(
        K,
        N,
        bias=False,
        device=device,
        dtype=torch.bfloat16
    )

    quantize_(
        model_fp8,
        Float8DynamicActivationFloat8WeightConfig()
    )

    x = torch.randn(
        M,
        K,
        device=device,
        dtype=torch.bfloat16
    )

    benchmark(
        "FP8 TorchAO",
        lambda: model_fp8(x),
        OPS
    )

except Exception as e:

    print("FP8 FAILED:")
    print(e)


# ============================================================
# INT8
# ============================================================

try:

    print("\nCreating INT8 model...")

    model_int8 = torch.nn.Linear(
        K,
        N,
        bias=False,
        device=device,
        dtype=torch.bfloat16
    )

    quantize_(
        model_int8,
        Int8DynamicActivationInt8WeightConfig()
    )

    x = torch.randn(
        M,
        K,
        device=device,
        dtype=torch.bfloat16
    )

    benchmark(
        "INT8 TorchAO",
        lambda: model_int8(x),
        OPS
    )

except Exception as e:

    print("INT8 FAILED:")
    print(e)


# ============================================================
# INT4
# ============================================================

try:

    print("\nCreating INT4 model...")

    model_int4 = torch.nn.Linear(
        K,
        N,
        bias=False,
        device=device,
        dtype=torch.bfloat16
    )

    quantize_(
        model_int4,
        Int4WeightOnlyConfig(
            group_size=128
        )
    )

    x = torch.randn(
        M,
        K,
        device=device,
        dtype=torch.bfloat16
    )

    benchmark(
        "INT4 TorchAO",
        lambda: model_int4(x),
        OPS
    )

except Exception as e:

    print("INT4 FAILED:")
    print(e)


print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)