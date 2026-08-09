#!/usr/bin/env python3

import time
import torch

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"
N = 4096                 # Matrix size
WARMUP = 20
ITERATIONS = 100

print("=" * 70)
print("NVIDIA CUDA Datatype Benchmark")
print("=" * 70)

# ------------------------------------------------------------
# GPU information
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

gpu = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)

print(f"GPU           : {gpu}")
print(f"Compute       : {props.major}.{props.minor}")
print(f"CUDA          : {torch.version.cuda}")
print(f"PyTorch       : {torch.__version__}")
print(f"Matrix size   : {N} x {N}")
print("=" * 70)


# ============================================================
# Benchmark helper
# ============================================================

def benchmark_gemm(name, dtype, operation="matmul"):
    """
    Benchmark C = A @ B.

    For floating point:
        FLOPs = 2*N^3

    For integer:
        Operations = 2*N^3
    """

    print(f"\n{name}")

    try:
        if dtype == "fp64":
            A = torch.randn((N, N), device=DEVICE, dtype=torch.float64)
            B = torch.randn((N, N), device=DEVICE, dtype=torch.float64)

            # Warmup
            for _ in range(WARMUP):
                C = torch.matmul(A, B)

            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for _ in range(ITERATIONS):
                C = torch.matmul(A, B)

            end.record()
            torch.cuda.synchronize()

        elif dtype == "fp32":
            A = torch.randn((N, N), device=DEVICE, dtype=torch.float32)
            B = torch.randn((N, N), device=DEVICE, dtype=torch.float32)

            for _ in range(WARMUP):
                C = torch.matmul(A, B)

            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for _ in range(ITERATIONS):
                C = torch.matmul(A, B)

            end.record()
            torch.cuda.synchronize()

        elif dtype == "fp16":
            A = torch.randn((N, N), device=DEVICE, dtype=torch.float16)
            B = torch.randn((N, N), device=DEVICE, dtype=torch.float16)

            for _ in range(WARMUP):
                C = torch.matmul(A, B)

            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for _ in range(ITERATIONS):
                C = torch.matmul(A, B)

            end.record()
            torch.cuda.synchronize()

        elif dtype == "bf16":
            A = torch.randn((N, N), device=DEVICE, dtype=torch.bfloat16)
            B = torch.randn((N, N), device=DEVICE, dtype=torch.bfloat16)

            for _ in range(WARMUP):
                C = torch.matmul(A, B)

            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for _ in range(ITERATIONS):
                C = torch.matmul(A, B)

            end.record()
            torch.cuda.synchronize()

        else:
            print("Datatype handled separately.")
            return None

        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / ITERATIONS

        # Matrix multiplication:
        # A(NxN) @ B(NxN)
        # approximately 2*N^3 floating-point operations
        operations = 2 * (N ** 3)

        tflops = operations / (avg_ms / 1000) / 1e12

        print(f"Average time : {avg_ms:.4f} ms")
        print(f"Performance  : {tflops:.2f} TFLOPS")

        return tflops

    except Exception as e:
        print(f"FAILED: {e}")
        return None


# ============================================================
# Integer benchmark
# ============================================================

def benchmark_int8():
    print("\nINT8")

    try:
        # INT8 GEMM
        A = torch.randint(
            -128, 127,
            (N, N),
            device=DEVICE,
            dtype=torch.int8
        )

        B = torch.randint(
            -128, 127,
            (N, N),
            device=DEVICE,
            dtype=torch.int8
        )

        # PyTorch does not provide the same optimized INT8
        # Tensor Core path through normal torch.matmul on
        # every version.
        #
        # Therefore this test is mainly a compatibility test.

        for _ in range(WARMUP):
            C = torch.matmul(A, B)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(ITERATIONS):
            C = torch.matmul(A, B)

        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / ITERATIONS

        operations = 2 * (N ** 3)
        tops = operations / (avg_ms / 1000) / 1e12

        print(f"Average time : {avg_ms:.4f} ms")
        print(f"Performance  : {tops:.2f} TOPS")

        return tops

    except Exception as e:
        print(f"INT8 unavailable through this PyTorch path: {e}")
        return None


# ============================================================
# FP8 benchmark
# ============================================================

def benchmark_fp8():

    print("\nFP8")

    try:
        # PyTorch FP8 datatype
        fp8_dtype = torch.float8_e4m3fn

        A = torch.randn(
            (N, N),
            device=DEVICE,
            dtype=torch.float32
        ).to(fp8_dtype)

        B = torch.randn(
            (N, N),
            device=DEVICE,
            dtype=torch.float32
        ).to(fp8_dtype)

        # FP8 matmul support varies by PyTorch/CUDA version.
        for _ in range(WARMUP):
            C = torch.matmul(A, B)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(ITERATIONS):
            C = torch.matmul(A, B)

        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / ITERATIONS

        operations = 2 * (N ** 3)
        tops = operations / (avg_ms / 1000) / 1e12

        print(f"Average time : {avg_ms:.4f} ms")
        print(f"Performance  : {tops:.2f} TOPS")

        return tops

    except Exception as e:
        print(f"FP8 unavailable through torch.matmul: {e}")
        return None


# ============================================================
# INT4 benchmark
# ============================================================

def benchmark_int4():

    print("\nINT4")

    print("""
INT4 does not have a normal torch.matmul(INT4) path that
directly exposes the L40 Tensor Core INT4 throughput.

For a real INT4 Tensor Core benchmark, use:
    - CUTLASS
    - cuBLASLt
    - TensorRT
    - Transformer Engine

Skipping generic PyTorch INT4 test.
""")

    return None


# ============================================================
# Run benchmarks
# ============================================================

results = {}

results["FP64"] = benchmark_gemm("FP64", "fp64")
results["FP32"] = benchmark_gemm("FP32", "fp32")
results["FP16"] = benchmark_gemm("FP16", "fp16")
results["BF16"] = benchmark_gemm("BF16", "bf16")

results["FP8"] = benchmark_fp8()
results["INT8"] = benchmark_int8()
results["INT4"] = benchmark_int4()


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"{'Datatype':<12} {'Performance':>18}")
print("-" * 70)

for dtype, value in results.items():

    if value is None:
        print(f"{dtype:<12} {'N/A':>18}")
    else:
        unit = "TFLOPS" if dtype.startswith("FP") or dtype == "BF16" else "TOPS"
        print(f"{dtype:<12} {value:>12.2f} {unit}")

print("=" * 70)

print("""
Notes:
- FP64/FP32/FP16/BF16 are GEMM benchmarks.
- FP8/INT8/INT4 require specialized kernels for meaningful
  Tensor Core throughput measurements.
- Increase N to 8192 or 16384 for a more stable GPU benchmark.
- Run several times and compare the results.
""")