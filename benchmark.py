import time
import importlib.util
import subprocess
import torch
from torch.nn import functional as F
from torch.nn import Linear

torch.set_grad_enabled(False)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    raise SystemExit("CUDA not available - this benchmark requires an NVIDIA GPU")

# (label, M, K, N)  M = sequence tokens, K/N = linear dims
# Wan 2.2: hidden=5120, FFN=20480, tokens at 896x896/105f = 338688
MATMUL_SIZES = [
    ("attn-4K",    4096,   5120,  5120),
    ("attn-16K",   16384,  5120,  5120),
    ("attn-90K",   90112,  5120,  5120),
    ("attn-339K",  338688, 5120,  5120),
    ("attn-442K",  442368, 5120,  5120),
    ("mlp-339K",   338688, 5120,  20480),
]

WAN_PARAMS = 14.29e9
TOKENS = {
    "640^2@105f": 172800,
    "896^2@105f": 338688,
    "1024^2@105f": 442368,
}


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def auto_iters(m, k, n):
    flops = 2.0 * m * k * n
    if flops > 1e13:
        return 3
    if flops > 1e12:
        return 5
    if flops > 1e11:
        return 15
    return 30


def auto_warm(m, k, n):
    return 1 if 2.0 * m * k * n > 1e12 else 3


def bench_linear(linear, x, iters, warm):
    fn = lambda: linear(x)
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def report_system():
    header("SYSTEM / KERNEL AVAILABILITY")
    print("GPU       :", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM      : {:.1f} GiB | SMs: {}".format(props.total_memory / 2**30, props.multi_processor_count))
    print("CUDA      :", torch.version.cuda)
    print("torch     :", torch.__version__)
    print("flash-sdp :", torch.backends.cuda.flash_sdp_enabled())
    print("mem_eff   :", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math-sdp  :", torch.backends.cuda.math_sdp_enabled())
    print("triton    :", "PRESENT" if importlib.util.find_spec("triton") else "NOT FOUND -> torchao falls back to non-triton kernels")
    try:
        import torchao
        print("torchao   :", getattr(torchao, "__version__", "unknown"))
    except Exception as err:
        print("torchao   : NOT IMPORTABLE:", err)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        print("nvidia-smi:", out.stdout.strip() if out.returncode == 0 else "n/a")
    except Exception:
        print("nvidia-smi: n/a")


def sweep_baseline():
    results = {}
    header("BASELINE NATIVE GEMM - TFLOPS by size")
    print(f"{'size':<12}{'FP32':>10}{'FP16':>10}{'BF16':>10}")
    for label, m, k, n in MATMUL_SIZES:
        flops = 2.0 * m * k * n
        iters, warm = auto_iters(m, k, n), auto_warm(m, k, n)
        row = [label]
        for name, dt in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
            bytes_out = (m * k + m * n) * (4 if dt == torch.float32 else 2)
            if bytes_out > 38e9:
                row.append(float("nan"))
                continue
            lin = Linear(k, n, bias=False, device=DEV, dtype=dt)
            x = torch.randn(m, k, device=DEV, dtype=dt)
            ms = bench_linear(lin, x, iters, warm)
            tflops = flops / (ms / 1e3) / 1e12
            results.setdefault(name, {})[label] = tflops
            row.append(tflops)
            del lin, x
        torch.cuda.empty_cache()
        print(f"{row[0]:<12}{row[1]:>10.1f}{row[2]:>10.1f}{row[3]:>10.1f}", flush=True)
    return results


def sweep_torchao():
    from torchao.quantization import (
        quantize_,
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int4WeightOnlyConfig,
        Float8WeightOnlyConfig,
        Float8DynamicActivationFloat8WeightConfig,
    )
    configs = [
        ("FP8-dyn (app transformer)", Float8DynamicActivationFloat8WeightConfig),
        ("FP8-wt-only", Float8WeightOnlyConfig),
        ("INT8-wt (app text_enc)", Int8WeightOnlyConfig),
        ("INT8-dyn", Int8DynamicActivationInt8WeightConfig),
        ("INT4-g128", Int4WeightOnlyConfig(group_size=128)),
    ]
    results = {}
    for name, cfg in configs:
        header("TORCHAO: " + name)
        print(f"{'size':<12}{'ms':>10}{'TFLOPS':>10}{'vs BF16':>9}{'mem':>8}")
        for label, m, k, n in MATMUL_SIZES:
            flops = 2.0 * m * k * n
            iters, warm = auto_iters(m, k, n), auto_warm(m, k, n)
            try:
                lin = Linear(k, n, bias=False, device=DEV, dtype=torch.bfloat16)
                quantize_(lin, cfg())
                torch._dynamo.reset()
                x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
                torch.cuda.reset_peak_memory_stats()
                ms = bench_linear(lin, x, iters, warm)
                tflops = flops / (ms / 1e3) / 1e12
                mem = torch.cuda.max_memory_allocated() / 2**30
                base = results.get("BF16", {}).get(label)
                vs = tflops / base if base else float("nan")
                results.setdefault(name, {})[label] = tflops
                print(f"{label:<12}{ms:>10.3f}{tflops:>10.1f}{vs:>9.2f}x{mem:>8.1f}", flush=True)
            except Exception as err:
                print(f"{label:<12}FAILED: {type(err).__name__}: {err}", flush=True)
            torch.cuda.empty_cache()
    return results


def sweep_sdpa():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    header("SDPA ATTENTION (native CUDA kernels - no triton required)")
    H, HD = 40, 128
    backends = [
        ("flash", SDPBackend.FLASH_ATTENTION, torch.backends.cuda.flash_sdp_enabled()),
        ("mem_eff", SDPBackend.EFFICIENT_ATTENTION, torch.backends.cuda.mem_efficient_sdp_enabled()),
        ("math", SDPBackend.MATH, True),
    ]
    for L in [4096, 90112, 338688]:
        q = torch.randn(1, H, L, HD, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(1, H, L, HD, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(1, H, L, HD, device=DEV, dtype=torch.bfloat16)
        flops = 2.0 * L * L * H * HD
        iters = 2 if L > 100000 else 5
        print(f"\nseq={L:<8} FLOPs/iter={flops/1e12:.2f} TFLOP")
        for bname, backend, enabled in backends:
            if not enabled or (backend == SDPBackend.MATH and L > 8192):
                print(f"  {bname:<10} skipped", flush=True)
                continue
            try:
                fn = lambda: F.scaled_dot_product_attention(q, k, v)
                with sdpa_kernel(backend):
                    for _ in range(1):
                        fn()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with sdpa_kernel(backend):
                    for _ in range(iters):
                        fn()
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / iters
                print(f"  {bname:<10} {ms:>8.2f} ms  {flops / (ms / 1e3) / 1e12:>8.1f} TFLOPS", flush=True)
            except Exception as err:
                print(f"  {bname:<10} FAILED: {type(err).__name__}: {err}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()


def bench_bandwidth():
    header("MEMORY BANDWIDTH + CPU-OFFLOAD SWAP COST")
    n = 2**30
    buf = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    _ = buf.clone()
    torch.cuda.synchronize()
    d2d = 2 * 2 * n / 1e9 / (time.time() - t0)
    print(f"GPU->GPU copy : {d2d:>8.0f} GB/s")
    t0 = time.time()
    c = buf.to("cpu")
    torch.cuda.synchronize()
    h2d_time = time.time() - t0
    print(f"GPU->CPU      : {2 * n / 1e9 / h2d_time:>8.0f} GB/s")
    t0 = time.time()
    g = c.to(DEV)
    torch.cuda.synchronize()
    d2h_time = time.time() - t0
    print(f"CPU->GPU      : {2 * n / 1e9 / d2h_time:>8.0f} GB/s")
    swap_gb = 14.3
    print(f"est. full 14.3GB transformer swap (CPU->GPU then GPU->CPU): "
          f"{swap_gb / (2 * n / 1e9 / h2d_time) + swap_gb / (2 * n / 1e9 / d2h_time):.2f} s")
    print("(your log shows ~13s/step overhead that is NOT this swap - it recurs within a resident phase)")
    del buf, c, g


def projection(results):
    header("PROJECTED WAN 14.3B TRANSFORMER FORWARD (weight-FLOPs, attention extra ~5-15%)")
    print(f"{'config':<28}" + "".join(f"{t:>16}" for t in TOKENS))
    names = ["BF16", "FP16"]
    for n in results:
        if n not in ["BF16", "FP16"]:
            names.append(n)
    for name in names:
        row = results.get(name)
        if not row:
            continue
        tflops = max(row.values())
        cells = []
        for label, T in TOKENS.items():
            flops = 2 * WAN_PARAMS * T
            cells.append(f"{flops / tflops / 1e12:>16.1f}s")
        print(f"{name:<28}" + "".join(cells))
    print("\nreference: your 896^2 @ 105f run measured ~75.6s per forward -> ~128 TFLOPS effective")


def summary(results):
    header("SUMMARY - best TFLOPS per config")
    rows = []
    for name, d in results.items():
        if not d:
            continue
        best = max(d.values())
        at = max(d, key=d.get)
        rows.append((best, name, at))
    rows.sort(reverse=True)
    print(f"{'TFLOPS':>8}  {'config':<28}  best-at")
    for tf, name, at in rows:
        print(f"{tf:>8.1f}  {name:<28}  {at}")


if __name__ == "__main__":
    report_system()
    baseline = sweep_baseline()
    torchao_results = sweep_torchao()
    results = {**baseline, **torchao_results}
    sweep_sdpa()
    bench_bandwidth()
    projection(results)
    summary(results)
    print("\nDONE")
