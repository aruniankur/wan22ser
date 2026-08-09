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

# Wan 2.2 14B geometry: hidden=5120, ffn=13824, heads=40, layers=40, patch=(1,2,2)
# tokens after VAE(4x temporal, 8x spatial) + patch(2x spatial)
#  640x640@105f -> 27*40*40 = 43,200
#  896x896@105f -> 27*56*56 = 84,672
# 1024x1024@105f -> 27*64*64 = 110,592
MATMUL_SIZES = [
    ("attn-4K",    4096,   5120,  5120),
    ("attn-16K",   16384,  5120,  5120),
    ("attn-43K-640", 43200, 5120, 5120),
    ("attn-85K-896", 84672, 5120, 5120),
    ("attn-111K-1024", 110592, 5120, 5120),
    ("ffn-85K-896", 84672, 5120, 13824),
]

WAN_PARAMS = 14.29e9
WAN_LAYERS = 40
WAN_HEAD_DIM = 5120
PATCHED_TOKENS = {
    "640^2@105f": 43200,
    "896^2@105f": 84672,
    "1024^2@105f": 110592,
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
    print("cudnn-sdp :", torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else "n/a")
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
    print(f"{'size':<16}{'FP32':>10}{'FP16':>10}{'BF16':>10}")
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
        print(f"{row[0]:<16}{row[1]:>10.1f}{row[2]:>10.1f}{row[3]:>10.1f}", flush=True)
    return results


def sweep_torchao(baseline):
    from torchao.quantization import (
        quantize_,
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int4WeightOnlyConfig,
        Float8WeightOnlyConfig,
        Float8DynamicActivationFloat8WeightConfig,
    )
    configs = [
        ("FP8-dyn (app transformer)", Float8DynamicActivationFloat8WeightConfig()),
        ("FP8-wt-only", Float8WeightOnlyConfig()),
        ("INT8-wt (app text_enc)", Int8WeightOnlyConfig()),
        ("INT8-dyn", Int8DynamicActivationInt8WeightConfig()),
        ("INT4-g128", Int4WeightOnlyConfig(group_size=128)),
    ]
    results = {}
    for name, cfg in configs:
        header("TORCHAO: " + name)
        print(f"{'size':<16}{'ms':>10}{'TFLOPS':>10}{'vs BF16':>9}{'mem':>8}")
        for label, m, k, n in MATMUL_SIZES:
            flops = 2.0 * m * k * n
            iters, warm = auto_iters(m, k, n), auto_warm(m, k, n)
            try:
                lin = Linear(k, n, bias=False, device=DEV, dtype=torch.bfloat16)
                quantize_(lin, cfg)
                torch._dynamo.reset()
                x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
                torch.cuda.reset_peak_memory_stats()
                ms = bench_linear(lin, x, iters, warm)
                tflops = flops / (ms / 1e3) / 1e12
                mem = torch.cuda.max_memory_allocated() / 2**30
                base = baseline.get("BF16", {}).get(label)
                vs = tflops / base if base else float("nan")
                results.setdefault(name, {})[label] = tflops
                print(f"{label:<16}{ms:>10.3f}{tflops:>10.1f}{vs:>9.2f}x{mem:>8.1f}", flush=True)
            except Exception as err:
                print(f"{label:<16}FAILED: {type(err).__name__}: {err}", flush=True)
            torch.cuda.empty_cache()
    return results


def time_sdpa(q, k, v, backend, iters, warm=1):
    from torch.nn.attention import sdpa_kernel
    fn = lambda: F.scaled_dot_product_attention(q, k, v)
    with sdpa_kernel(backend):
        for _ in range(warm):
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
    return start.elapsed_time(end) / iters


def sweep_sdpa():
    from torch.nn.attention import SDPBackend
    header("SDPA ATTENTION (84,672 = real patched seq at 896^2@105f)")
    sdpa_tflops = {}
    H, HD = 40, 128
    cudnn_backend = getattr(SDPBackend, "CUDNN_ATTENTION", None)
    backends = [("mem_eff", SDPBackend.EFFICIENT_ATTENTION)]
    if cudnn_backend is not None:
        backends.append(("cudnn", cudnn_backend))
    if torch.backends.cuda.flash_sdp_enabled():
        backends.append(("flash", SDPBackend.FLASH_ATTENTION))
    for L in [4096, 43200, 84672, 110592]:
        for dt in [torch.bfloat16, torch.float16]:
            q = torch.randn(1, H, L, HD, device=DEV, dtype=dt)
            k = torch.randn(1, H, L, HD, device=DEV, dtype=dt)
            v = torch.randn(1, H, L, HD, device=DEV, dtype=dt)
            flops = 2.0 * L * L * H * HD
            iters = 2 if L > 50000 else 5
            print(f"\nseq={L:<8} dtype={str(dt).replace('torch.',''):<9} FLOPs/iter={flops/1e12:.2f} TFLOP")
            for bname, backend in backends:
                try:
                    ms = time_sdpa(q, k, v, backend, iters)
                    tf = flops / (ms / 1e3) / 1e12
                    sdpa_tflops.setdefault(bname, {})[L] = tf
                    print(f"  {bname:<10} {ms:>9.2f} ms  {tf:>8.1f} TFLOPS", flush=True)
                except Exception as err:
                    print(f"  {bname:<10} FAILED: {str(err)[:90]}", flush=True)
            del q, k, v
            torch.cuda.empty_cache()
    if cudnn_backend is not None and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        header("CUDA GRAPHS-FREE EXPERIMENT: enable_cudnn_sdp(True) re-test")
        torch.backends.cuda.enable_cudnn_sdp(True)
        print("cudnn-sdp now:", torch.backends.cuda.cudnn_sdp_enabled(), flush=True)
        for L in [84672, 110592]:
            for dt in [torch.bfloat16, torch.float16]:
                q = torch.randn(1, H, L, HD, device=DEV, dtype=dt)
                k = torch.randn(1, H, L, HD, device=DEV, dtype=dt)
                v = torch.randn(1, H, L, HD, device=DEV, dtype=dt)
                flops = 2.0 * L * L * H * HD
                try:
                    ms = time_sdpa(q, k, v, cudnn_backend, 2)
                    tf = flops / (ms / 1e3) / 1e12
                    sdpa_tflops.setdefault("cudnn", {})[L] = tf
                    print(f"seq={L} dtype={str(dt).replace('torch.','')} cudnn {ms:>9.2f} ms  {tf:>8.1f} TFLOPS", flush=True)
                except Exception as err:
                    print(f"seq={L} dtype={str(dt).replace('torch.','')} cudnn FAILED: {str(err)[:120]}", flush=True)
                del q, k, v
                torch.cuda.empty_cache()
    return sdpa_tflops


class ProxyWanBlock(torch.nn.Module):
    def __init__(self, dim=5120, ffn=13824, heads=40):
        super().__init__()
        self.heads = heads
        self.norm1 = torch.nn.RMSNorm(dim, eps=1e-6)
        self.to_q = Linear(dim, dim, bias=True)
        self.to_k = Linear(dim, dim, bias=True)
        self.to_v = Linear(dim, dim, bias=True)
        self.to_out = Linear(dim, dim, bias=True)
        self.ffn_up = Linear(dim, ffn, bias=True)
        self.ffn_down = Linear(ffn, dim, bias=True)

    def forward(self, x):
        h = self.norm1(x)
        q = self.to_q(h).unflatten(2, (self.heads, -1)).transpose(1, 2)
        k = self.to_k(h).unflatten(2, (self.heads, -1)).transpose(1, 2)
        v = self.to_v(h).unflatten(2, (self.heads, -1)).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        x = x + self.to_out(attn)
        y = self.ffn_down(F.silu(self.ffn_up(x)))
        return x + y


def proxy_block_test():
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    header("REAL-SHAPE PROXY: 1 Wan-like block @ 84,672 tokens (fp8-dyn vs bf16)")
    L = 84672
    x = torch.randn(1, L, 5120, device=DEV, dtype=torch.bfloat16)
    for name in ["BF16 (no quant)", "FP8-dyn"]:
        model = ProxyWanBlock().to(DEV).to(torch.bfloat16)
        if name.startswith("FP8"):
            quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        fn = lambda: model(x)
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_block = start.elapsed_time(end) / 3
        n_params = sum(p.numel() for p in model.parameters())
        weight_flops = 2 * n_params * L
        print(f"{name:<20} {per_block:>7.1f} ms/block | {per_block/1000*40:>7.1f} s projected x40 blocks | "
              f"{weight_flops/(per_block/1e3)/1e12:>6.1f} TFLOPS", flush=True)
        del model
        torch.cuda.empty_cache()
    print("\nreference: your app measures ~75.6s per real forward (fp8-dyn, UniPC 6 steps)")
    print("          proxy x40 should land in the same ballpark if the microbench is representative")


def bench_bandwidth():
    header("MEMORY BANDWIDTH + CPU-OFFLOAD SWAP COST")
    n = 2**30
    buf = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        _ = buf.clone()
    end.record()
    torch.cuda.synchronize()
    d2d_s = start.elapsed_time(end) / 1000 / 10
    print(f"GPU->GPU copy : {2 * 2 * n / 1e9 / d2d_s:>8.0f} GB/s")
    t0 = time.perf_counter()
    c = buf.to("cpu")
    torch.cuda.synchronize()
    d2h_s = time.perf_counter() - t0
    print(f"GPU->CPU      : {2 * n / 1e9 / d2h_s:>8.0f} GB/s")
    t0 = time.perf_counter()
    g = c.to(DEV)
    torch.cuda.synchronize()
    h2d_s = time.perf_counter() - t0
    print(f"CPU->GPU      : {2 * n / 1e9 / h2d_s:>8.0f} GB/s")
    swap_gb = 14.3
    print(f"est. full 14.3GB transformer swap (CPU->GPU then GPU->CPU): "
          f"{swap_gb / (2 * n / 1e9 / d2h_s) + swap_gb / (2 * n / 1e9 / h2d_s):.2f} s")
    del buf, c, g


def projection(results, sdpa_tflops):
    header("PROJECTED REAL FORWARD 896^2@105f (weights + full attention, 40 blocks)")
    gemm = results.get("FP8-dyn (app transformer)", {})
    gemm_tf = max(gemm.values()) if gemm else 128.0
    attn_tf = 46.0
    for bname in ["mem_eff", "cudnn"]:
        d = sdpa_tflops.get(bname, {})
        if d and 84672 in d:
            attn_tf = max(attn_tf, d[84672])
            print(f"using SDPA backend '{bname}' @ {d[84672]:.0f} TFLOPS for attention")
            break
    seq = PATCHED_TOKENS["896^2@105f"]
    w_time = 2 * WAN_PARAMS * seq / gemm_tf / 1e12
    a_time = WAN_LAYERS * 2 * seq * seq * WAN_HEAD_DIM / attn_tf / 1e12
    print(f"weights (fp8-dyn {gemm_tf:.0f} TFLOPS) : {w_time:>6.1f} s")
    print(f"attention (SDPA {attn_tf:.0f} TFLOPS)  : {a_time:>6.1f} s")
    print(f"projected forward                    : {w_time + a_time:>6.1f} s  (observed: 75.6s)")
    print(f"\nattention share of forward: {a_time/(w_time+a_time)*100:.0f}% -> attention backend is the lever")


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
    torchao_results = sweep_torchao(baseline)
    results = {**baseline, **torchao_results}
    sdpa_tflops = sweep_sdpa()
    proxy_block_test()
    bench_bandwidth()
    projection(results, sdpa_tflops)
    summary(results)
    print("\nDONE")
