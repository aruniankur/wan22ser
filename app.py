import os
import copy
import random
import tempfile
import warnings
import time
import gc
import threading
import uuid
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch._dynamo
from torch.nn import functional as F
from PIL import Image

# --- CUDNN ATTENTION BACKEND (must be set BEFORE diffusers is imported) ---
# diffusers 0.39 routes Wan attention through dispatch_attention_fn and reads
# DIFFUSERS_ATTN_BACKEND at import time. `_native_cudnn` wraps SDPA in
# sdpa_kernel(CUDNN_ATTENTION) -> ~74 TFLOPS vs ~49 mem_eff on the L40.
# Disable with CUDNN_ATTN=0 to revert to default dispatch.
if os.environ.get("CUDNN_ATTN", "1") == "1" and torch.cuda.is_available():
    os.environ["DIFFUSERS_ATTN_BACKEND"] = "_native_cudnn"
    print("[DEBUG] attention backend -> _native_cudnn (cuDNN SDPA)", flush=True)
    
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import gradio as gr
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SASolverScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
)
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.utils.export_utils import export_to_video

from torchao.quantization import quantize_, Int8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig
import lora_loader

# Compatibility shim: dev versions of diffusers define TorchaoLoraLinear with a
# required 'get_apply_tensor_subclass' kwarg that older torchao releases lack.
try:
    from diffusers.models.lora import TorchaoLoraLinear
    _orig_init = TorchaoLoraLinear.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault('get_apply_tensor_subclass', None)
        return _orig_init(self, *args, **kwargs)
    TorchaoLoraLinear.__init__ = _patched_init
    print("[DEBUG] Applied TorchaoLoraLinear compatibility shim", flush=True)
except ImportError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")

print(f"[DEBUG] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)} | VRAM total={props.total_memory / 2**30:.1f} GiB", flush=True)
else:
    print("[DEBUG] CUDA NOT AVAILABLE - will fall back to CPU (very slow / may fail)", flush=True)

# --- FRAME EXTRACTION JS & LOGIC ---

# JS to grab timestamp from the output video
get_timestamp_js = """
function() {
    // Select the video element specifically inside the component with id 'generated-video'
    const video = document.querySelector('#generated-video video');
    
    if (video) {
        console.log("Video found! Time: " + video.currentTime);
        return video.currentTime;
    } else {
        console.log("No video element found.");
        return 0;
    }
}
"""


def extract_frame(video_path, timestamp):
    # Safety check: if no video is present
    if not video_path:
        return None
    
    print(f"Extracting frame at timestamp: {timestamp}") 
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None

    # Calculate frame number
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_num = int(float(timestamp) * fps)
    
    # Cap total frames to prevent errors at the very end of video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_frame_num >= total_frames:
        target_frame_num = total_frames - 1
    
    # Set position
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert from BGR (OpenCV) to RGB (Gradio)
        # Gradio Image component handles Numpy array -> PIL conversion automatically
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return None

# --- END FRAME EXTRACTION LOGIC ---


def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()


def gpu_mem_gib():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
    return 0.0, 0.0


def ram_mem_gib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return 0.0


# --- MEMORY INSTRUMENTATION (always on) ---

RAM_PEAK_GIB = 0.0
_RAM_WATCHER_STARTED = False
PHASE_STATS = {}  # component name -> max VRAM peak (GiB) seen during its forward calls


def _ram_watcher():
    global RAM_PEAK_GIB
    while True:
        try:
            rss = ram_mem_gib()
            if rss > RAM_PEAK_GIB:
                RAM_PEAK_GIB = rss
        except Exception:
            pass
        time.sleep(0.2)


def start_ram_watcher():
    global _RAM_WATCHER_STARTED
    if not _RAM_WATCHER_STARTED:
        _RAM_WATCHER_STARTED = True
        threading.Thread(target=_ram_watcher, daemon=True).start()
        print("[DEBUG] RAM peak watcher started (200ms poll)", flush=True)


def dbg(label, t0=None):
    alloc, reserved = gpu_mem_gib()
    elapsed = f" (+{time.time() - t0:.1f}s)" if t0 else ""
    print(
        f"[DEBUG] {label}{elapsed} | GPU alloc={alloc:.2f} GiB reserved={reserved:.2f} GiB | RAM={ram_mem_gib():.2f} GiB | RAMpeak={RAM_PEAK_GIB:.2f} GiB",
        flush=True,
    )


def dbg_peak(label, t0=None):
    alloc, reserved = gpu_mem_gib()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    elapsed = f" (+{time.time() - t0:.1f}s)" if t0 else ""
    print(
        f"[DEBUG] {label}{elapsed} | GPU alloc={alloc:.2f} GiB reserved={reserved:.2f} GiB | PEAK-since-reset={peak:.2f} GiB | RAM={ram_mem_gib():.2f} GiB | RAMpeak={RAM_PEAK_GIB:.2f} GiB",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _make_phase_hooks(name):
    state = {"t0": None, "base_ram": None, "peak": 0.0}

    def pre_hook(module, args):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        state["t0"] = time.time()
        state["base_ram"] = ram_mem_gib()
        alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(
            f"[PHASE] >>> {name} START | GPU alloc={alloc:.2f} GiB | RAM={state['base_ram']:.2f} GiB | RAMpeak={RAM_PEAK_GIB:.2f} GiB",
            flush=True,
        )

    def post_hook(module, args, output):
        dt = time.time() - state["t0"]
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        state["peak"] = max(state["peak"], peak)
        PHASE_STATS[name] = state["peak"]
        print(
            f"[PHASE] <<< {name} END +{dt:.1f}s | PEAK VRAM={peak:.2f} GiB | GPU alloc={alloc:.2f} GiB | RAM={ram_mem_gib():.2f} GiB | RAMpeak={RAM_PEAK_GIB:.2f} GiB",
            flush=True,
        )

    return pre_hook, post_hook


def _make_method_phase_hook(name, obj, method_name):
    # VAE decode/encode are methods (not forward), so wrap them directly.
    state = {"t0": None, "peak": 0.0}
    orig = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        state["t0"] = time.time()
        alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(
            f"[PHASE] >>> {name}.{method_name} START | GPU alloc={alloc:.2f} GiB | RAM={ram_mem_gib():.2f} GiB | RAMpeak={RAM_PEAK_GIB:.2f} GiB",
            flush=True,
        )
        out = orig(*args, **kwargs)
        dt = time.time() - state["t0"]
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        state["peak"] = max(state["peak"], peak)
        PHASE_STATS[name] = state["peak"]
        print(
            f"[PHASE] <<< {name}.{method_name} END +{dt:.1f}s | PEAK VRAM={peak:.2f} GiB | RAM={ram_mem_gib():.2f} GiB | RAMpeak={RAM_PEAK_GIB:.2f} GiB",
            flush=True,
        )
        return out

    setattr(obj, method_name, wrapped)
    return wrapped


def register_phase_hooks(pipe):
    for name in ["text_encoder", "transformer", "transformer_2"]:
        comp = getattr(pipe, name, None)
        if comp is None:
            continue
        pre_hook, post_hook = _make_phase_hooks(name)
        comp.register_forward_pre_hook(pre_hook)
        comp.register_forward_hook(post_hook)
        print(f"[DEBUG] phase hook registered: {name}", flush=True)
    if getattr(pipe, "vae", None) is not None:
        for m in ["encode", "decode"]:
            if hasattr(pipe.vae, m):
                _make_method_phase_hook("vae", pipe.vae, m)
                print(f"[DEBUG] phase hook registered: vae.{m}", flush=True)


def _make_step_cb(total_steps, t_pipe):
    last = {"t": t_pipe}

    def step_cb(pipe, step, timestep, callback_kwargs):
        now = time.time()
        dt = now - last["t"]
        last["t"] = now
        alloc, reserved = gpu_mem_gib()
        print(
            f"[STEP] {step + 1}/{total_steps} +{dt:.1f}s | GPU alloc={alloc:.2f} GiB reserved={reserved:.2f} GiB | RAM={ram_mem_gib():.2f} GiB | RAMpeak={RAM_PEAK_GIB:.2f} GiB",
            flush=True,
        )
        return callback_kwargs

    return step_cb


# RIFE
if not os.path.exists("RIFEv4.26_0921.zip"):
    print("Downloading RIFE Model...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://huggingface.co/thornmaze/RIFE/resolve/main/RIFEv4.26_0921.zip",
        "RIFEv4.26_0921.zip"
    )
    import zipfile
    with zipfile.ZipFile("RIFEv4.26_0921.zip") as z:
        z.extractall(".")
    print("RIFE Model extracted.")

start_ram_watcher()

# sys.path.append(os.getcwd())

from train_log.RIFE_HDv3 import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] RIFE device: {device}", flush=True)
rife_model = Model()
rife_model.load_model("train_log", -1)
rife_model.eval()
dbg("RIFE model loaded")


@torch.no_grad()
def interpolate_bits(frames_np, multiplier=2, scale=1.0):
    """
    Interpolation maintaining Numpy Float 0-1 format.
    Args:
        frames_np: Numpy Array (Time, Height, Width, Channels) - Float32 [0.0, 1.0]
        multiplier: int (2, 4, 8)
    Returns:
        List of Numpy Arrays (Height, Width, Channels) - Float32 [0.0, 1.0]
    """
    
    # Handle input shape
    if isinstance(frames_np, list):
        # Convert list of arrays to one big array for easier shape handling if needed, 
        # but here we just grab dims from first frame
        T = len(frames_np)
        H, W, C = frames_np[0].shape
    else:
        T, H, W, C = frames_np.shape

    # 1. No Interpolation Case
    if multiplier < 2:
        # Just convert 4D array to list of 3D arrays
        if isinstance(frames_np, np.ndarray):
            return list(frames_np)
        return frames_np

    n_interp = multiplier - 1
    
    # Pre-calc padding for RIFE (requires dimensions divisible by 32/scale)
    tmp = max(128, int(128 / scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    padding = (0, pw - W, 0, ph - H)

    # Helper: Numpy (H, W, C) Float -> Tensor (1, C, H, W) Half
    def to_tensor(frame_np):
        # frame_np is float32 0-1
        t = torch.from_numpy(frame_np).to(device)
        # HWC -> CHW
        t = t.permute(2, 0, 1).unsqueeze(0)
        return F.pad(t, padding).half()

    # Helper: Tensor (1, C, H, W) Half -> Numpy (H, W, C) Float
    def from_tensor(tensor):
        # Crop padding
        t = tensor[0, :, :H, :W]
        # CHW -> HWC
        t = t.permute(1, 2, 0)
        # Keep as float32, range 0-1
        return t.float().cpu().numpy()

    def make_inference(I0, I1, n):
        if rife_model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(rife_model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
            return res
        else:
            middle = rife_model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = make_inference(I0, middle, n=n//2)
            second_half = make_inference(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    output_frames = []

    # Process Frames
    # Load first frame into GPU
    I1 = to_tensor(frames_np[0])

    total_steps = T - 1

    with tqdm(total=total_steps, desc="Interpolating", unit="frame") as pbar:
    
        for i in range(total_steps):
            I0 = I1
            # Add original frame to output
            output_frames.append(from_tensor(I0))
    
            # Load next frame
            I1 = to_tensor(frames_np[i+1])
    
            # Generate intermediate frames
            mid_tensors = make_inference(I0, I1, n_interp)
    
            # Append intermediate frames
            for mid in mid_tensors:
                output_frames.append(from_tensor(mid))

            if (i + 1) % 50 == 0:
                pbar.update(50)
        pbar.update(total_steps % 50)
        
        # Add the very last frame
        output_frames.append(from_tensor(I1))
    
    # Cleanup
    del I0, I1, mid_tensors
    torch.cuda.empty_cache()

    return output_frames


# WAN

MODEL_ID = "thornmaze/WAMU_v3_WAN2.2_I2V_LIGHTNING"

LORA_MODELS = []

MAX_DIM = 1500
MIN_DIM = 480
MULTIPLE_OF = 16
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 640
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 321

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

SCHEDULER_MAP = {
    "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
    "SASolver": SASolverScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

# --- BF16 NORMS PATCH (remove fp32 activation upcasts) ---
# Wan 2.2's transformer upcasts the hidden state to fp32 at every norm and
# residual-add (40 blocks x ~6 copies) -> ~40GB peak activation memory.
# This patch runs those ops in the input dtype (bf16) instead. Disable with BF16_NORMS=0.
BF16_NORMS = os.environ.get("BF16_NORMS", "1") == "1"

if BF16_NORMS:
    try:
        import inspect
        import textwrap
        from diffusers.models.normalization import FP32LayerNorm
        from diffusers.models.transformers import transformer_wan as _tw
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock

        # 1) LayerNorm: run in input dtype instead of upcasting to fp32.
        def _bf16_layer_norm_forward(self, inputs):
            origin_dtype = inputs.dtype
            return F.layer_norm(
                inputs,
                self.normalized_shape,
                self.weight.to(origin_dtype) if self.weight is not None else None,
                self.bias.to(origin_dtype) if self.bias is not None else None,
                self.eps,
            )

        # 2/3) Transformer block + main forward: rewrite the INSTALLED source so the
        # patch matches whatever diffusers version is present (0.35.x ... 0.38.x).
        # Only the fp32 upcasts are removed; the re-executed body is the installed
        # body verbatim (incl. the @apply_lora_scale decorator in >=0.36), resolved
        # in the module's own namespace -> no hardcoded module-level names.
        def _bf16_rewrite(cls, method, replacements):
            src = textwrap.dedent(inspect.getsource(getattr(cls, method)))
            applied = []
            for old, new in replacements:
                if old in src:
                    src = src.replace(old, new)
                    applied.append(old)
            if not applied:
                return []
            exec(compile(src, f"<patched {cls.__name__}.{method}>", "exec"), _tw.__dict__)
            setattr(cls, method, _tw.__dict__[method])
            return applied

        _bf16_block_rewrites = [
            ("temb.float()", "temb.to(hidden_states.dtype)"),
            ("self.norm1(hidden_states.float())", "self.norm1(hidden_states)"),
            ("self.norm2(hidden_states.float())", "self.norm2(hidden_states)"),
            ("self.norm3(hidden_states.float())", "self.norm3(hidden_states)"),
            (
                "hidden_states.float() + attn_output",
                "hidden_states + attn_output.to(hidden_states.dtype)",
            ),
            (
                "hidden_states = hidden_states + attn_output",
                "hidden_states = hidden_states + attn_output.to(hidden_states.dtype)",
            ),
            (
                "hidden_states.float() + ff_output.float()",
                "hidden_states + ff_output.to(hidden_states.dtype)",
            ),
        ]
        _bf16_tail_rewrites = [
            ("norm_out(hidden_states.float())", "norm_out(hidden_states)"),
        ]

        patched = []

        if "inputs.float()" in inspect.getsource(FP32LayerNorm.forward):
            FP32LayerNorm.forward = _bf16_layer_norm_forward
            patched.append("FP32LayerNorm -> bf16")
        else:
            print("[DEBUG] BF16 norms patch: FP32LayerNorm already runs in input dtype - skipping", flush=True)

        _applied = _bf16_rewrite(WanTransformerBlock, "forward", _bf16_block_rewrites)
        if _applied:
            patched.append(f"WanTransformerBlock residuals -> bf16 ({len(_applied)}/{len(_bf16_block_rewrites)})")
        else:
            print("[DEBUG] BF16 norms patch: WanTransformerBlock.forward: no upcast patterns matched - skipping", flush=True)

        _applied = _bf16_rewrite(WanTransformer3DModel, "forward", _bf16_tail_rewrites)
        if _applied:
            patched.append("WanTransformer3DModel final norm -> bf16")
        else:
            print("[DEBUG] BF16 norms patch: WanTransformer3DModel.forward: no upcast patterns matched - skipping", flush=True)

        print(f"[DEBUG] BF16 norms patch: ENABLED ({', '.join(patched)}) - set BF16_NORMS=0 to disable", flush=True)
    except Exception as _bf16_err:
        print(f"[DEBUG] BF16 norms patch: FAILED ({_bf16_err}) - falling back to fp32 norms", flush=True)

# --- END BF16 NORMS PATCH ---

print(f"[DEBUG] Loading pipeline {MODEL_ID} (bf16, CPU)...", flush=True)
t0 = time.time()
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
dbg("pipeline loaded", t0)

try:
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    _ab_name, _ab_fn = _AttentionBackendRegistry.get_active_backend()
    print(f"[DEBUG] active diffusers attention backend = {_ab_name}", flush=True)
except Exception as _ab_err:
    print(f"[DEBUG] could not read attention backend: {_ab_err}", flush=True)

for name in ["text_encoder", "transformer", "transformer_2", "vae"]:
    comp = getattr(pipe, name, None)
    if comp is not None:
        nparams = sum(p.numel() for p in comp.parameters())
        print(f"[DEBUG] component {name}: params={nparams / 1e9:.2f}B device={next(comp.parameters()).device}", flush=True)

t1 = time.time()
quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
torch._dynamo.reset()
dbg("text_encoder quantized (int8)", t1)


fp8_config = Float8DynamicActivationFloat8WeightConfig()

t1 = time.time()
quantize_(pipe.transformer, fp8_config)
torch._dynamo.reset()
dbg("transformer quantized (int8)", t1)

t1 = time.time()
quantize_(pipe.transformer_2, fp8_config)
torch._dynamo.reset()
dbg("transformer_2 quantized (int8)", t1)

t1 = time.time()
pipe.enable_model_cpu_offload()
dbg("model CPU offload enabled (components stream to GPU per phase)", t1)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
dbg("VAE tiling/slicing enabled")

original_scheduler = copy.deepcopy(pipe.scheduler)
register_phase_hooks(pipe)
dbg("model load complete")

for i, lora in enumerate(LORA_MODELS):
    name_high_tr = lora["high_tr"].split(".")[0].split("/")[-1] + "Hh"
    name_low_tr = lora["low_tr"].split(".")[0].split("/")[-1] + "Ll"
    
    try: 
        pipe.load_lora_weights(
            lora["repo_id"],
            weight_name=lora["high_tr"],
            adapter_name=name_high_tr
        )
    
        kwargs_lora = {"load_into_transformer_2": True}
        pipe.load_lora_weights(
            lora["repo_id"],
            weight_name=lora["low_tr"],
            adapter_name=name_low_tr,
            **kwargs_lora
        )
    
        pipe.set_adapters([name_high_tr, name_low_tr], adapter_weights=[1.0, 1.0])
    
        pipe.fuse_lora(adapter_names=[name_high_tr], lora_scale=lora["high_scale"], components=["transformer"])
        pipe.fuse_lora(adapter_names=[name_low_tr], lora_scale=lora["low_scale"], components=["transformer_2"])
    
        pipe.unload_lora_weights()

        print(f"Applied: {lora['high_tr']}, hs={lora['high_scale']}/ls={lora['low_scale']}, {i+1}/{len(LORA_MODELS)}") 
    except Exception as e:
        print("Error:", str(e))
        print("Failed LoRA:", name_high_tr)
        pipe.unload_lora_weights()

# if os.path.exists(CACHE_DIR):
#     shutil.rmtree(CACHE_DIR)
#     print("Deleted Hugging Face cache.")
# else:
#     print("No hub cache found.")

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"


def model_title():
    return "## Wan 2.2 I2V 14B Lightning — NSFW"


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    target_width, target_height = image.size
    scale = max(width / target_width, height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    left, top = (new_width - width) // 2, (new_height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def resize_and_crop_to_match(target_image, reference_image):
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def get_num_frames(duration_seconds: float):
    raw = int(round(duration_seconds * FIXED_FPS))
    raw = max(MIN_FRAMES_MODEL, min(MAX_FRAMES_MODEL, raw))
    return ((raw - 1) // 4) * 4 + 1


def clamp_dim(value: int) -> int:
    value = round(value / MULTIPLE_OF) * MULTIPLE_OF
    return max(MIN_DIM, min(MAX_DIM, value))


def compute_tokens_text(width: int, height: int, duration_seconds: float) -> str:
    num_frames = get_num_frames(duration_seconds)
    latent_frames = (num_frames - 1) // 4 + 1
    lw, lh = width // 8, height // 8
    tokens = latent_frames * lw * lh
    if tokens >= 1_000_000:
        tokens_str = f"{tokens / 1e6:.1f}M"
    elif tokens >= 1_000:
        tokens_str = f"{tokens / 1e3:.0f}K"
    else:
        tokens_str = str(tokens)
    return (
        f"**Estimated tokens:** {tokens_str} ({tokens:,}) "
        f"= {latent_frames} latent frames × {lw} × {lh}"
    )


def on_image_upload(image, user_override, duration_seconds):
    if image is None:
        return gr.update(), gr.update(), "", user_override, ""
    width, height = image.size
    auto_w, auto_h = clamp_dim(width), clamp_dim(height)
    size_text = f"**Image size:** {width} × {height} px"
    if (auto_w, auto_h) != (width, height):
        size_text += f"  →  Output: {auto_w} × {auto_h} px"
    if user_override:
        return gr.update(), gr.update(), size_text, user_override, compute_tokens_text(auto_w, auto_h, duration_seconds)
    return auto_w, auto_h, size_text, False, compute_tokens_text(auto_w, auto_h, duration_seconds)


def mark_user_override():
    return True


def run_inference(
    resized_image,
    processed_last_image,
    prompt,
    steps,
    negative_prompt,
    num_frames,
    guidance_scale,
    guidance_scale_2,
    current_seed,
    scheduler_name,
    flow_shift,
    frame_multiplier,
    quality,
    duration_seconds,
    safe_mode=False,
    lora_groups=None,
    progress=gr.Progress(track_tqdm=True),
):
    scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if scheduler_class.__name__ != pipe.scheduler.config._class_name or flow_shift != pipe.scheduler.config.get("flow_shift", "shift"):
        config = copy.deepcopy(original_scheduler.config)
        if scheduler_class == FlowMatchEulerDiscreteScheduler:
            config['shift'] = flow_shift
        else:
            config['flow_shift'] = flow_shift
        pipe.scheduler = scheduler_class.from_config(config)

    clear_vram()

    task_name = str(uuid.uuid4())[:8]
    print(f"Generating {num_frames} frames, task: {task_name}, {duration_seconds}, {resized_image.size}, lora={lora_groups}")
    dbg(f"run_inference start (steps={steps}, guidance={guidance_scale}/{guidance_scale_2}, frames={num_frames}, scheduler={scheduler_name})")
    start = time.time()
    PHASE_STATS.clear()

    lora_loaded = False
    if lora_groups:
        try:
            for idx, name in enumerate(lora_groups):
                if name and name != "(None)":
                    lora_loader.load_lora_to_pipe(pipe, name, adapter_name=f"lora_{idx}")
            lora_loaded = True
            print(f"LoRA loaded: {lora_groups}")
        except Exception as e:
            print(f"LoRA warning: {e}")
            pipe.unload_lora_weights()
            lora_loaded = False

    try:
        t_pipe = time.time()
        pipe_cb = _make_step_cb(int(steps), t_pipe)
        result = pipe(
            image=resized_image,
            last_image=processed_last_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=resized_image.height,
            width=resized_image.width,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            guidance_scale_2=float(guidance_scale_2),
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed),
            callback_on_step_end=pipe_cb,
            output_type="np"
        )
    except RuntimeError as e:
        if "cudnn" in str(e).lower() or "no available kernel" in str(e).lower():
            print(f"[DEBUG] cuDNN attention failed, retrying once with native backend: {e}", flush=True)
            from diffusers.models.attention_dispatch import _AttentionBackendRegistry
            _AttentionBackendRegistry.set_active_backend("native")
            torch._dynamo.reset()
            if lora_loaded:
                lora_loader.unload_lora(pipe)
            pipe.scheduler = original_scheduler
            clear_vram()
            t_pipe = time.time()
            pipe_cb = _make_step_cb(int(steps), t_pipe)
            result = pipe(
                image=resized_image,
                last_image=processed_last_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=resized_image.height,
                width=resized_image.width,
                num_frames=num_frames,
                guidance_scale=float(guidance_scale),
                guidance_scale_2=float(guidance_scale_2),
                num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(current_seed),
                callback_on_step_end=pipe_cb,
                output_type="np"
            )
            print("[DEBUG] retry with native backend succeeded", flush=True)
        else:
            raise
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}")
        dbg("OOM caught - GPU state before cleanup")
        if lora_loaded:
            lora_loader.unload_lora(pipe)
        pipe.scheduler = original_scheduler
        clear_vram()
        dbg("OOM cleanup done")
        raise gr.Error(
            "Out of GPU memory. Lower the duration/resolution or set both guidance scales to 1.0, then retry."
        )

    if lora_loaded:
        lora_loader.unload_lora(pipe)

    if PHASE_STATS:
        phases = ", ".join(f"{k}={v:.2f}GiB" for k, v in PHASE_STATS.items() if v > 0)
        overall = max(PHASE_STATS.values())
        print(f"[DEBUG] PHASE peak VRAM | {phases} | overall={overall:.2f} GiB", flush=True)
    dbg("inference finished", t_pipe)
    print("gen time passed:", time.time() - start)

    raw_frames_np = result.frames[0]  # Returns (T, H, W, C) float32
    pipe.scheduler = original_scheduler

    frame_factor = frame_multiplier // FIXED_FPS
    if frame_factor > 1:
        start = time.time()
        print(f"Processing frames (RIFE Multiplier: {frame_factor}x)...")
        rife_model.device()
        rife_model.flownet = rife_model.flownet.half()
        final_frames = interpolate_bits(raw_frames_np, multiplier=int(frame_factor))
        print("Interpolation time passed:", time.time() - start)
    else:
        final_frames = list(raw_frames_np)
    dbg_peak("frame processing done")

    final_fps = FIXED_FPS * int(frame_factor)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

    start = time.time()
    with tqdm(total=3, desc="Rendering Media", unit="clip") as pbar:
        pbar.update(2)
        export_to_video(final_frames, video_path, fps=final_fps, quality=quality)
        pbar.update(1)
    print(f"Export time passed, {final_fps} FPS:", time.time() - start)
    dbg_peak("export done")

    return video_path, task_name


def generate_video(
    input_image,
    last_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    quality=5,
    scheduler="UniPCMultistep",
    flow_shift=6.0,
    frame_multiplier=16,
    safe_mode=False,
    lora_groups=None,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    video_component=True,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate a video from an input image using the Wan 2.2 14B I2V model with Lightning LoRA.
    This function takes an input image and generates a video animation based on the provided
    prompt and parameters. It uses an FP8 qunatized Wan 2.2 14B Image-to-Video model in with Lightning LoRA
    for fast generation in 4-8 steps.
    Args:
        input_image (PIL.Image): The input image to animate. Will be resized to target dimensions.
        last_image (PIL.Image, optional): The optional last image for the video.
        prompt (str): Text prompt describing the desired animation or motion.
        steps (int, optional): Number of inference steps. More steps = higher quality but slower.
            Defaults to 4. Range: 1-30.
        negative_prompt (str, optional): Negative prompt to avoid unwanted elements.
            Defaults to default_negative_prompt (contains unwanted visual artifacts).
        duration_seconds (float, optional): Duration of the generated video in seconds.
            Defaults to 2. Clamped between MIN_FRAMES_MODEL/FIXED_FPS and MAX_FRAMES_MODEL/FIXED_FPS.
        guidance_scale (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        guidance_scale_2 (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        seed (int, optional): Random seed for reproducible results. Defaults to 42.
            Range: 0 to MAX_SEED (2147483647).
        randomize_seed (bool, optional): Whether to use a random seed instead of the provided seed.
            Defaults to False.
        quality (float, optional): Video output quality. Default is 5. Uses variable bit rate.
            Highest quality is 10, lowest is 1.
        scheduler (str, optional): The name of the scheduler to use for inference. Defaults to "UniPCMultistep".
        flow_shift (float, optional): The flow shift value for compatible schedulers. Defaults to 6.0.
        frame_multiplier (int, optional): The int value for fps enhancer
        width (int, optional): Output width in pixels. Must be a multiple of MULTIPLE_OF (16).
            Defaults to DEFAULT_WIDTH (640).
        height (int, optional): Output height in pixels. Must be a multiple of MULTIPLE_OF (16).
            Defaults to DEFAULT_HEIGHT (640).
        video_component(bool, optional): Show video player in output.
            Defaults to True.
        progress (gr.Progress, optional): Gradio progress tracker. Defaults to gr.Progress(track_tqdm=True).
    Returns:
        tuple: A tuple containing:
            - video_path (str): Path for the video component.
            - video_path (str): Path for the file download component. Attempt to avoid reconversion in video component.
            - current_seed (int): The seed used for generation.
    Raises:
        gr.Error: If input_image is None (no image uploaded).
    Note:
        - Frame count is calculated as duration_seconds * FIXED_FPS (24)
        - Output dimensions are adjusted to be multiples of MOD_VALUE (32)
        - The function uses GPU acceleration for inference
        - Generation time varies based on steps and duration (see get_duration function)
    """
    
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image, int(width), int(height))

    processed_last_image = None
    if last_image:
        processed_last_image = resize_and_crop_to_match(last_image, resized_image)

    video_path, task_n = run_inference(
        resized_image,
        processed_last_image,
        prompt,
        steps,
        negative_prompt,
        num_frames,
        guidance_scale,
        guidance_scale_2,
        current_seed,
        scheduler,
        flow_shift,
        frame_multiplier,
        quality,
        duration_seconds,
        safe_mode,
        lora_groups,
        progress,
    )
    print(f"GPU complete: {task_n} | RAM peak so far={RAM_PEAK_GIB:.2f} GiB")

    return (video_path if video_component else None), video_path, current_seed


CSS = """
#hidden-timestamp {
    opacity: 0;
    height: 0px;
    width: 0px;
    margin: 0px;
    padding: 0px;
    overflow: hidden;
    position: absolute;
    pointer-events: none;
}
"""


with gr.Blocks(delete_cache=(3600, 10800)) as demo:
    gr.Markdown(model_title())
    gr.Markdown("Run Wan 2.2 in just 4-8 steps, fp8 quantization & AoT compilation - compatible with 🧨 diffusers and ZeroGPU")

    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image", sources=["upload", "clipboard"])
            image_size_info = gr.Markdown("")
            user_override_state = gr.State(False)
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")
            frame_multi = gr.Dropdown(
                choices=[FIXED_FPS, FIXED_FPS*2, FIXED_FPS*4, FIXED_FPS*8],
                value=FIXED_FPS,
                label="Video Fluidity (Frames per Second)",
                info="Extra frames will be generated using flow estimation, which estimates motion between frames to make the video smoother."
            )
            safe_mode_checkbox = gr.Checkbox(
                label="🛠️ Safe Mode",
                value=True,
                info="Requests 30% extra processing time to try to prevent unfinished tasks when the server is busy."
            )
            with gr.Accordion("Advanced Settings", open=False):
                last_image_component = gr.Image(type="pil", label="Last Image (Optional)", sources=["upload", "clipboard"])
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, info="Used if any Guidance Scale > 1.", lines=3)
                quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=6, label="Video Quality", info="If set to 10, the generated video may be too large and won't play in the Gradio preview.")
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Inference Steps")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale - high noise stage", info="Values above 1 increase GPU usage and may take longer to process.")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2 - low noise stage")
                scheduler_dropdown = gr.Dropdown(
                    label="Scheduler",
                    choices=list(SCHEDULER_MAP.keys()),
                    value="UniPCMultistep",
                    info="Select a custom scheduler."
                )
                flow_shift_slider = gr.Slider(minimum=0.5, maximum=15.0, step=0.1, value=3.0, label="Flow Shift")
                with gr.Row():
                    width_input = gr.Slider(minimum=MIN_DIM, maximum=MAX_DIM, step=MULTIPLE_OF, value=DEFAULT_WIDTH, label="Output Width (px)", info="Steps of 16. Image is cover-cropped to fit.")
                    height_input = gr.Slider(minimum=MIN_DIM, maximum=MAX_DIM, step=MULTIPLE_OF, value=DEFAULT_HEIGHT, label="Output Height (px)", info="Higher values use much more VRAM/time — lower if you hit OOM.")
                tokens_info = gr.Markdown(compute_tokens_text(DEFAULT_WIDTH, DEFAULT_HEIGHT, 3.5))
                lora_dropdown = gr.Dropdown(choices=lora_loader.get_lora_choices(), label="LoRA (NSFW)", multiselect=True, info="Select scenario LoRAs")
                play_result_video = gr.Checkbox(label="Display result", value=True, interactive=True)

            generate_button = gr.Button("Generate Video", variant="primary")

        with gr.Column():
            # ASSIGNED elem_id="generated-video" so JS can find it
            video_output = gr.Video(label="Generated Video", autoplay=True, sources=["upload"], buttons=["download", "share"], interactive=True, elem_id="generated-video")
            
            # --- Frame Grabbing UI ---
            with gr.Row():
                grab_frame_btn = gr.Button("📸 Use Current Frame as Input", variant="secondary")
                timestamp_box = gr.Number(value=0, label="Timestamp", visible=True, elem_id="hidden-timestamp")
            # -------------------------
            
            file_output = gr.File(label="Download Video")

    ui_inputs = [
        input_image_component, last_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox,
        quality_slider, scheduler_dropdown, flow_shift_slider, frame_multi,
        safe_mode_checkbox,
        lora_dropdown,
        width_input, height_input,
        play_result_video
    ]
    
    generate_button.click(
        fn=generate_video, 
        inputs=ui_inputs, 
        outputs=[video_output, file_output, seed_input]
    )

    input_image_component.change(
        fn=on_image_upload,
        inputs=[input_image_component, user_override_state, duration_seconds_input],
        outputs=[width_input, height_input, image_size_info, user_override_state, tokens_info],
    )
    width_input.change(fn=mark_user_override, outputs=[user_override_state])
    height_input.change(fn=mark_user_override, outputs=[user_override_state])
    for dim_input in [width_input, height_input, duration_seconds_input]:
        dim_input.change(
            fn=compute_tokens_text,
            inputs=[width_input, height_input, duration_seconds_input],
            outputs=[tokens_info],
        )
    
    # --- Frame Grabbing Events ---
    # 1. Click button -> JS runs -> puts time in hidden number box
    grab_frame_btn.click(
        fn=None,
        inputs=None,
        outputs=[timestamp_box],
        js=get_timestamp_js
    )
    
    # 2. Hidden number box changes -> Python runs -> puts frame in Input Image
    timestamp_box.change(
        fn=extract_frame,
        inputs=[video_output, timestamp_box],
        outputs=[input_image_component]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        css=CSS,
        show_error=True,
    )