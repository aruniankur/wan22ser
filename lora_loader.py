"""
LoRA Loader for WAN 2.2 I2V - multi-repo.
Baseline: lkzd7/WAN2.2_LoraSet_NSFW. Plus curated I2V act/helper LoRAs from
Lythiga/WAN2.2_I2V_Lora, rolechar/rc_wan2.2_i2v_loraset_nsfw_private,
lopi999/Wan2.2-I2V_General-NSFW-LoRA. T2V / lightning / furry / dance / male-char filtered out.
"""
import os
import urllib.parse
import re
from huggingface_hub import hf_hub_download

LORA_REPO = "lkzd7/WAN2.2_LoraSet_NSFW"
HF_TOKEN = os.environ.get("HF_TOKEN")  # authenticated downloads (covers private repos)

# Pinned commit hashes (2026-07-19) — protection from breaking upstream changes
PINNED_REVISIONS = {
    "lkzd7/WAN2.2_LoraSet_NSFW": "ccfa867ce4b1",
    "Lythiga/WAN2.2_I2V_Lora": "67ef542e6faa",
    "rolechar/rc_wan2.2_i2v_loraset_nsfw_private": "4855a2f216ac",
    "lopi999/Wan2.2-I2V_General-NSFW-LoRA": "aeef17d7fa51",
}

LORA_FILES = [
    "Blink_Squatting_Cowgirl_Position_I2V_HIGH.safetensors",
    "Blink_Squatting_Cowgirl_Position_I2V_LOW.safetensors",
    "PENISLORA_22_i2v_HIGH_e320.safetensors",
    "PENISLORA_22_i2v_LOW_e496.safetensors",
    "Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_high_noise.safetensors",
    "Pornmaster_wan 2.2_14b_I2V_bukkake_v1.4_low_noise.safetensors",
    "W22_Multiscene_Photoshoot_Softcore_i2v_HN.safetensors",
    "W22_Multiscene_Photoshoot_Softcore_i2v_LN.safetensors",
    "WAN-2.2-I2V-Double-Blowjob-HIGH-v1.safetensors",
    "WAN-2.2-I2V-Double-Blowjob-LOW-v1.safetensors",
    "WAN-2.2-I2V-HandjobBlowjobCombo-HIGH-v1.safetensors",
    "WAN-2.2-I2V-HandjobBlowjobCombo-LOW-v1.safetensors",
    "WAN-2.2-I2V-SensualTeasingBlowjob-HIGH-v1.safetensors",
    "WAN-2.2-I2V-SensualTeasingBlowjob-LOW-v1.safetensors",
    "iGOON_Blink_Blowjob_I2V_HIGH.safetensors",
    "iGOON_Blink_Blowjob_I2V_LOW.safetensors",
    "iGoon - Blink_Front_Doggystyle_I2V_HIGH.safetensors",
    "iGoon - Blink_Front_Doggystyle_I2V_LOW.safetensors",
    "iGoon - Blink_Missionary_I2V_HIGH.safetensors",
    "iGoon - Blink_Missionary_I2V_LOW v2.safetensors",
    "iGoon - Blink_Missionary_I2V_LOW.safetensors",
    "iGoon%20-%20Blink_Back_Doggystyle_HIGH.safetensors",
    "iGoon%20-%20Blink_Back_Doggystyle_LOW.safetensors",
    "iGoon%20-%20Blink_Facial_I2V_HIGH.safetensors",
    "iGoon%20-%20Blink_Facial_I2V_LOW.safetensors",
    "iGoon_Blink_Missionary_I2V_HIGH v2.safetensors",
    "iGoon_Blink_Titjob_I2V_HIGH.safetensors",
    "iGoon_Blink_Titjob_I2V_LOW.safetensors",
    "lips-bj_high_noise.safetensors",
    "lips-bj_low_noise.safetensors",
    "mql_casting_sex_doggy_kneel_diagonally_behind_vagina_wan22_i2v_v1_high_noise.safetensors",
    "mql_casting_sex_doggy_kneel_diagonally_behind_vagina_wan22_i2v_v1_low_noise.safetensors",
    "mql_casting_sex_reverse_cowgirl_lie_front_vagina_wan22_i2v_v1_high_noise.safetensors",
    "mql_casting_sex_reverse_cowgirl_lie_front_vagina_wan22_i2v_v1_low_noise.safetensors",
    "mql_casting_sex_spoon_wan22_i2v_v1_high_noise.safetensors",
    "mql_casting_sex_spoon_wan22_i2v_v1_low_noise.safetensors",
    "mql_massage_tits_wan22_i2v_v1_high_noise.safetensors",
    "mql_massage_tits_wan22_i2v_v1_low_noise.safetensors",
    "mql_panties_aside_wan22_i2v_v1_high_noise.safetensors",
    "mql_panties_aside_wan22_i2v_v1_low_noise.safetensors",
    "sfbehind_v2.1_high_noise.safetensors",
    "sfbehind_v2.1_low_noise.safetensors",
    "sid3l3g_transition_v2.0_H.safetensors",
    "sid3l3g_transition_v2.0_L.safetensors",
    "wan2.2_i2v_high_ulitmate_pussy_asshole.safetensors",
    "wan2.2_i2v_low_ulitmate_pussy_asshole.safetensors",
    "wan22-mouthfull-140epoc-high-k3nk.safetensors",
    "wan22-mouthfull-152epoc-low-k3nk.safetensors",
]

# group -> {"HIGH": (repo, file) | None, "LOW": (repo, file) | None}
LORA_PAIRS = {}
for f in LORA_FILES:
    name = urllib.parse.unquote(f).replace(".safetensors", "")
    is_high = bool(re.search(r'(high|HN|_H\b)', name, re.IGNORECASE))
    is_low = bool(re.search(r'(low|LN|_L\b)', name, re.IGNORECASE))
    group = re.sub(r'[\s_-]*(high|low|noise|HN|LN)([\s_-]*noise)?[\s_-]*(v?\d+(\.\d+)?)?\s*$', '', name, flags=re.IGNORECASE).strip()
    group = re.sub(r'[\s_]+$', '', group)
    LORA_PAIRS.setdefault(group, {"HIGH": None, "LOW": None})
    if is_high:
        LORA_PAIRS[group]["HIGH"] = (LORA_REPO, f)
    elif is_low:
        LORA_PAIRS[group]["LOW"] = (LORA_REPO, f)

L = "Lythiga/WAN2.2_I2V_Lora"
R = "rolechar/rc_wan2.2_i2v_loraset_nsfw_private"
P = "lopi999/Wan2.2-I2V_General-NSFW-LoRA"

# curated extra act/helper LoRAs (label -> (repo, high_file, low_file|None))
EXTRA = {
    # --- deepthroat ---
    "Ultimate Deepthroat": (L, "wan22-ultimatedeepthroat-i2v-102epoc-high-k3nk.safetensors", "wan22-ultimatedeepthroat-I2V-101epoc-low-k3nk.safetensors"),
    "BBC Deepthroat": (L, "wan22-bbcdeepthroat-115epoc-high-k3nk.safetensors", "wan22-bbcdeepthroat-155epoc-low-720-k3nk.safetensors"),
    "Throat V2": (L, "Wan22_ThroatV2_High.safetensors", "Wan22_ThroatV2_Low.safetensors"),
    "JFJ Deepthroat": (L, "jfj-deepthroat-W22-I2V-HN.safetensors", None),
    # --- blowjob extra ---
    "BBC Blowjob Extreme": (L, "BBC Blowjobs Extreme_high_noise.safetensors", "BBC Blowjobs Extreme_low_noise.safetensors"),
    "POV Blowjob": (L, "Pov_Blowjob_Wan2.2_high_I2V_v1.0.safetensors", None),
    # --- cunnilingus ---
    "Cunnilingus": (L, "wan22-cunilingus-I2V-106epoc-high.safetensors", "wan22-cunilingus-I2V-72epoc-low.safetensors"),
    # --- cowgirl ---
    "Reverse Cowgirl v2": (L, "wan22.r3v3rs3_c0wg1rl-14b-High-i2v_e70.safetensors", "wan22.r3v3rs3_c0wg1rl-14b-Low-i2v_e70.safetensors"),
    # --- doggy / from behind ---
    "Doggy Front View v2": (L, "doggy_style_sex_front_view_Wan2.2_I2V_high_v1.0.safetensors", "doggy_style_sex_front_view_Wan2.2_I2V_Low_v1.0.safetensors"),
    "Fucked From Behind": (L, "derpfuckedfrombehind_000002125_high_noise.safetensors", "derpfuckedfrombehind_000002125_low_noise.safetensors"),
    "Prone Bone": (L, "Pronebone_high_noise.safetensors", "Pronebone_low_noise.safetensors"),
    "Reverse Suspended Congress": (L, "reverse_suspended_congress_I2V_high.safetensors", "reverse_suspended_congress_I2V_low.safetensors"),
    # --- missionary ---
    "POV Missionary": (L, "wan2.2_i2v_highnoise_pov_missionary_v1.0.safetensors", "wan2.2_i2v_lownoise_pov_missionary_v1.0.safetensors"),
    "Mating Press": (L, "mating_press_high.safetensors", "mating_press_low.safetensors"),
    # --- standing ---
    "Standing Upright Sex": (L, "Upright_sex_Wan2.2_14B_TI2V_HIGH_v1.0.safetensors", "Upright_sex_Wan2.2_14B_TI2V_LOW_v1.0.safetensors"),
    # --- anal / ass ---
    "Oral Insertion": (L, "wan2.2-i2v-high-oral-insertion-v1.0.safetensors", "wan2.2-i2v-low-oral-insertion-v1.0.safetensors"),
    # --- handjob / balls ---
    "Handjob": (L, "WAN-2.2-I2V-Handjob-HIGH-v1.safetensors", "WAN-2.2-I2V-Handjob-LOW-v1.safetensors"),
    "Balls Sucking": (L, "WAN-2.2-I2V_Balls_sucking_High_noise.safetensors", "WAN-2.2-I2V_Balls_sucking_Low_noise.safetensors"),
    # --- fingering ---
    "Fingering": (R, "Sensual_fingering_v1_high_noise.safetensors", "Sensual_fingering_v1_low_noise.safetensors"),
    # --- cumshot / creampie ---
    "Facesplash Cumshot": (L, "wan22-f4c3spl4sh-100epoc-high-k3nk.safetensors", "wan22-f4c3spl4sh-154epoc-low-k3nk.safetensors"),
    "Creampie CRM": (L, "Creampie CRM-FULL-EPOCH-80-HIGH.safetensors", "Creampie CRM-FULL-EPOCH-80-LOW.safetensors"),
    "Pornmaster Creampie": (L, "Pornmaster_wan 2.2_14b_I2V_Creampie_v1_high_noise.safetensors", "Pornmaster_wan 2.2_14b_I2V_Creampie_v1_low_noise.safetensors"),
    "Cum (generic)": (L, "Wan22_Cum_high_noise_1.V1.safetensors", "Wan22_Cum_low_noise_1.V1.safetensors"),
    # --- fetish / expression ---
    "French Kiss": (L, "WAN2.2-FrenchKiss_HighNoise.safetensors", "WAN2.2-FrenchKiss_LowNoise.safetensors"),
    "Twerking": (L, "Twerking_I2V_high_noise.safetensors", "Twerking_I2V_low_noise.safetensors"),
    "Ahegao Drooling": (R, "middlefinger_drooling_ahegao_high.safetensors", "middlefinger_drooling_ahegao_low.safetensors"),
    # --- helpers: size / penetration / anatomy / camera / booster ---
    "General NSFW Booster": (P, "NSFW-22-H-e8.safetensors", "NSFW-22-L-e8.safetensors"),
    "Penetration Insert": (L, "PenInsert_high_noise.safetensors", "PenInsert_low_noise.safetensors"),
    "Biggest Cock (size)": (L, "WAN-2.2-I2V_BiggestCock_high_noise_V1.safetensors", "WAN-2.2-I2V_BiggestCock_low_noise_V1.safetensors"),
    "Doggy Slider": (L, "I2V_doggyslider_high.safetensors", "I2V_doggyslider_low.safetensors"),
    "Pussy Helper": (R, "wan2.2_i2v_ulitmate_pussy_helper_high.safetensors", "wan2.2_i2v_ulitmate_pussy_helper_low.safetensors"),
    "FOV Slider (camera)": (L, "wan2.2-i2v-high-sex-fov-slider-v1.0.safetensors", "wan2.2-i2v-low-sex-fov-slider-v1.0.safetensors"),
    "Smashcut (camera)": (R, "wan2.2-i2v-sex-smashcut-v1.0-high.safetensors", "wan2.2-i2v-sex-smashcut-v1.0-low.safetensors"),
}
for label, (repo, hi, lo) in EXTRA.items():
    LORA_PAIRS.setdefault(label, {"HIGH": None, "LOW": None})
    if hi:
        LORA_PAIRS[label]["HIGH"] = (repo, hi)
    if lo:
        LORA_PAIRS[label]["LOW"] = (repo, lo)


def get_lora_choices():
    choices = []
    for group in sorted(LORA_PAIRS.keys()):
        p = LORA_PAIRS[group]
        if p["HIGH"] and p["LOW"]:
            choices.append(group)
        elif p["HIGH"]:
            choices.append(f"{group} (HIGH only)")
        elif p["LOW"]:
            choices.append(f"{group} (LOW only)")
    return choices


def download_lora(group_name):
    if not group_name:
        return None, None
    clean_name = re.sub(r'\s*\(HIGH only\)|\s*\(LOW only\)', '', group_name)
    if clean_name not in LORA_PAIRS:
        return None, None
    pair = LORA_PAIRS[clean_name]
    high_path, low_path = None, None
    if pair["HIGH"]:
        repo, fn = pair["HIGH"]
        high_path = hf_hub_download(repo, fn, token=HF_TOKEN, revision=PINNED_REVISIONS.get(repo))
    if pair["LOW"]:
        repo, fn = pair["LOW"]
        low_path = hf_hub_download(repo, fn, token=HF_TOKEN, revision=PINNED_REVISIONS.get(repo))
    return high_path, low_path


def load_lora_to_pipe(pipe, group_name, adapter_name="lora"):
    high_path, low_path = download_lora(group_name)
    if high_path and low_path:
        pipe.load_lora_weights(high_path, adapter_name=f"{adapter_name}_high")
        pipe.load_lora_weights(low_path, adapter_name=f"{adapter_name}_low")
        print(f"Loaded LoRA pair: {group_name}")
        return True
    elif high_path:
        pipe.load_lora_weights(high_path, adapter_name=adapter_name)
        print(f"Loaded LoRA: {group_name}")
        return True
    elif low_path:
        pipe.load_lora_weights(low_path, adapter_name=adapter_name)
        print(f"Loaded LoRA (low): {group_name}")
        return True
    return False


def unload_lora(pipe):
    try:
        pipe.unload_lora_weights()
    except:
        pass
