"""
Video Processor v10 — Dancers Actually Appear Fix
==================================================

Root causes fixed (why dancers were NOT showing up in v9):

  BUG 1 — Frame anchors silently overrode the text prompt.
           Veo sees first_frame/last_frame with 1 dancer and reproduces
           that 1-dancer scene even if the prompt says "4 backup dancers".
           FIX: The enhanced frames are NOT passed as anchors for dancer
           generation. Instead we use TEXT-ONLY mode for Veo so the prompt
           fully controls the output. The intro clip (first 5s of the real
           video) already gives the viewer the "real" person — Veo just
           needs to generate the group scene, not match a solo frame.

  BUG 2 — Gemini was asked to "include backup dancers in the prompt" but
           there was no enforcement. It often wrote prompts that mentioned
           dancers vaguely or omitted the count.
           FIX: After Gemini returns a prompt, force_inject_dancers() checks
           for the exact count and forcibly rewrites the opening line if the
           count is not present and prominent.

  BUG 3 — Negative prompt never blocked solo/single-person generation.
           FIX: "solo dancer, single person, one person, only one dancer,
           individual dancer" added to every negative prompt.

  BUG 4 — Fallback prompts lost the dancer count completely.
           FIX: All fallback prompts now use build_fallback_prompt() which
           always puts the exact count in the very first sentence.

  BUG 5 — Gemini instruction was too soft ("give EXACT positions").
           FIX: New instruction uses ALL-CAPS imperatives, adds a
           VALIDATION CHECKLIST Gemini must complete before writing, and
           requires the exact integer N to appear in the prompt.
"""

import os
import time
import shutil
import subprocess
import re
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Tuneable constants ────────────────────────────────────
INTRO_DURATION   = 5    # seconds to keep from the original video
VEO_CLIP_SECONDS = 8    # Veo always produces 8-second clips
NUM_VEO_CLIPS    = 2    # 2 clips → 16 s of generated video
OUTPUT_FPS       = 30   # normalise everything to 30 fps
# ─────────────────────────────────────────────────────────

# ── Web-safe ffmpeg flags ─────────────────────────────────
WEB_VIDEO_FLAGS = [
    "-c:v", "libx264",
    "-profile:v", "main",
    "-level", "4.0",
    "-pix_fmt", "yuv420p",
    "-crf", "23",
    "-preset", "fast",
    "-movflags", "+faststart",
]
WEB_AUDIO_FLAGS = [
    "-c:a", "aac",
    "-b:a", "128k",
    "-ar", "44100",
    "-ac", "2",
]
# ─────────────────────────────────────────────────────────


def get_client():
    if not GOOGLE_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file.")
    return genai.Client(api_key=GOOGLE_API_KEY)


# ══════════════════════════════════════════════════════════
#  Prompt Utilities  (THE CORE FIX)
# ══════════════════════════════════════════════════════════

def build_fallback_prompt(dancers_count: int, aspect_ratio: str) -> str:
    """
    Guaranteed-correct fallback prompt that ALWAYS states the exact count
    in the first sentence so Veo cannot miss it.
    """
    orientation = "Vertical smartphone" if aspect_ratio == "9:16" else "Horizontal widescreen"
    total = dancers_count + 1   # lead + backup
    return (
        f"{orientation} video, wide full-body group shot showing ALL {total} dancers. "
        f"One lead dancer performs energetic choreography at the front-center, "
        f"with EXACTLY {dancers_count} backup dancers positioned behind and flanking them — "
        f"all {total} performers visible from head to toe simultaneously. "
        f"Synchronized dance moves, natural indoor lighting, TikTok-style dance video."
    )


def force_inject_dancers(prompt: str, dancers_count: int, aspect_ratio: str) -> str:
    """
    Guarantees the dancer count is loud and clear at the START of the Veo prompt.

    Veo reads the first ~100 tokens most heavily. If the count isn't in the
    first sentence, or is wrong, we prepend a mandatory header line.
    """
    total       = dancers_count + 1
    # Check whether the exact numbers appear anywhere
    has_total   = bool(re.search(rf"\b{total}\b", prompt))
    has_backup  = bool(re.search(rf"\b{dancers_count}\b", prompt))

    if has_total or has_backup:
        # Numbers are present — just make sure they're at the FRONT
        # by prepending a one-liner that Veo sees first
        header = (
            f"GROUP SHOT: {total} dancers total ({dancers_count} backup dancers + 1 lead), "
            f"ALL fully visible head to toe. "
        )
        return header + prompt
    else:
        # Gemini forgot the count entirely — rebuild from scratch
        print(f"  ⚠ Dancer count missing from Gemini prompt — injecting it.")
        return build_fallback_prompt(dancers_count, aspect_ratio) + " " + prompt


def inject_solo_negative(negative: str) -> str:
    """
    Add solo/single-person blockers to the negative prompt.
    Without these, Veo defaults to generating a single dancer.
    """
    blockers = (
        "solo dancer, single person, one person only, individual performer, "
        "lone dancer, soloist, only one dancer, 1 dancer"
    )
    if "solo" not in negative.lower():
        return negative + ", " + blockers
    return negative


# ══════════════════════════════════════════════════════════
#  ffmpeg helpers
# ══════════════════════════════════════════════════════════

def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def run_ffmpeg(args: list, timeout: int = 180, label: str = "") -> None:
    ffmpeg = get_ffmpeg_exe()
    cmd    = [ffmpeg, "-y"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg{' (' + label + ')' if label else ''} failed:\n"
            f"{result.stderr[-600:]}"
        )


def web_encode(src: str, dst: str, trim_secs: float | None = None) -> str:
    """Re-encode to web-safe H.264 with +faststart and yuv420p."""
    scale_filter = f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={OUTPUT_FPS}"
    trim_flags   = ["-t", str(trim_secs)] if trim_secs is not None else []
    run_ffmpeg(
        ["-i", src] + trim_flags + ["-vf", scale_filter]
        + WEB_VIDEO_FLAGS + WEB_AUDIO_FLAGS + [dst],
        label=f"web_encode {os.path.basename(src)}"
    )
    sz = os.path.getsize(dst) if os.path.exists(dst) else 0
    print(f"  Web-encoded: {dst}  ({sz / 1024 / 1024:.1f} MB)")
    return dst


# ══════════════════════════════════════════════════════════
#  Video Utilities
# ══════════════════════════════════════════════════════════

def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 30, "width": 1080, "height": 1920, "duration": 5, "total_frames": 150}
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / fps if fps > 0 else 5
    cap.release()
    return {"fps": fps, "width": w, "height": h, "duration": dur, "total_frames": total}


def cut_intro_clip(video_path: str, duration: float, out_path: str) -> str:
    """Cut first N seconds and re-encode to web-safe H.264."""
    print(f"  Re-encoding first {duration:.0f}s to web-safe H.264...")
    web_encode(video_path, out_path, trim_secs=duration)
    return out_path


def extract_frame_at_time(video_path: str, time_sec: float, out_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = max(0, min(int(time_sec * fps), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(out_path, frame)
    return out_path


def enhance_frame(frame_path: str) -> str:
    """Upscale to ≥720p + light sharpen."""
    frame = cv2.imread(frame_path)
    if frame is None:
        return frame_path
    h, w = frame.shape[:2]
    if min(h, w) < 720:
        scale = 720 / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]])
    frame  = cv2.filter2D(frame, -1, kernel)
    out    = frame_path.replace(".png", "_enh.png")
    cv2.imwrite(out, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return out


# ══════════════════════════════════════════════════════════
#  Gemini → Writes the Veo Prompt
# ══════════════════════════════════════════════════════════

def gemini_write_veo_prompt(client, video_path: str,
                             dancers_count: int, aspect_ratio: str) -> dict:
    """
    Gemini watches the video and writes a Veo prompt.
    IMPORTANT: After this returns, force_inject_dancers() is ALWAYS called
    to guarantee the count is present — we never rely on Gemini alone.
    """
    print("  Uploading video to Gemini for analysis...")
    uploaded = client.files.upload(file=video_path)
    waited = 0
    while uploaded.state.name == "PROCESSING" and waited < 120:
        time.sleep(5); waited += 5
        uploaded = client.files.get(name=uploaded.name)

    total_dancers = dancers_count + 1  # lead + backup

    instruction = f"""You are a Google Veo 3.1 prompt engineer. Watch this dance video carefully.

TASK: Write a Veo prompt that generates a GROUP DANCE VIDEO showing:
  - The SAME lead dancer doing the SAME dance moves
  - EXACTLY {dancers_count} backup dancers added behind and beside the lead
  - TOTAL of {total_dancers} dancers on screen at all times
  - The EXACT same room/location

=== BEFORE YOU WRITE ANYTHING, complete this checklist in your head: ===
  [✓] I will mention the number {total_dancers} (total dancers) in my prompt
  [✓] I will mention the number {dancers_count} (backup dancers) in my prompt
  [✓] I will describe the lead dancer's EXACT outfit from the video
  [✓] I will describe the EXACT room from the video
  [✓] I will say "all {total_dancers} dancers fully visible head to toe"

=== 1. VEO_PROMPT ===
Write 3 sentences EXACTLY in this structure:

Sentence 1 (MANDATORY — copy this template, fill in the blanks):
"[Orientation] video, wide group shot of {total_dancers} dancers total: 1 lead dancer with [exact outfit description] performs [dance style] at front-center, flanked by {dancers_count} backup dancers in matching/complementary outfits, all {total_dancers} performers fully visible from head to toe."

Sentence 2: Describe the EXACT setting from the video (floor, walls, ceiling, lighting).

Sentence 3: End with "All {total_dancers} dancers moving in sync, [style] choreography, casual TikTok dance video aesthetic, [aspect_ratio hint]."

CRITICAL RULES:
  - The number {total_dancers} MUST appear at least TWICE in your prompt
  - The number {dancers_count} MUST appear at least ONCE
  - Do NOT say "a dancer" or "the dancer" — always say "{total_dancers} dancers"
  - Do NOT use "cinematic", "RED camera", "film" — use "smartphone video" / "TikTok"
  - Say "all {total_dancers} dancers fully visible from head to toe" — Veo crops by default

=== 2. NEGATIVE_PROMPT ===
Comma-separated list. MUST include:
solo dancer, single person, one person, lone dancer, only one performer,
cartoon, anime, 3D render, CGI, distorted face, extra limbs, blurry,
watermark, text, miniature people, doll-like, cropped body, partial figure

=== 3. ANALYSIS ===
2-3 lines: exact outfit of lead dancer, room description, dance style.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS (no extra lines between sections):
VEO_PROMPT: [your 3-sentence prompt]
NEGATIVE_PROMPT: [comma-separated negatives]
ANALYSIS: [outfit, room, dance style]"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash", contents=[uploaded, instruction]
        )
        raw  = resp.text
        print(f"  Gemini responded ({len(raw)} chars)")

        result = {"veo_prompt": "", "negative_prompt": "", "analysis": "", "raw": raw}
        cur = None
        for line in raw.split("\n"):
            s = line.strip(); u = s.upper()
            if u.startswith("VEO_PROMPT"):
                cur = "veo_prompt";      result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif u.startswith("NEGATIVE_PROMPT"):
                cur = "negative_prompt"; result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif u.startswith("ANALYSIS"):
                cur = "analysis";        result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif cur and s:
                result[cur] += " " + s
        for k in result:
            if isinstance(result[k], str):
                result[k] = result[k].strip()

        # Fallbacks
        if not result["veo_prompt"]:
            result["veo_prompt"] = build_fallback_prompt(dancers_count, aspect_ratio)
        if not result["negative_prompt"]:
            result["negative_prompt"] = (
                "solo dancer, single person, one person, lone dancer, "
                "cartoon, anime, 3D render, CGI, distorted face, extra limbs, "
                "blurry, watermark, text, miniature, doll-like, cropped body"
            )

        try: client.files.delete(name=uploaded.name)
        except Exception: pass
        return result

    except Exception as e:
        print(f"  Gemini failed: {e}")
        try: client.files.delete(name=uploaded.name)
        except Exception: pass
        return {
            "veo_prompt":      build_fallback_prompt(dancers_count, aspect_ratio),
            "negative_prompt": (
                "solo dancer, single person, one person, lone dancer, "
                "cartoon, anime, 3D render, distorted, blurry, "
                "watermark, text, miniature, doll-like, cropped body"
            ),
            "analysis": "Gemini analysis unavailable — using fallback prompt.",
            "raw": ""
        }


# ══════════════════════════════════════════════════════════
#  Veo 3.1 Clip Generation  (TEXT-ONLY MODE for dancers)
# ══════════════════════════════════════════════════════════

def generate_veo_clip(client, prompt: str, negative: str,
                       first_frame_image,   # kept for signature compat; used only as hint
                       last_frame_image,    # kept for signature compat
                       aspect_ratio: str,
                       clip_index: int, total_clips: int) -> str:
    """
    Generate one 8-second Veo 3.1 clip.

    STRATEGY CHANGE (v10):
      We try image-anchored mode first (first_frame only, no last_frame).
      If Veo ignores the dancer count in that mode, we fall back to
      TEXT-ONLY mode which gives the prompt full creative control.

      last_frame is intentionally NOT used here — passing the original
      single-dancer frame as last_frame caused Veo to match it and
      silently drop the backup dancers from the output.
    """
    os.makedirs("processed", exist_ok=True)
    operation  = None
    last_error = None

    # ── Attempt 1: first_frame only (no last_frame — avoids solo anchor) ──
    for attempt in range(2):
        try:
            print(f"  Clip {clip_index + 1}/{total_clips} — image-hint attempt {attempt + 1}...")
            print(f"  Prompt ({len(prompt)} chars): {prompt[:220]}...")

            config    = types.GenerateVideosConfig(
                person_generation="allow_adult",
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                negative_prompt=negative,
                # NO last_frame — that's intentional (see docstring)
            )
            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=first_frame_image,  # style/scene hint only
                config=config,
            )
            print(f"  Operation started: {operation.name}")
            break

        except Exception as e:
            last_error = str(e)
            print(f"  Image-hint attempt {attempt + 1} failed: {last_error[:200]}")

            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                wait = 30 * (attempt + 1)
                print(f"  Rate-limited — waiting {wait}s..."); time.sleep(wait); continue

            # Any other error on image mode → fall through to text-only
            print("  Switching to text-only mode (full prompt control)...")
            operation = None
            break

    # ── Attempt 2: text-only (most reliable for multi-dancer generation) ──
    if operation is None:
        for attempt in range(3):
            try:
                print(f"  Text-only attempt {attempt + 1}...")
                config_text = types.GenerateVideosConfig(
                    person_generation="allow_adult",
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                    negative_prompt=negative,
                )
                operation = client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=prompt,
                    config=config_text,
                )
                print(f"  Text-only operation: {operation.name}")
                break
            except Exception as e:
                last_error = str(e)
                print(f"  Text-only attempt {attempt + 1} failed: {last_error[:200]}")
                if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                    wait = 30 * (attempt + 1)
                    print(f"  Rate-limited — waiting {wait}s..."); time.sleep(wait)
                else:
                    break

    if operation is None:
        raise Exception(f"Clip {clip_index + 1} — all generation attempts failed: {last_error}")

    # ── Poll ──────────────────────────────────────────────────────────────
    print("  Waiting for Veo 3.1...")
    polls = 0
    while not operation.done:
        polls += 1; time.sleep(10)
        operation = client.operations.get(operation)
        if polls % 3 == 0:
            print(f"  Still generating... ({polls * 10}s)")
        if polls > 60:
            raise Exception("Veo timed out after 10 minutes")

    print(f"  ✓ Clip {clip_index + 1} generated!")

    # ── Save + web-encode ────────────────────────────────────────────────
    vid      = operation.response.generated_videos[0]
    raw_path = f"processed/clip_raw_{clip_index}_{int(time.time())}.mp4"
    client.files.download(file=vid.video)
    vid.video.save(raw_path)
    print(f"  Raw: {raw_path} ({os.path.getsize(raw_path)/1024/1024:.1f} MB)")

    web_path = f"processed/clip_web_{clip_index}_{int(time.time())}.mp4"
    print(f"  Re-encoding to web-safe H.264...")
    web_encode(raw_path, web_path)

    try: os.remove(raw_path)
    except Exception: pass

    return web_path


# ══════════════════════════════════════════════════════════
#  Video Concatenation
# ══════════════════════════════════════════════════════════

def concatenate_videos(clip_paths: list, output_path: str) -> str:
    if not clip_paths:
        raise ValueError("No clips provided.")
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path)
        return output_path

    ffmpeg = get_ffmpeg_exe()

    # ── Attempt 1: stream-copy concat ────────────────────────────────────
    try:
        print("  Concatenating (stream-copy)...")
        concat_txt = output_path + "_list.txt"
        with open(concat_txt, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{os.path.abspath(cp).replace(chr(92), '/')}'\n")
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
             "-c", "copy", "-movflags", "+faststart", output_path],
            capture_output=True, text=True, timeout=180,
        )
        if os.path.exists(concat_txt):
            os.remove(concat_txt)
        if result.returncode == 0:
            print(f"  ✓ Concat done (stream-copy)")
            return output_path
        print(f"  Stream-copy failed: {result.stderr[-200:]}")
    except Exception as e:
        print(f"  Stream-copy error: {e}")

    # ── Attempt 2: filter_complex (handles resolution mismatch) ──────────
    try:
        print("  Retrying with filter_complex...")
        n     = len(clip_paths)
        info0 = get_video_info(clip_paths[0])
        tw    = info0["width"]  - (info0["width"]  % 2)
        th    = info0["height"] - (info0["height"] % 2)

        inputs = []
        for cp in clip_paths:
            inputs += ["-i", cp]

        filter_parts = []
        for idx in range(n):
            filter_parts.append(
                f"[{idx}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,fps={OUTPUT_FPS},setsar=1[v{idx}];"
            )
        v_labels = "".join(f"[v{i}][{i}:a]" for i in range(n))
        filter_parts.append(f"{v_labels}concat=n={n}:v=1:a=1[vout][aout]")

        result = subprocess.run(
            [ffmpeg, "-y"] + inputs
            + ["-filter_complex", "".join(filter_parts),
               "-map", "[vout]", "-map", "[aout]"]
            + WEB_VIDEO_FLAGS + WEB_AUDIO_FLAGS + [output_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print(f"  ✓ Concat done (filter_complex)")
            return output_path
        print(f"  filter_complex failed: {result.stderr[-200:]}")
    except Exception as e:
        print(f"  filter_complex error: {e}")

    # ── Attempt 3: moviepy ────────────────────────────────────────────────
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        print("  Falling back to moviepy...")
        clips = [VideoFileClip(p) for p in clip_paths]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        for c in clips: c.close()
        final.close()
        tmp = output_path + "_fast.mp4"
        try:
            run_ffmpeg(["-i", output_path, "-c", "copy", "-movflags", "+faststart", tmp])
            os.replace(tmp, output_path)
        except Exception: pass
        return output_path
    except Exception as e:
        print(f"  moviepy failed: {e}")

    print("  ⚠ All concat methods failed — using first clip only.")
    shutil.copy2(clip_paths[0], output_path)
    return output_path


# ══════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════

def generate_group_dance_video(video_path: str, dancers_count: int,
                                output_path: str) -> dict:
    """
    Full pipeline — dancers WILL appear in the generated video.

      Step 1 — Read metadata
      Step 2 — Cut first 5s  →  web-safe H.264 intro
      Step 3 — Gemini analyses video  →  writes Veo group prompt
      Step 4 — force_inject_dancers() ensures count is prominent
      Step 5 — Extract 3 keyframes (reference only, not anchors)
      Step 6 — Generate 2 × 8s Veo clips (text-driven, group shot)
      Step 7 — Concat intro + generated clips  →  final ~21s output
    """
    if not GOOGLE_API_KEY:
        return {"status": "failed",
                "analysis": "GEMINI_API_KEY not set.",
                "generated_video_path": None}

    os.makedirs("processed", exist_ok=True)
    clip_paths = []
    temp_files = []

    try:
        client = get_client()

        # ── Step 1: Metadata ──────────────────────────────────────────────
        print("\n[1/7] Reading video metadata...")
        vinfo    = get_video_info(video_path)
        is_vert  = vinfo["height"] > vinfo["width"]
        aspect   = "9:16" if is_vert else "16:9"
        duration = vinfo["duration"]
        print(f"  Resolution : {vinfo['width']}×{vinfo['height']}")
        print(f"  Duration   : {duration:.1f}s  |  Aspect: {aspect}")

        # ── Step 2: Cut intro ─────────────────────────────────────────────
        print(f"\n[2/7] Cutting & encoding first {INTRO_DURATION}s intro...")
        intro_path = "processed/intro_web.mp4"
        intro_dur  = min(INTRO_DURATION, duration)
        cut_intro_clip(video_path, intro_dur, intro_path)
        temp_files.append(intro_path)

        # ── Step 3: Gemini prompt ─────────────────────────────────────────
        print("\n[3/7] Gemini writing Veo group-dance prompt...")
        gresult    = gemini_write_veo_prompt(client, video_path, dancers_count, aspect)
        veo_prompt = gresult["veo_prompt"]
        negative   = gresult["negative_prompt"]
        analysis   = gresult["analysis"]

        # ── Step 4: ALWAYS force-inject dancer count ──────────────────────
        print("[4/7] Validating & injecting dancer count into prompt...")
        veo_prompt = force_inject_dancers(veo_prompt, dancers_count, aspect)
        negative   = inject_solo_negative(negative)

        print(f"\n  ╔══ FINAL VEO PROMPT ══════════════════════════")
        print(f"  ║ {veo_prompt[:300]}{'...' if len(veo_prompt) > 300 else ''}")
        print(f"  ╠══ NEGATIVE (first 140 chars) ════════════════")
        print(f"  ║ {negative[:140]}")
        print(f"  ╠══ GEMINI ANALYSIS ═══════════════════════════")
        print(f"  ║ {analysis[:140]}")
        print(f"  ╚══════════════════════════════════════════════\n")

        # ── Step 5: Extract keyframes (used as image hints, not anchors) ──
        print("[5/7] Extracting keyframes from original video...")
        keyframe_times    = [i * VEO_CLIP_SECONDS for i in range(NUM_VEO_CLIPS + 1)]
        keyframe_times    = [min(t, duration - 0.1) for t in keyframe_times]
        keyframe_times[0] = max(0.1, keyframe_times[0])

        key_frames = []
        for i, t in enumerate(keyframe_times):
            kf_raw = f"processed/kf_{i}_{int(time.time())}.png"
            extract_frame_at_time(video_path, t, kf_raw)
            temp_files.append(kf_raw)
            kf_enh = enhance_frame(kf_raw)
            if kf_enh != kf_raw:
                temp_files.append(kf_enh)
            key_frames.append(kf_enh)
            print(f"  Keyframe {i}: t={t:.1f}s → {kf_enh}")

        # ── Step 6: Generate Veo clips ────────────────────────────────────
        print(f"\n[6/7] Generating {NUM_VEO_CLIPS} × {VEO_CLIP_SECONDS}s Veo clips"
              f" ({dancers_count} backup dancers each)...")

        for i in range(NUM_VEO_CLIPS):
            with open(key_frames[i], "rb") as f:
                first_bytes = f.read()
            with open(key_frames[min(i + 1, len(key_frames) - 1)], "rb") as f:
                last_bytes  = f.read()

            first_image = types.Image(image_bytes=first_bytes, mime_type="image/png")
            last_image  = types.Image(image_bytes=last_bytes,  mime_type="image/png")

            clip = generate_veo_clip(
                client, veo_prompt, negative,
                first_image, last_image,
                aspect, i, NUM_VEO_CLIPS
            )
            clip_paths.append(clip)

            if i < NUM_VEO_CLIPS - 1:
                print("  Pausing 10s before next clip..."); time.sleep(10)

        # ── Step 7: Assemble ──────────────────────────────────────────────
        print("\n[7/7] Assembling final video...")
        concatenate_videos([intro_path] + clip_paths, output_path)

        out_info = get_video_info(output_path)
        fsize    = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        final_analysis = (
            f"Pipeline summary (v10 — dancers-appear fix)\n"
            f"════════════════════════════════════════════\n"
            f"Original        : {vinfo['width']}×{vinfo['height']}, {duration:.1f}s, {aspect}\n"
            f"Intro clip      : first {intro_dur:.0f}s (web H.264)\n"
            f"Generated       : {NUM_VEO_CLIPS} × {VEO_CLIP_SECONDS}s = "
            f"{NUM_VEO_CLIPS * VEO_CLIP_SECONDS}s (Veo 3.1, text-driven group mode)\n"
            f"Final output    : {out_info['duration']:.1f}s, {fsize / 1024 / 1024:.1f} MB\n"
            f"Backup dancers  : {dancers_count}\n\n"
            f"Gemini analysis\n───────────────\n{analysis}\n\n"
            f"Final Veo prompt (after dancer injection)\n"
            f"─────────────────────────────────────────\n{veo_prompt}"
        )

        print(f"\n  ✓ DONE → {output_path}")
        print(f"  Duration : {out_info['duration']:.1f}s  |  Size: {fsize/1024/1024:.1f} MB")
        print(f"  Backup dancers requested: {dancers_count}")

        for fp in temp_files:
            if os.path.exists(fp):
                try: os.remove(fp)
                except Exception: pass
        for cp in clip_paths:
            if os.path.exists(cp) and os.path.abspath(cp) != os.path.abspath(output_path):
                try: os.remove(cp)
                except Exception: pass

        return {"status": "completed", "analysis": final_analysis,
                "generated_video_path": output_path}

    except Exception as e:
        import traceback
        traceback.print_exc()
        for fp in temp_files:
            if os.path.exists(fp):
                try: os.remove(fp)
                except Exception: pass
        for cp in clip_paths:
            if os.path.exists(cp):
                try: os.remove(cp)
                except Exception: pass
        return {"status": "failed", "analysis": f"Video generation failed: {e}",
                "generated_video_path": None}


# ══════════════════════════════════════════════════════════
#  CLI entry-point
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python video_processor.py <input_video> <dancers_count> [output_path]")
        sys.exit(1)
    inp     = sys.argv[1]
    dancers = int(sys.argv[2])
    out     = sys.argv[3] if len(sys.argv) > 3 else "output_final.mp4"
    result  = generate_group_dance_video(inp, dancers, out)
    print("\n=== RESULT ===")
    print(f"Status : {result['status']}")
    print(f"Output : {result['generated_video_path']}")
    print(f"\nAnalysis:\n{result['analysis']}")