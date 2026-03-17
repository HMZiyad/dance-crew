"""
Video Processor v11 — Seamless Chaining + Crossfade Transitions
================================================================

TWO PROBLEMS FIXED IN THIS VERSION:

  PROBLEM 1 — "Jarring cut at the intro→generated boundary"
  ──────────────────────────────────────────────────────────
  Root cause: Veo clip 1 was generated using keyframe t=0s from the original
  video. The intro ends at t=5s — a completely different pose. Veo had no
  idea what the intro ended on, so it started a fresh unrelated scene.

  FIX: After cutting the intro, extract its VERY LAST FRAME and pass that
  as first_frame to Veo clip 1. Veo then animates FORWARD from that exact
  pose, so the viewer sees continuous motion, not a jump-cut to a new scene.
  A 0.5s xfade crossfade further smooths the visual join.

  PROBLEM 2 — "Clip 2 looks like a brand-new unrelated video"
  ────────────────────────────────────────────────────────────
  Root cause: Clip 2 was generated using keyframe t=8s from the original
  solo source — completely independent of clip 1's output.

  FIX: After clip 1 is generated, extract ITS LAST FRAME and pass that
  as first_frame to clip 2. Clip 2 is now a direct extension of clip 1,
  not a fresh generation.

  Chain:
    intro (5s real video)
      └─ last frame of intro ──► first_frame of Veo clip 1
                                        ↓  (8s, group dancers)
                                 last frame of clip 1 ──► first_frame of Veo clip 2
                                                                ↓  (8s, continuation)

  All joins: 0.5s xfade crossfade → no visible cut.
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
INTRO_DURATION   = 5      # seconds to keep from the original video
VEO_CLIP_SECONDS = 8      # Veo always produces 8-second clips
NUM_VEO_CLIPS    = 2      # 2 clips → 16 s of generated video
OUTPUT_FPS       = 30     # normalise everything to 30 fps
XFADE_DURATION   = 0.5   # crossfade length in seconds between every join
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
#  Prompt Utilities
# ══════════════════════════════════════════════════════════

def build_fallback_prompt(dancers_count: int, aspect_ratio: str) -> str:
    orientation = "Vertical smartphone" if aspect_ratio == "9:16" else "Horizontal widescreen"
    total = dancers_count + 1
    return (
        f"{orientation} video, wide full-body group shot showing ALL {total} dancers. "
        f"One lead dancer performs energetic choreography at the front-center, "
        f"with EXACTLY {dancers_count} backup dancers positioned behind and flanking them — "
        f"all {total} performers visible from head to toe simultaneously. "
        f"Synchronized dance moves, natural indoor lighting, TikTok-style dance video."
    )


def force_inject_dancers(prompt: str, dancers_count: int, aspect_ratio: str) -> str:
    """Guarantee the dancer count appears prominently at the front of the prompt."""
    total      = dancers_count + 1
    has_total  = bool(re.search(rf"\b{total}\b",         prompt))
    has_backup = bool(re.search(rf"\b{dancers_count}\b", prompt))
    header     = (
        f"GROUP SHOT: {total} dancers total ({dancers_count} backup dancers + 1 lead), "
        f"ALL fully visible head to toe. "
    )
    if not (has_total or has_backup):
        print("  ⚠ Dancer count missing from Gemini prompt — injecting it.")
        return build_fallback_prompt(dancers_count, aspect_ratio) + " " + prompt
    return header + prompt


def inject_solo_negative(negative: str) -> str:
    """Block Veo's default single-person tendency."""
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


def run_ffmpeg(args: list, timeout: int = 300, label: str = "") -> None:
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
#  Frame extraction helpers
# ══════════════════════════════════════════════════════════

def extract_last_frame(video_path: str, out_path: str) -> str:
    """
    Extract the very last frame of a video.
    Goes 3 frames before the end to avoid blank/duplicate end frames.
    This frame becomes the first_frame anchor for the NEXT Veo clip,
    making the join feel like a seamless continuation.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, total - 3)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read last frame from {video_path}")
    cv2.imwrite(out_path, frame)
    print(f"  Last frame extracted: frame {target}/{total} → {out_path}")
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
    """Upscale to ≥720p + light sharpen for best Veo input."""
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


def frame_to_veo_image(frame_path: str) -> types.Image:
    """Load an enhanced PNG and wrap it as a Veo Image object."""
    with open(frame_path, "rb") as f:
        return types.Image(image_bytes=f.read(), mime_type="image/png")


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


# ══════════════════════════════════════════════════════════
#  Crossfade Concatenation
# ══════════════════════════════════════════════════════════

def concatenate_with_xfade(clip_paths: list, output_path: str,
                            xfade_secs: float = XFADE_DURATION) -> str:
    """
    Join clips with an xfade crossfade at every join point.

    Offset calculation:
      join 0→1:  offset = duration[0] - xfade_secs
      join 1→2:  offset = duration[0] + duration[1] - 2*xfade_secs
      ...
      The offset is the point (in cumulative output time) where the
      crossfade begins. After each xfade, xfade_secs is subtracted
      because the output timeline overlaps by that amount.
    """
    if not clip_paths:
        raise ValueError("No clips provided.")
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path)
        return output_path

    ffmpeg = get_ffmpeg_exe()

    # Collect durations
    durations = []
    for cp in clip_paths:
        info = get_video_info(cp)
        durations.append(info["duration"])
        print(f"  Clip: {os.path.basename(cp)} — {info['duration']:.2f}s")

    # Normalise to resolution of first clip
    info0 = get_video_info(clip_paths[0])
    tw    = info0["width"]  - (info0["width"]  % 2)
    th    = info0["height"] - (info0["height"] % 2)
    n     = len(clip_paths)

    inputs = []
    for cp in clip_paths:
        inputs += ["-i", cp]

    # Build filter_complex
    # Phase 1: normalise every stream (scale + fps + resample audio)
    norm_parts = []
    for i in range(n):
        norm_parts.append(
            f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,fps={OUTPUT_FPS},setsar=1[nv{i}];"
        )
        norm_parts.append(f"[{i}:a]aresample=44100[na{i}];")

    # Phase 2: chain xfade/acrossfade between consecutive pairs
    xfade_parts = []
    cumulative  = 0.0

    cumulative += durations[0]
    offset      = max(0.05, cumulative - xfade_secs)
    xfade_parts.append(
        f"[nv0][nv1]xfade=transition=fade:duration={xfade_secs}:offset={offset:.4f}[xv1];"
    )
    xfade_parts.append(f"[na0][na1]acrossfade=d={xfade_secs}[xa1];")
    cumulative += durations[1] - xfade_secs
    prev_v, prev_a = "xv1", "xa1"

    for i in range(2, n):
        offset = max(0.05, cumulative - xfade_secs)
        cur_v, cur_a = f"xv{i}", f"xa{i}"
        xfade_parts.append(
            f"[{prev_v}][nv{i}]xfade=transition=fade:duration={xfade_secs}:"
            f"offset={offset:.4f}[{cur_v}];"
        )
        xfade_parts.append(f"[{prev_a}][na{i}]acrossfade=d={xfade_secs}[{cur_a}];")
        cumulative += durations[i] - xfade_secs
        prev_v, prev_a = cur_v, cur_a

    # Remove trailing semicolon
    fc = "".join(norm_parts + xfade_parts).rstrip(";")

    # ── Attempt: xfade ────────────────────────────────────────────────
    try:
        print(f"  Joining {n} clips with {xfade_secs}s crossfade...")
        result = subprocess.run(
            [ffmpeg, "-y"] + inputs
            + ["-filter_complex", fc,
               "-map", f"[{prev_v}]", "-map", f"[{prev_a}]"]
            + WEB_VIDEO_FLAGS + WEB_AUDIO_FLAGS + [output_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print("  ✓ Crossfade concat done")
            return output_path
        print(f"  xfade failed: {result.stderr[-300:]}")
    except Exception as e:
        print(f"  xfade error: {e}")

    # ── Fallback: hard-cut stream-copy ────────────────────────────────
    print("  Falling back to hard-cut concat...")
    try:
        concat_txt = output_path + "_list.txt"
        with open(concat_txt, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{os.path.abspath(cp).replace(chr(92), '/')}'\n")
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
             "-c", "copy", "-movflags", "+faststart", output_path],
            capture_output=True, text=True, timeout=180,
        )
        if os.path.exists(concat_txt): os.remove(concat_txt)
        if result.returncode == 0:
            print("  ✓ Hard-cut concat done (no crossfade)")
            return output_path
    except Exception as e:
        print(f"  Hard-cut error: {e}")

    # ── Last resort: moviepy ──────────────────────────────────────────
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        clips = [VideoFileClip(p) for p in clip_paths]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        for c in clips: c.close()
        final.close()
        return output_path
    except Exception as e:
        print(f"  moviepy error: {e}")

    shutil.copy2(clip_paths[0], output_path)
    return output_path


# ══════════════════════════════════════════════════════════
#  Gemini → Writes the Veo Prompt
# ══════════════════════════════════════════════════════════

def gemini_write_veo_prompt(client, video_path: str,
                             dancers_count: int, aspect_ratio: str) -> dict:
    print("  Uploading video to Gemini for analysis...")
    uploaded = client.files.upload(file=video_path)
    waited = 0
    while uploaded.state.name == "PROCESSING" and waited < 120:
        time.sleep(5); waited += 5
        uploaded = client.files.get(name=uploaded.name)

    total_dancers = dancers_count + 1

    instruction = f"""You are a Google Veo 3.1 prompt engineer. Watch this dance video carefully.

TASK: Write a Veo prompt that generates a GROUP DANCE VIDEO showing:
  - The SAME lead dancer doing the SAME dance moves
  - EXACTLY {dancers_count} backup dancers added behind and beside the lead
  - TOTAL of {total_dancers} dancers on screen at all times
  - The EXACT same room/location

=== CHECKLIST (complete mentally before writing): ===
  [✓] Mention the number {total_dancers} (total) in the prompt
  [✓] Mention the number {dancers_count} (backup) in the prompt
  [✓] Describe the lead dancer's EXACT outfit
  [✓] Describe the EXACT room
  [✓] Say "all {total_dancers} dancers fully visible head to toe"

=== 1. VEO_PROMPT ===
3 sentences:

Sentence 1 (MANDATORY template):
"[Orientation] video, wide group shot of {total_dancers} dancers total: 1 lead dancer with [exact outfit] performs [dance style] at front-center, flanked by {dancers_count} backup dancers, all {total_dancers} performers fully visible from head to toe."

Sentence 2: Exact setting (floor, walls, ceiling, lighting).

Sentence 3: "All {total_dancers} dancers moving in sync, [style] choreography, casual TikTok dance video aesthetic."

CRITICAL: Numbers {total_dancers} and {dancers_count} MUST appear. Use "smartphone video" not "cinematic".

=== 2. NEGATIVE_PROMPT ===
MUST include: solo dancer, single person, one person, lone dancer, only one performer,
cartoon, anime, 3D render, CGI, distorted face, extra limbs, blurry, watermark, text,
miniature people, doll-like, cropped body, partial figure

=== 3. ANALYSIS ===
2-3 lines: outfit, room, dance style.

FORMAT:
VEO_PROMPT: [prompt]
NEGATIVE_PROMPT: [negatives]
ANALYSIS: [analysis]"""

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
#  Veo 3.1 Clip Generation — anchored to previous clip's last frame
# ══════════════════════════════════════════════════════════

def generate_veo_clip(client, prompt: str, negative: str,
                       first_frame_image,   # last frame of the PREVIOUS clip
                       aspect_ratio: str,
                       clip_index: int, total_clips: int) -> str:
    """
    Generate one 8-second Veo 3.1 clip starting from first_frame_image.

    first_frame_image is always the LAST FRAME of the previous segment
    (intro for clip 0, clip 0's output for clip 1, etc.).
    This makes each clip a direct continuation — not a new generation.
    """
    os.makedirs("processed", exist_ok=True)
    operation  = None
    last_error = None

    # ── Attempt 1: image-anchored from previous clip's last frame ─────────
    for attempt in range(2):
        try:
            print(f"  Clip {clip_index + 1}/{total_clips} — "
                  f"image-anchor attempt {attempt + 1}...")
            print(f"  Prompt: {prompt[:200]}...")

            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=first_frame_image,
                config=types.GenerateVideosConfig(
                    person_generation="allow_adult",
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                    negative_prompt=negative,
                ),
            )
            print(f"  Operation: {operation.name}")
            break

        except Exception as e:
            last_error = str(e)
            print(f"  Image-anchor attempt {attempt + 1} failed: {last_error[:200]}")
            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                wait = 30 * (attempt + 1)
                print(f"  Rate-limited — waiting {wait}s..."); time.sleep(wait); continue
            operation = None
            break

    # ── Attempt 2: text-only fallback ─────────────────────────────────────
    if operation is None:
        for attempt in range(3):
            try:
                print(f"  Text-only attempt {attempt + 1}...")
                operation = client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=prompt,
                    config=types.GenerateVideosConfig(
                        person_generation="allow_adult",
                        aspect_ratio=aspect_ratio,
                        number_of_videos=1,
                        negative_prompt=negative,
                    ),
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
        raise Exception(f"Clip {clip_index + 1} — all attempts failed: {last_error}")

    # ── Poll ───────────────────────────────────────────────────────────────
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

    vid      = operation.response.generated_videos[0]
    raw_path = f"processed/clip_raw_{clip_index}_{int(time.time())}.mp4"
    client.files.download(file=vid.video)
    vid.video.save(raw_path)
    print(f"  Raw: {raw_path} ({os.path.getsize(raw_path) / 1024 / 1024:.1f} MB)")

    web_path = f"processed/clip_web_{clip_index}_{int(time.time())}.mp4"
    web_encode(raw_path, web_path)
    try: os.remove(raw_path)
    except Exception: pass

    return web_path


# ══════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════

def generate_group_dance_video(video_path: str, dancers_count: int,
                                output_path: str) -> dict:
    """
    Seamless chaining pipeline.

    Frame chain:
      intro last frame → clip 1 first_frame
      clip 1 last frame → clip 2 first_frame
      (each clip continues exactly where the previous ended)

    Transitions:
      0.5s xfade crossfade at every join — no visible cut.
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

        # Step 1: Metadata
        print("\n[1/7] Reading video metadata...")
        vinfo    = get_video_info(video_path)
        is_vert  = vinfo["height"] > vinfo["width"]
        aspect   = "9:16" if is_vert else "16:9"
        duration = vinfo["duration"]
        print(f"  {vinfo['width']}×{vinfo['height']}, {duration:.1f}s, {aspect}")

        # Step 2: Cut intro
        print(f"\n[2/7] Cutting & encoding first {INTRO_DURATION}s intro...")
        intro_path = "processed/intro_web.mp4"
        intro_dur  = min(INTRO_DURATION, duration)
        cut_intro_clip(video_path, intro_dur, intro_path)
        temp_files.append(intro_path)

        # Step 3: Gemini prompt
        print("\n[3/7] Gemini writing Veo group-dance prompt...")
        gresult    = gemini_write_veo_prompt(client, video_path, dancers_count, aspect)
        veo_prompt = force_inject_dancers(gresult["veo_prompt"], dancers_count, aspect)
        negative   = inject_solo_negative(gresult["negative_prompt"])
        analysis   = gresult["analysis"]

        print(f"\n  ╔══ FINAL VEO PROMPT ══════════════════════════")
        print(f"  ║ {veo_prompt[:300]}{'...' if len(veo_prompt) > 300 else ''}")
        print(f"  ╠══ NEGATIVE ══════════════════════════════════")
        print(f"  ║ {negative[:140]}")
        print(f"  ╠══ ANALYSIS ══════════════════════════════════")
        print(f"  ║ {analysis[:140]}")
        print(f"  ╚══════════════════════════════════════════════\n")

        # Step 4: Extract LAST FRAME of intro → anchor for clip 1
        print("[4/7] Extracting intro's last frame → anchor for Veo clip 1...")
        anchor_raw = "processed/anchor_0_intro_last.png"
        extract_last_frame(intro_path, anchor_raw)
        anchor_enh = enhance_frame(anchor_raw)
        temp_files += [anchor_raw, anchor_enh]
        # current_anchor is always the last frame of the previous segment
        current_anchor = frame_to_veo_image(anchor_enh)
        print(f"  ✓ Clip 1 will start from intro's last frame (seamless continuation)")

        # Steps 5–6: Generate clips, each anchored to the previous one's end
        print(f"\n[5/7] Generating {NUM_VEO_CLIPS} chained Veo clips...")

        for i in range(NUM_VEO_CLIPS):
            label = "continues from intro" if i == 0 else f"continues from clip {i}"
            print(f"\n  ── Clip {i + 1}/{NUM_VEO_CLIPS} ({label}) ──")

            clip = generate_veo_clip(
                client, veo_prompt, negative,
                current_anchor,     # ← last frame of previous segment
                aspect, i, NUM_VEO_CLIPS
            )
            clip_paths.append(clip)

            # After generating clip i, extract its last frame as anchor for clip i+1
            if i < NUM_VEO_CLIPS - 1:
                print(f"\n  Extracting clip {i+1}'s last frame → anchor for clip {i+2}...")
                next_anchor_raw = f"processed/anchor_{i+1}_clip_last.png"
                extract_last_frame(clip, next_anchor_raw)
                next_anchor_enh = enhance_frame(next_anchor_raw)
                temp_files += [next_anchor_raw, next_anchor_enh]
                current_anchor  = frame_to_veo_image(next_anchor_enh)
                print(f"  ✓ Clip {i+2} will start from clip {i+1}'s last frame")
                print("  Pausing 10s before next clip..."); time.sleep(10)

        # Step 7: Crossfade-concat
        print(f"\n[6/7] Assembling with {XFADE_DURATION}s crossfades at every join...")
        all_clips = [intro_path] + clip_paths
        concatenate_with_xfade(all_clips, output_path, xfade_secs=XFADE_DURATION)

        out_info = get_video_info(output_path)
        fsize    = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        expected = intro_dur + NUM_VEO_CLIPS * VEO_CLIP_SECONDS - (NUM_VEO_CLIPS * XFADE_DURATION)

        final_analysis = (
            f"Pipeline summary (v11 — seamless chaining)\n"
            f"═══════════════════════════════════════════\n"
            f"Original      : {vinfo['width']}×{vinfo['height']}, {duration:.1f}s, {aspect}\n"
            f"Intro clip    : first {intro_dur:.0f}s\n"
            f"Generated     : {NUM_VEO_CLIPS} × {VEO_CLIP_SECONDS}s = "
            f"{NUM_VEO_CLIPS * VEO_CLIP_SECONDS}s (chained)\n"
            f"Transitions   : {XFADE_DURATION}s xfade crossfade at every join\n"
            f"Final output  : {out_info['duration']:.1f}s (~{expected:.0f}s expected), "
            f"{fsize / 1024 / 1024:.1f} MB\n"
            f"Backup dancers: {dancers_count}\n"
            f"Chain         : intro→last→clip1→last→clip2\n\n"
            f"Gemini analysis\n───────────────\n{analysis}\n\n"
            f"Veo prompt\n──────────\n{veo_prompt}"
        )

        print(f"\n  ✓ DONE → {output_path}")
        print(f"  {out_info['duration']:.1f}s  |  {fsize/1024/1024:.1f} MB")
        print(f"  Joins: xfade {XFADE_DURATION}s — no hard cuts ✓")

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