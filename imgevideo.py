"""
imgevideo.py — Image → Dance Video  (with face swap post-processing)
=====================================================================

WHY THE OLD APPROACH FAILED
────────────────────────────
Prompt engineering + image anchoring does NOT reliably preserve faces in Veo.
Veo is a generative video model — it interprets the reference image as a
style/scene hint, not a biometric template. The face drifts or becomes
unrecognisable no matter how detailed the prompt is.

THE CORRECT APPROACH
────────────────────
Step 1 — Let Veo generate the dance video freely (great motion, wrong face).
Step 2 — Post-process: use InsightFace's inswapper_128 model to swap the
          REFERENCE FACE onto EVERY FRAME of the generated video.

Result: Veo's fluid dance motion + the person's actual face from the photo.

InsightFace pipeline:
  • buffalo_l       — face detection + 512-d ArcFace embedding
  • inswapper_128   — ONNX face-swap model (128×128, identity-preserving)

Both models are downloaded automatically on first run to ~/.insightface/

Dependencies:
  pip install insightface onnxruntime-gpu   # GPU (faster)
  pip install insightface onnxruntime       # CPU (slower, ~1-3 fps)

Full pipeline:
  1. Load + enhance reference photo
  2. Gemini writes a dance-style Veo prompt (body/scene only, not face)
  3. Generate 2 × 8s Veo clips → 16s raw dance video
  4. Face swap: replace every face in every frame with the reference face
  5. Re-encode swapped frames → web-safe H.264 MP4
"""

import os
import time
import shutil
import subprocess
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

import choreography_library

# ── Constants ─────────────────────────────────────────────
VEO_CLIP_SECONDS  = 8
NUM_CLIPS         = 2        # 2 × 8s = 16 s total
OUTPUT_FPS        = 30
XFADE_DURATION    = 0.5
FACE_SWAP_BATCH   = 30       # process N frames at a time (memory control)

DANCE_STYLES = [
    "hip-hop",
    "breakdancing / b-boy",
    "popping & locking",
    "krump",
    "waacking",
    "vogueing",
    "contemporary",
    "ballet",
    "salsa",
    "bachata",
    "twerk",
    "K-pop",
    "bollywood",
    "afrobeats",
    "jazz funk",
    "tutting",
]

WEB_VIDEO_FLAGS = [
    "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
    "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
    "-movflags", "+faststart",
]
WEB_AUDIO_FLAGS = [
    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
]
# ─────────────────────────────────────────────────────────


def get_client():
    if not GOOGLE_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file.")
    return genai.Client(api_key=GOOGLE_API_KEY)


# ══════════════════════════════════════════════════════════
#  InsightFace — face swap engine
# ══════════════════════════════════════════════════════════

def load_face_swap_models():
    """
    Load InsightFace buffalo_l (detection) and inswapper_128 (swap).
    Models are downloaded automatically on first run.
    Returns (app, swapper) or raises ImportError with install instructions.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
    except ImportError:
        raise ImportError(
            "insightface not installed.\n"
            "Run:  pip install insightface onnxruntime\n"
            "GPU:  pip install insightface onnxruntime-gpu"
        )

    print("  Loading InsightFace buffalo_l (detection)...")
    face_app = FaceAnalysis(name="buffalo_l",
                            providers=["CUDAExecutionProvider",
                                       "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("  Loading inswapper_128 (face swap)...")
    # Model path: ~/.insightface/models/inswapper_128.onnx
    model_dir  = Path.home() / ".insightface" / "models"
    model_path = model_dir / "inswapper_128.onnx"

    if not model_path.exists():
        print("  inswapper_128.onnx not found — downloading (~500 MB)...")
        model_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request

        # Primary source: HuggingFace
        urls = [
            "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
        ]
        downloaded = False
        for url in urls:
            try:
                print(f"  Trying: {url}")
                urllib.request.urlretrieve(url, str(model_path))
                print(f"  ✓ Downloaded to {model_path}")
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed ({e}) — trying next source...")

        if not downloaded:
            raise RuntimeError(
                "Could not auto-download inswapper_128.onnx.\n"
                "Please download it manually from:\n"
                "  https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx\n"
                f"And place it at: {model_path}"
            )

    swapper = get_model(str(model_path),
                        providers=["CUDAExecutionProvider",
                                   "CPUExecutionProvider"])
    print("  ✓ Face swap models loaded")
    return face_app, swapper


def get_reference_face(face_app, image_path: str):
    """
    Detect and return the largest face in the reference photo.
    This face object contains the 512-d ArcFace embedding used for swapping.
    """
    img   = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read reference image: {image_path}")
    faces = face_app.get(img)
    if not faces:
        raise ValueError(
            "No face detected in the reference photo. "
            "Please use a clear front-facing photo with good lighting."
        )
    # Pick the largest detected face (most prominent)
    ref_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    print(f"  ✓ Reference face detected  "
          f"(bbox area: {int((ref_face.bbox[2]-ref_face.bbox[0])*(ref_face.bbox[3]-ref_face.bbox[1]))}px²)")
    return ref_face


def swap_faces_in_video(raw_video_path: str, ref_face,
                         face_app, swapper,
                         output_path: str) -> str:
    """
    Read every frame of raw_video_path, detect all faces, replace each with
    ref_face using inswapper_128, write swapped frames to output_path.

    Returns output_path (web-safe H.264).
    """
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {raw_video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write frames to a lossless temporary AVI (avoid re-encoding artefacts mid-pipeline)
    tmp_avi = output_path.replace(".mp4", "_raw_swap.avi")
    fourcc  = cv2.VideoWriter_fourcc(*"XVID")
    writer  = cv2.VideoWriter(tmp_avi, fourcc, fps, (w, h))

    print(f"  Face-swapping {total} frames  ({w}×{h} @ {fps:.0f}fps)...")
    swapped_count = 0
    no_face_count = 0
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_app.get(frame)
        if faces:
            # Swap every detected face with the reference
            result = frame.copy()
            for face in faces:
                result = swapper.get(result, face, ref_face, paste_back=True)
            writer.write(result)
            swapped_count += 1
        else:
            # No face detected in this frame — write as-is
            writer.write(frame)
            no_face_count += 1

        frame_idx += 1
        if frame_idx % 60 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  Swap progress: {frame_idx}/{total} ({pct:.0f}%)  "
                  f"swapped={swapped_count}  no-face={no_face_count}")

    cap.release()
    writer.release()
    print(f"  ✓ Swap done: {swapped_count} frames swapped, "
          f"{no_face_count} frames had no detectable face")

    # Extract audio from original (swapper only touches video frames)
    print("  Re-encoding swapped frames to web-safe H.264...")
    ffmpeg = get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-i", tmp_avi,           # swapped video (no audio)
            "-i", raw_video_path,    # original (for audio track)
            "-map", "0:v",
            "-map", "1:a?",          # optional — Veo clips may have no audio
            "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={OUTPUT_FPS}",
        ]
        + WEB_VIDEO_FLAGS + WEB_AUDIO_FLAGS
        + [output_path],
        capture_output=True, text=True, timeout=600,
    )
    try:
        os.remove(tmp_avi)
    except Exception:
        pass

    if result.returncode != 0:
        raise RuntimeError(f"Re-encode after face swap failed:\n{result.stderr[-400:]}")

    sz = os.path.getsize(output_path)
    print(f"  Face-swapped output: {output_path}  ({sz/1024/1024:.1f} MB)")
    return output_path


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
    result = subprocess.run([ffmpeg, "-y"] + args,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg{' (' + label + ')' if label else ''} failed:\n"
            f"{result.stderr[-600:]}"
        )


def web_encode(src: str, dst: str) -> str:
    run_ffmpeg(
        ["-i", src,
         "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={OUTPUT_FPS}"]
        + WEB_VIDEO_FLAGS + WEB_AUDIO_FLAGS + [dst],
        label="web_encode"
    )
    print(f"  Web-encoded: {dst}  ({os.path.getsize(dst)/1024/1024:.1f} MB)")
    return dst


def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"fps": 30, "width": 1080, "height": 1920, "duration": 8}
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "width": w, "height": h, "duration": total / fps}


# ══════════════════════════════════════════════════════════
#  Image helpers
# ══════════════════════════════════════════════════════════

def load_and_enhance_image(image_path: str, tmp_dir: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    if min(h, w) < 720:
        scale = 720 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[0, -0.25, 0], [-0.25, 2.0, -0.25], [0, -0.25, 0]])
    img    = cv2.filter2D(img, -1, kernel)
    out    = os.path.join(tmp_dir, "ref_enhanced.png")
    cv2.imwrite(out, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"  Enhanced ref image: {img.shape[1]}×{img.shape[0]}  → {out}")
    return out


def extract_last_frame(video_path: str, out_path: str) -> str:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 3))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read last frame: {video_path}")
    cv2.imwrite(out_path, frame)
    return out_path


def image_bytes_from_path(path: str) -> types.Image:
    with open(path, "rb") as f:
        return types.Image(image_bytes=f.read(), mime_type="image/png")


# ══════════════════════════════════════════════════════════
#  Gemini — writes a BODY/SCENE Veo prompt (not face)
# ══════════════════════════════════════════════════════════

def gemini_write_dance_prompt(client, image_path: str,
                               dance_style: str,
                               dancers_count: int,
                               aspect_ratio: str) -> dict:
    """
    Gemini looks at the photo and writes a Veo prompt focused on:
      - The outfit / clothing (so Veo generates the RIGHT clothes)
      - The setting / environment
      - The dance style and energy
    We do NOT try to describe or preserve the face — the face swap handles that.
    """
    print("  Uploading image to Gemini for outfit/scene analysis...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    total = dancers_count + 1
    orient = "Vertical smartphone video" if aspect_ratio == "9:16" else "Horizontal widescreen video"
    backup_line = (
        f"with {dancers_count} backup dancers performing synchronized moves "
        f"behind and beside them, all {total} dancers fully visible from head to toe,"
        if dancers_count > 0 else "full body visible from head to toe,"
    )

    choreo_moves = choreography_library.get_choreo(dance_style)

    instruction = f"""You are a Veo 3.1 prompt engineer specialising in dance videos.
Look at this photo and write a prompt for a {dance_style} dance video.

FOCUS ON:
  1. The person's EXACT OUTFIT (every garment, colour, pattern, footwear, accessories)
  2. A real {dance_style}-appropriate environment (specific floor type, walls, lighting)
  3. SPECIFIC choreographic moves listed below — these MUST appear in the prompt

CHOREOGRAPHY MOVES TO INCLUDE:
{choreo_moves}

DO NOT describe the face, skin tone, or facial features.

Write exactly 3 sentences:

Sentence 1:
"{orient}, wide full-body shot. A dancer wearing [EXACT OUTFIT from photo] is {choreo_moves[:200]}, {backup_line}."

Sentence 2: The setting — one sentence describing a real {dance_style}-appropriate venue: floor surface, wall, lighting.

Sentence 3: "{'All ' + str(total) + ' performers in perfect sync, ' if dancers_count > 0 else ''}high-energy {dance_style} performance, smooth tracking camera, cinematic 24fps, TikTok dance video aesthetic, professional choreography."

Also write:
NEGATIVE_PROMPT: static pose, standing still, walking, cartoon, anime, 3D render, CGI, extra limbs, distorted body, blurry face, watermark, text, miniature people, doll-like, cropped body, partial figure, {"solo dancer, single person," if dancers_count > 0 else ""}low quality

FORMAT:
VEO_PROMPT: [prompt]
NEGATIVE_PROMPT: [negatives]
OUTFIT_DESCRIPTION: [outfit only, 1-2 sentences]"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[image_part, instruction]
        )
        raw    = resp.text
        result = {"veo_prompt": "", "negative_prompt": "", "outfit_description": "", "raw": raw}
        cur    = None
        for line in raw.split("\n"):
            s = line.strip(); u = s.upper()
            if u.startswith("VEO_PROMPT"):
                cur = "veo_prompt";          result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif u.startswith("NEGATIVE_PROMPT"):
                cur = "negative_prompt";     result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif u.startswith("OUTFIT_DESCRIPTION"):
                cur = "outfit_description";  result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif cur and s:
                result[cur] += " " + s
        for k in result:
            if isinstance(result[k], str):
                result[k] = result[k].strip()

        if not result["veo_prompt"]:
            result["veo_prompt"] = (
                f"{orient}, wide full-body shot. A dancer performs dynamic "
                f"{dance_style} choreography, {backup_line} expressive fluid movement. "
                f"Indoor dance studio with hardwood floor and warm stage lighting. "
                f"Energetic {dance_style} moves, smooth camera, TikTok aesthetic."
            )
        if not result["negative_prompt"]:
            result["negative_prompt"] = (
                "cartoon, anime, 3D render, CGI, extra limbs, distorted body, "
                "blurry, watermark, text, miniature, doll-like, cropped body"
            )
        print(f"  Gemini analysis done ({len(raw)} chars)")
        return result

    except Exception as e:
        print(f"  Gemini failed: {e}")
        return {
            "veo_prompt": (
                f"{orient}, wide full-body shot. A dancer performs energetic "
                f"{dance_style} choreography, {backup_line} fluid expressive movement. "
                f"Indoor studio, hardwood floor, warm lighting. TikTok dance aesthetic."
            ),
            "negative_prompt": (
                "cartoon, anime, 3D render, CGI, extra limbs, blurry, "
                "watermark, text, miniature, doll-like, cropped body"
            ),
            "outfit_description": "",
            "raw": ""
        }


# ══════════════════════════════════════════════════════════
#  Veo clip generation
# ══════════════════════════════════════════════════════════

def generate_veo_clip(client, prompt: str, negative: str,
                       anchor_image, aspect_ratio: str,
                       clip_index: int, total_clips: int) -> str:
    """Generate one raw 8-second Veo clip (no face swap yet)."""
    os.makedirs("processed", exist_ok=True)
    operation  = None
    last_error = None

    for attempt in range(2):
        try:
            print(f"  Veo clip {clip_index+1}/{total_clips} — attempt {attempt+1}...")
            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=anchor_image,
                config=types.GenerateVideosConfig(
                    person_generation="allow_adult",
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                    negative_prompt=negative,
                ),
            )
            print(f"  Operation: {operation.name}"); break
        except Exception as e:
            last_error = str(e)
            print(f"  Attempt {attempt+1} failed: {last_error[:200]}")
            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                time.sleep(30 * (attempt + 1)); continue
            operation = None; break

    # text-only fallback
    if operation is None:
        for attempt in range(3):
            try:
                print(f"  Text-only attempt {attempt+1}...")
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
                print(f"  Text-only op: {operation.name}"); break
            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                    time.sleep(30 * (attempt + 1))
                else:
                    break

    if operation is None:
        raise Exception(f"Clip {clip_index+1} failed: {last_error}")

    print("  Waiting for Veo 3.1...")
    polls = 0
    while not operation.done:
        polls += 1; time.sleep(10)
        operation = client.operations.get(operation)
        if polls % 3 == 0:
            print(f"  Generating... ({polls*10}s)")
        if polls > 72:
            raise Exception("Veo timed out")

    print(f"  ✓ Veo clip {clip_index+1} done!")
    
    # Defensive check for response
    if not operation.response or not operation.response.generated_videos:
        err_msg = f"Veo generation failed. Error: {operation.error}" if operation.error else "Veo generation failed: No video returned (possibly blocked by safety filters)."
        raise Exception(err_msg)

    vid      = operation.response.generated_videos[0]
    raw_path = f"processed/imgvid_raw_{clip_index}_{int(time.time())}.mp4"
    client.files.download(file=vid.video)
    vid.video.save(raw_path)
    print(f"  Saved raw: {raw_path} ({os.path.getsize(raw_path)/1024/1024:.1f} MB)")
    return raw_path   # return RAW — face swap happens later


# ══════════════════════════════════════════════════════════
#  Crossfade concat (operates on already-encoded clips)
# ══════════════════════════════════════════════════════════

def concatenate_with_xfade(clip_paths: list, output_path: str,
                            xfade_secs: float = XFADE_DURATION) -> str:
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path); return output_path

    ffmpeg    = get_ffmpeg_exe()
    durations = [get_video_info(cp)["duration"] for cp in clip_paths]
    info0     = get_video_info(clip_paths[0])
    tw = info0["width"]  - (info0["width"]  % 2)
    th = info0["height"] - (info0["height"] % 2)
    n  = len(clip_paths)

    inputs = []
    for cp in clip_paths: inputs += ["-i", cp]

    norm = []
    for i in range(n):
        norm.append(
            f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,fps={OUTPUT_FPS},setsar=1[nv{i}];"
        )
        norm.append(f"[{i}:a]aresample=44100[na{i}];")

    xf  = []
    cum = durations[0]
    off = max(0.05, cum - xfade_secs)
    xf += [f"[nv0][nv1]xfade=transition=fade:duration={xfade_secs}:offset={off:.4f}[xv1];",
           f"[na0][na1]acrossfade=d={xfade_secs}[xa1];"]
    cum += durations[1] - xfade_secs
    pv, pa = "xv1", "xa1"

    for i in range(2, n):
        off  = max(0.05, cum - xfade_secs)
        cv_, ca_ = f"xv{i}", f"xa{i}"
        xf += [f"[{pv}][nv{i}]xfade=transition=fade:duration={xfade_secs}:offset={off:.4f}[{cv_}];",
               f"[{pa}][na{i}]acrossfade=d={xfade_secs}[{ca_}];"]
        cum += durations[i] - xfade_secs
        pv, pa = cv_, ca_

    fc = "".join(norm + xf).rstrip(";")
    try:
        result = subprocess.run(
            [ffmpeg, "-y"] + inputs
            + ["-filter_complex", fc, "-map", f"[{pv}]", "-map", f"[{pa}]"]
            + WEB_VIDEO_FLAGS + WEB_AUDIO_FLAGS + [output_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print("  ✓ Crossfade concat done"); return output_path
        print(f"  xfade failed, hard-cut fallback: {result.stderr[-200:]}")
    except Exception as e:
        print(f"  xfade error: {e}")

    concat_txt = output_path + "_list.txt"
    with open(concat_txt, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.abspath(cp).replace(chr(92),'/')}'\n")
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
         "-c", "copy", "-movflags", "+faststart", output_path],
        capture_output=True, text=True, timeout=180,
    )
    if os.path.exists(concat_txt): os.remove(concat_txt)
    return output_path


# ══════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════

def generate_dance_from_image(image_path: str, dance_style: str,
                               dancers_count: int, output_path: str) -> dict:
    """
    Full pipeline:
      1  Enhance reference photo
      2  Detect reference face (InsightFace)
      3  Gemini writes outfit-based Veo prompt
      4  Generate Veo clip 1 (raw, wrong face)
      5  Generate Veo clip 2, anchored to clip 1's last frame (raw)
      6  Face-swap reference face onto EVERY frame of both clips
      7  Crossfade-concat swapped clips → final 16s web-safe MP4
    """
    if not GOOGLE_API_KEY:
        return {"status": "failed",
                "analysis": "GEMINI_API_KEY not set.",
                "generated_video_path": None}

    os.makedirs("processed", exist_ok=True)
    raw_clips     = []   # Veo raw outputs (before face swap)
    swapped_clips = []   # after face swap
    temp_files    = []

    try:
        client = get_client()

        # ── 1. Prepare image ─────────────────────────────────────────────
        print(f"\n[1/7] Preparing reference image...")
        ref_path = load_and_enhance_image(image_path, "processed")
        temp_files.append(ref_path)

        img_check = cv2.imread(ref_path)
        ih, iw    = img_check.shape[:2]
        aspect    = "9:16" if ih > iw else "16:9"
        print(f"  {iw}×{ih}  aspect={aspect}")

        # ── 2. Detect reference face ─────────────────────────────────────
        print("\n[2/7] Detecting reference face (InsightFace)...")
        face_app, swapper = load_face_swap_models()
        ref_face          = get_reference_face(face_app, ref_path)

        # ── 3. Gemini writes the Veo prompt ──────────────────────────────
        print(f"\n[3/7] Gemini writing '{dance_style}' prompt...")
        gresult   = gemini_write_dance_prompt(
            client, ref_path, dance_style, dancers_count, aspect
        )
        prompt   = gresult["veo_prompt"]
        negative = gresult["negative_prompt"]
        outfit   = gresult["outfit_description"]

        print(f"\n  ╔══ VEO PROMPT ════════════════════════════════")
        print(f"  ║ {prompt[:280]}")
        print(f"  ╠══ NEGATIVE ══════════════════════════════════")
        print(f"  ║ {negative[:140]}")
        print(f"  ╚══════════════════════════════════════════════\n")

        # ── 4. Generate Veo clip 1 ───────────────────────────────────────
        print("[4/7] Generating Veo clip 1/2...")
        ref_veo = image_bytes_from_path(ref_path)
        raw1    = generate_veo_clip(client, prompt, negative,
                                    ref_veo, aspect, 0, NUM_CLIPS)
        raw_clips.append(raw1)

        # ── 5. Generate Veo clip 2 (anchored to clip 1's last frame) ─────
        print("\n[5/7] Extracting clip 1's last frame → anchor for clip 2...")
        lf1_raw = "processed/imgvid_lf1.png"
        extract_last_frame(raw1, lf1_raw)
        # Enhance last frame slightly
        lf_img  = cv2.imread(lf1_raw)
        k       = np.array([[0, -0.25, 0], [-0.25, 2.0, -0.25], [0, -0.25, 0]])
        lf_img  = cv2.filter2D(lf_img, -1, k)
        lf1_enh = lf1_raw.replace(".png", "_enh.png")
        cv2.imwrite(lf1_enh, lf_img)
        temp_files += [lf1_raw, lf1_enh]

        # For clip 2, add "continuation" header to prompt
        cont_prompt = (
            f"CONTINUATION: same dancers, same room, no cuts, same outfit. "
            f"Continue {dance_style} choreography seamlessly. "
        ) + prompt

        print("  Generating Veo clip 2/2 (continuation)...")
        time.sleep(8)
        lf1_veo = image_bytes_from_path(lf1_enh)
        raw2    = generate_veo_clip(client, cont_prompt, negative,
                                    lf1_veo, aspect, 1, NUM_CLIPS)
        raw_clips.append(raw2)

        # ── 6. Face swap on both clips ────────────────────────────────────
        print("\n[6/7] Applying face swap to all generated clips...")
        for i, raw_clip in enumerate(raw_clips):
            swapped_path = f"processed/imgvid_swapped_{i}_{int(time.time())}.mp4"
            print(f"  Face-swapping clip {i+1}/{NUM_CLIPS}...")
            swap_faces_in_video(raw_clip, ref_face, face_app, swapper, swapped_path)
            swapped_clips.append(swapped_path)
            # Remove raw clip to save disk
            try: os.remove(raw_clip)
            except Exception: pass

        # ── 7. Concat swapped clips ───────────────────────────────────────
        print(f"\n[7/7] Assembling final video with {XFADE_DURATION}s crossfades...")
        concatenate_with_xfade(swapped_clips, output_path)

        out_info = get_video_info(output_path)
        fsize    = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        final_analysis = (
            f"Image-to-Dance Video (imgevideo.py — face swap pipeline)\n"
            f"══════════════════════════════════════════════════════════\n"
            f"Dance style     : {dance_style}\n"
            f"Backup dancers  : {dancers_count}\n"
            f"Face method     : InsightFace inswapper_128 (frame-by-frame swap)\n"
            f"Output          : {out_info['duration']:.1f}s, {fsize/1024/1024:.1f} MB\n"
            f"Encoding        : H.264/yuv420p/+faststart\n\n"
            f"Outfit detected\n───────────────\n{outfit}\n\n"
            f"Veo prompt used\n───────────────\n{prompt}"
        )

        print(f"\n  ✓ DONE → {output_path}  ({out_info['duration']:.1f}s, {fsize/1024/1024:.1f} MB)")

        for fp in temp_files:
            if os.path.exists(fp):
                try: os.remove(fp)
                except Exception: pass
        for cp in swapped_clips:
            if os.path.exists(cp) and os.path.abspath(cp) != os.path.abspath(output_path):
                try: os.remove(cp)
                except Exception: pass

        return {"status": "completed", "analysis": final_analysis,
                "generated_video_path": output_path}

    except Exception as e:
        import traceback; traceback.print_exc()
        for fp in temp_files:
            if os.path.exists(fp):
                try: os.remove(fp)
                except Exception: pass
        for cp in raw_clips + swapped_clips:
            if os.path.exists(cp):
                try: os.remove(cp)
                except Exception: pass
        return {"status": "failed",
                "analysis": f"Image-to-video generation failed: {e}",
                "generated_video_path": None}


# ══════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    print("Available dance styles:", ", ".join(DANCE_STYLES))
    if len(sys.argv) < 4:
        print("Usage: python imgevideo.py <image> <dance_style> <backup_dancers> [output.mp4]")
        sys.exit(1)
    img     = sys.argv[1]
    style   = sys.argv[2]
    dancers = int(sys.argv[3])
    out     = sys.argv[4] if len(sys.argv) > 4 else "imgvid_output.mp4"
    result  = generate_dance_from_image(img, style, dancers, out)
    print(f"\nStatus : {result['status']}")
    print(f"Output : {result['generated_video_path']}")
    print(f"\n{result['analysis']}")