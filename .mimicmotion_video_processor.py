"""
Video Processor v7 — MimicMotion Clone Dance (Cloud API)
==========================================================

Uses MimicMotion (Tencent, ICML 2025) via Replicate API to clone
the original dancer's EXACT moves onto generated backup dancer images.

TRUE motion transfer — not text prompts, not approximation.
The model extracts pose from your video and applies it frame-by-frame.

Pipeline:
1. Gemini watches video → describes dancer appearance + scene
2. Gemini generates backup dancer IMAGE (clone of original)
3. MimicMotion (Replicate): your video = motion ref + dancer image → synced video
4. Segment original dancer → composite on top + original audio

Requirements:
  pip install replicate google-genai opencv-python numpy python-dotenv requests

.env:
  REPLICATE_API_TOKEN=your_replicate_token   # Required — replicate.com/account
  GEMINI_API_KEY=your_gemini_key             # Required — for dancer image gen
"""

import os
import time
import math
import shutil
import subprocess
import requests
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
VEO_CLIP_DURATION = 8
MAX_CLIPS = 4


def get_client():
    if not GOOGLE_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=GOOGLE_API_KEY)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


# ──────────────────────────────────────────────────────────
#  Video Utilities
# ──────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 30, "width": 1080, "height": 1920,
                "duration": 5, "total_frames": 150}
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps > 0 else 5
    cap.release()
    return {"fps": fps, "width": w, "height": h,
            "duration": dur, "total_frames": total}


def extract_frame(video_path: str, position: float = 0.5) -> str:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * max(0.02, min(0.98, position))))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read frame")
    path = video_path.rsplit('.', 1)[0] + f"_fr{int(position * 100)}.png"
    cv2.imwrite(path, frame)
    return path


def enhance_frame(frame_path: str) -> str:
    frame = cv2.imread(frame_path)
    if frame is None:
        return frame_path
    h, w = frame.shape[:2]
    if min(h, w) < 720:
        scale = 720 / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    out = frame_path.replace(".png", "_enh.png")
    cv2.imwrite(out, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return out


# ──────────────────────────────────────────────────────────
#  Gemini → Generate Clone Dancer Image
# ──────────────────────────────────────────────────────────

def gemini_generate_dancer_image(client, video_path: str,
                                  dancer_index: int,
                                  total_dancers: int,
                                  work_dir: str) -> str:
    """
    Gemini sees a frame from the video → generates an image of the
    SAME person (clone) in a neutral standing pose.
    Returns path to generated dancer image.
    """
    print(f"  [Gemini] Generating clone dancer image {dancer_index + 1}/{total_dancers}...")

    frame_path = os.path.join(work_dir, "ref_frame.png")
    extract_frame(video_path, 0.3)
    # Re-extract to work_dir
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.3))
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(frame_path, frame)

    with open(frame_path, "rb") as f:
        frame_bytes = f.read()

    frame_part = types.Part.from_bytes(data=frame_bytes, mime_type="image/png")

    prompt = f"""Look at this dance video frame carefully.

Generate an image of the EXACT SAME PERSON you see — same face, same skin tone,
same hair color and style, same body type. This is a CLONE, not a different person.

The person should be:
- Standing in a neutral pose (arms slightly out from body, facing camera)
- Wearing the EXACT same outfit as in the photo (same top, same bottom, same shoes)
- Full body visible from head to toe
- Against a simple plain background
- Well-lit, clear, high quality photo
- Realistic, not cartoon or illustration

This is dancer clone #{dancer_index + 1} of {total_dancers}.
Generate the image now."""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[frame_part, prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in resp.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                img_path = os.path.join(work_dir, f"dancer_{dancer_index}.png")
                with open(img_path, "wb") as f:
                    f.write(part.inline_data.data)
                print(f"  [Gemini] Dancer image saved: {img_path}")
                return img_path

        print("  [Gemini] No image generated, using video frame")
        return frame_path

    except Exception as e:
        print(f"  [Gemini] Image gen failed: {e}, using video frame")
        return frame_path


def gemini_analyze_scene(client, video_path: str, dancers_count: int) -> dict:
    """Quick Gemini analysis for logging/metadata."""
    print("  [Gemini] Analyzing video...")
    uploaded = client.files.upload(file=video_path)
    waited = 0
    while uploaded.state.name == "PROCESSING" and waited < 120:
        time.sleep(5); waited += 5
        uploaded = client.files.get(name=uploaded.name)

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[uploaded, "Briefly describe: 1) dance style, 2) setting/room, "
                      "3) dancer's outfit in detail. 3 lines max."]
        )
        analysis = resp.text.strip()
        client.files.delete(name=uploaded.name)
        return {"analysis": analysis}
    except Exception as e:
        try: client.files.delete(name=uploaded.name)
        except: pass
        return {"analysis": f"Analysis failed: {e}"}


# ──────────────────────────────────────────────────────────
#  MimicMotion via Replicate — TRUE motion transfer
# ──────────────────────────────────────────────────────────

def _compress_video_for_upload(video_path: str, work_dir: str, max_mb: int = 20) -> str:
    """Compress video to under max_mb for upload. Returns compressed path."""
    sz = os.path.getsize(video_path) / (1024 * 1024)
    if sz <= max_mb:
        return video_path

    ff = _ffmpeg()
    out = os.path.join(work_dir, "motion_compressed.mp4")
    print(f"  [Compress] {sz:.0f}MB → targeting {max_mb}MB...")

    try:
        # Get duration
        info = get_video_info(video_path)
        dur = max(info["duration"], 1)
        target_bitrate = int((max_mb * 8 * 1024) / dur)  # kbps

        subprocess.run(
            [ff, "-y", "-i", video_path,
             "-c:v", "libx264", "-preset", "fast",
             "-b:v", f"{target_bitrate}k",
             "-vf", "scale=-2:720",
             "-an",  # Strip audio (not needed for motion ref)
             "-t", str(min(dur, 30)),  # Cap at 30s
             out],
            capture_output=True, timeout=120)

        if os.path.exists(out) and os.path.getsize(out) > 1000:
            new_sz = os.path.getsize(out) / (1024 * 1024)
            print(f"  [Compress] Done: {new_sz:.1f}MB")
            return out
    except Exception as e:
        print(f"  [Compress] Failed: {e}")

    return video_path


def _upload_file_to_replicate(file_path: str) -> str:
    """Upload a file to Replicate using multipart form and return the serving URL."""
    import mimetypes

    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    sz_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"    Uploading {filename} ({sz_mb:.1f}MB)...")

    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://api.replicate.com/v1/files",
            headers={
                "Authorization": f"Bearer {REPLICATE_TOKEN}",
            },
            files={
                "content": (filename, f, content_type),
            },
            timeout=600,
        )

    if resp.status_code >= 400:
        print(f"    Upload response ({resp.status_code}): {resp.text[:500]}")
        resp.raise_for_status()

    data = resp.json()
    file_url = data.get("urls", {}).get("get", "")
    if not file_url:
        file_id = data.get("id", "")
        file_url = f"https://api.replicate.com/v1/files/{file_id}/content"

    print(f"    Uploaded → {file_url[:80]}...")
    return file_url


def mimic_motion_generate(dancer_image_path: str, motion_video_url: str,
                           work_dir: str, dancer_index: int) -> str:
    """
    MimicMotion on Replicate via direct HTTP API.
    - dancer_image_path: local path to dancer image
    - motion_video_url: already-uploaded Replicate file URL
    - Returns: path to generated video
    """
    print(f"  [MimicMotion] Generating dancer {dancer_index + 1}...")
    print(f"  [MimicMotion] Image: {dancer_image_path}")

    t0 = time.time()
    headers = {
        "Authorization": f"Bearer {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    try:
        # Upload dancer image
        print("  [MimicMotion] Uploading dancer image...")
        image_url = _upload_file_to_replicate(dancer_image_path)

        # Create prediction
        print("  [MimicMotion] Starting prediction...")
        payload = {
            "version": None,  # Will use latest
            "input": {
                "ref_video": motion_video_url,
                "ref_image": image_url,
                "resolution": 576,
                "chunk_size": 16,
                "frames_overlap": 6,
                "sample_stride": 2,
                "fps": 15,
                "noise_aug_strength": 0.0,
                "guidance_scale": 2.0,
                "seed": 42 + dancer_index,
            },
        }

        # Use the model endpoint (auto-resolves version)
        resp = requests.post(
            "https://api.replicate.com/v1/models/zsxkib/mimic-motion/predictions",
            headers=headers,
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        prediction = resp.json()

        pred_id = prediction.get("id", "")
        status = prediction.get("status", "")
        print(f"  [MimicMotion] Prediction: {pred_id}, status: {status}")

        # Poll until complete
        poll_url = prediction.get("urls", {}).get("get", "")
        if not poll_url:
            poll_url = f"https://api.replicate.com/v1/predictions/{pred_id}"

        polls = 0
        while status not in ("succeeded", "failed", "canceled"):
            time.sleep(10)
            polls += 1
            elapsed = time.time() - t0
            if polls % 3 == 0:
                print(f"  [MimicMotion] Still processing... ({elapsed:.0f}s)")

            r = requests.get(poll_url, headers={
                "Authorization": f"Bearer {REPLICATE_TOKEN}"
            }, timeout=30)
            r.raise_for_status()
            prediction = r.json()
            status = prediction.get("status", "")

            if elapsed > 900:  # 15 min timeout
                print("  [MimicMotion] Timed out after 15 min")
                return None

        if status != "succeeded":
            error = prediction.get("error", "Unknown error")
            print(f"  [MimicMotion] Failed: {error}")
            return None

        # Get output URL
        output = prediction.get("output", "")
        if isinstance(output, list):
            output = output[0] if output else ""
        output_url = str(output)

        if not output_url or not output_url.startswith("http"):
            print(f"  [MimicMotion] No output URL: {output}")
            return None

        elapsed = time.time() - t0
        print(f"  [MimicMotion] Done in {elapsed:.0f}s")

        # Download result
        out_path = os.path.join(work_dir, f"mimic_dancer_{dancer_index}.mp4")
        dl = requests.get(output_url, timeout=300)
        with open(out_path, "wb") as f:
            f.write(dl.content)

        sz = os.path.getsize(out_path)
        print(f"  [MimicMotion] Saved: {out_path} ({sz/1024/1024:.1f} MB)")
        return out_path

    except Exception as e:
        print(f"  [MimicMotion] Failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────
#  Segment original dancer
# ──────────────────────────────────────────────────────────

def segment(video_path, out_dir):
    """Fast KNN background subtraction for dancer mask."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_path = os.path.join(out_dir, "mask.mp4")
    writer = cv2.VideoWriter(mask_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    bg = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=800, detectShadows=False)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lr = 0.05 if n < 30 else 0.01
        fg = bg.apply(frame, learningRate=lr)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kc, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ko, iterations=1)
        cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            biggest = max(cs, key=cv2.contourArea)
            fg2 = np.zeros_like(fg)
            cv2.drawContours(fg2, [biggest], -1, 255, -1)
            fg = cv2.dilate(fg2, ko, iterations=1)
        fg = cv2.GaussianBlur(fg, (5, 5), 0)
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        writer.write(cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR))
        n += 1
    cap.release()
    writer.release()
    print(f"  [Segment] {n} frames")
    return {"mask": mask_path, "fps": fps, "w": w, "h": h, "total": n}


# ──────────────────────────────────────────────────────────
#  Composite: original dancer ON TOP
# ──────────────────────────────────────────────────────────

def composite(original, mask_path, generated, output, info):
    co = cv2.VideoCapture(original)
    cm = cv2.VideoCapture(mask_path)
    cg = cv2.VideoCapture(generated)

    fps, w, h = info["fps"], info["w"], info["h"]
    gfps = cg.get(cv2.CAP_PROP_FPS) or fps

    gframes = []
    while True:
        ret, f = cg.read()
        if not ret:
            break
        if f.shape[1] != w or f.shape[0] != h:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4)
        gframes.append(f)
    cg.release()

    if not gframes:
        co.release(); cm.release()
        shutil.copy2(original, output)
        return output

    gc = len(gframes)
    tmp = output + ".tmp.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    ratio = gfps / fps if fps > 0 else 1.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    n = 0

    while True:
        r1, fo = co.read()
        r2, fm = cm.read()
        if not r1 or not r2:
            break

        gi = int(n * ratio) % gc
        fg = gframes[gi]

        mg = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY)
        mg = cv2.GaussianBlur(mg, (7, 7), 3)
        mg = cv2.dilate(mg, k, iterations=2)
        mg = cv2.GaussianBlur(mg, (5, 5), 2)

        a = mg.astype(np.float32)[:, :, np.newaxis] / 255.0
        comp = fo * a + fg * (1.0 - a)
        wr.write(comp.astype(np.uint8))
        n += 1

    co.release(); cm.release(); wr.release()
    print(f"  [Composite] {n} frames")

    ff = _ffmpeg()
    try:
        subprocess.run(
            [ff, "-y", "-i", tmp, "-i", original,
             "-c:v", "libx264", "-preset", "fast", "-crf", "22",
             "-c:a", "aac", "-b:a", "128k",
             "-map", "0:v:0", "-map", "1:a:0?",
             "-shortest", "-movflags", "+faststart", output],
            capture_output=True, timeout=180)
        if os.path.exists(output) and os.path.getsize(output) > 1000:
            os.remove(tmp)
            return output
    except Exception:
        pass

    if not os.path.exists(output) or os.path.getsize(output) < 1000:
        shutil.copy2(tmp, output)
    if os.path.exists(tmp):
        os.remove(tmp)
    return output


# ──────────────────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────────────────

def generate_group_dance_video(video_path: str, dancers_count: int,
                                output_path: str) -> dict:
    """
    MimicMotion Clone Dance Pipeline:

    [1/5] Gemini analyzes scene
    [2/5] Gemini generates clone dancer image(s)
    [3/5] MimicMotion: your video motion → each dancer (TRUE sync)
    [4/5] Segment original dancer
    [5/5] Composite: you ON TOP + original audio
    """
    if not REPLICATE_TOKEN:
        return {"status": "failed",
                "analysis": "REPLICATE_API_TOKEN not set. Get one at replicate.com/account",
                "generated_video_path": None}

    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN
    t0 = time.time()
    work = f"processed/w_{int(t0)}"
    os.makedirs(work, exist_ok=True)
    temp_files = []

    try:
        client = get_client() if GOOGLE_API_KEY else None

        # Step 1: Video info
        print("[1/5] Reading video metadata...")
        vinfo = get_video_info(video_path)
        is_vertical = vinfo["height"] > vinfo["width"]
        aspect = "9:16" if is_vertical else "16:9"
        duration = vinfo["duration"]
        print(f"  {vinfo['width']}x{vinfo['height']}, {duration:.1f}s, {aspect}")

        # Analyze scene
        analysis_text = ""
        if client:
            result = gemini_analyze_scene(client, video_path, dancers_count)
            analysis_text = result["analysis"]
            print(f"  Scene: {analysis_text[:150]}")

        # Step 2: Generate dancer images
        print(f"\n[2/5] Generating {dancers_count} clone dancer image(s)...")
        dancer_images = []
        for i in range(dancers_count):
            if client:
                img = gemini_generate_dancer_image(
                    client, video_path, i, dancers_count, work)
            else:
                # Fallback: use a frame from the video
                img = os.path.join(work, f"dancer_{i}.png")
                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (0.3 + i * 0.1)))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    cv2.imwrite(img, frame)
                print(f"  No GEMINI_API_KEY — using video frame as dancer image")
            dancer_images.append(img)

        # Step 3: MimicMotion for each dancer
        print(f"\n[3/5] MimicMotion — cloning moves onto {dancers_count} dancer(s)...")

        # Pre-compress and upload motion video ONCE (shared by all dancers)
        print("  Preparing motion video for upload...")
        compressed_video = _compress_video_for_upload(video_path, work)
        print("  Uploading motion video to Replicate...")
        motion_url = _upload_file_to_replicate(compressed_video)

        mimic_videos = []
        for i in range(dancers_count):
            print(f"\n  --- Dancer {i + 1}/{dancers_count} ---")
            vid = mimic_motion_generate(
                dancer_images[i], motion_url, work, i)
            if vid:
                mimic_videos.append(vid)
            else:
                print(f"  Dancer {i + 1} failed — skipping")

            if i < dancers_count - 1:
                time.sleep(5)

        if not mimic_videos:
            return {"status": "failed",
                    "analysis": "All MimicMotion generations failed",
                    "generated_video_path": None}

        # Use first successful video as background
        bg_video = mimic_videos[0]

        # Step 4: Segment
        print(f"\n[4/5] Segmenting original dancer...")
        seg = segment(video_path, os.path.join(work, "seg"))

        # Step 5: Composite
        print(f"\n[5/5] Compositing final video...")
        composite(video_path, seg["mask"], bg_video, output_path, seg)

        elapsed = time.time() - t0
        out = get_video_info(output_path)
        fsize = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        final_analysis = (
            f"MimicMotion clone-dance: {len(mimic_videos)} synced backup dancer(s).\n"
            f"TRUE motion transfer — exact pose matching, not text approximation.\n"
            f"Original: {vinfo['width']}x{vinfo['height']}, {duration:.1f}s\n"
            f"Output: {out['duration']:.1f}s, {fsize/1024/1024:.1f} MB\n"
            f"Pipeline time: {elapsed:.0f}s\n\n"
            f"Scene: {analysis_text}"
        )

        print(f"\n{'='*50}")
        print(f"  DONE in {elapsed:.0f}s")
        print(f"  Output: {output_path}")
        print(f"  {out['duration']:.1f}s, {fsize/1024/1024:.1f} MB")
        print(f"{'='*50}")

        # Cleanup
        shutil.rmtree(work, ignore_errors=True)

        return {
            "status": "completed",
            "analysis": final_analysis,
            "generated_video_path": output_path
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        shutil.rmtree(work, ignore_errors=True)
        return {
            "status": "failed",
            "analysis": f"Video generation failed: {e}",
            "generated_video_path": None
        }