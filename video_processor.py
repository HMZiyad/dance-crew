"""
Video Processor v7 — Veo 3.1 Frame-by-Frame Chaining (Best Motion Sync)
=========================================================================

Approach: Extract keyframes every 8s from your video → each consecutive
pair becomes first_frame + last_frame for Veo 3.1 → Veo interpolates
the motion between YOUR actual poses → chain all clips together.

Result: dancer hits YOUR exact pose every 8 seconds.

Pipeline:
1. Gemini watches the full video → writes a Veo-optimized prompt
2. Extract keyframes every 8s from video → enhance for Veo input
3. Each keyframe pair → first_frame + last_frame → Veo 3.1 generates clip
4. For longer videos: chain multiple 8s clips → concatenate

No compositing, no keying, no overlays. Pure Veo generation with pose anchoring.
"""

import os
import time
import math
import shutil
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
VEO_CLIP_DURATION = 8
MAX_CLIPS = 4  # Max 4 clips = 32 seconds


def get_client():
    if not GOOGLE_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file.")
    return genai.Client(api_key=GOOGLE_API_KEY)


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
    """Extract a single frame at given position (0.0-1.0)."""
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


def extract_frame_at_time(video_path: str, time_sec: float, out_path: str) -> str:
    """Extract a single frame at a specific time in seconds."""
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
    """Ensure 720p+ and light sharpen for best Veo input."""
    frame = cv2.imread(frame_path)
    if frame is None:
        return frame_path
    h, w = frame.shape[:2]
    # Upscale small videos
    if min(h, w) < 720:
        scale = 720 / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    # Subtle sharpen
    kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    out = frame_path.replace(".png", "_enh.png")
    cv2.imwrite(out, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return out


# ──────────────────────────────────────────────────────────
#  Gemini → Writes the Veo Prompt (the critical step)
# ──────────────────────────────────────────────────────────

def gemini_write_veo_prompt(client, video_path: str,
                             dancers_count: int,
                             aspect_ratio: str) -> dict:
    """
    Gemini watches the video and writes an optimized Veo prompt.
    This is the most important step — the prompt quality determines everything.
    """
    print("  Uploading video to Gemini...")
    uploaded = client.files.upload(file=video_path)
    waited = 0
    while uploaded.state.name == "PROCESSING" and waited < 120:
        time.sleep(5)
        waited += 5
        uploaded = client.files.get(name=uploaded.name)

    instruction = f"""You are a Google Veo 3.1 prompt engineer. Watch this dance video carefully.

I will use your prompt to generate a NEW video using Veo 3.1 (first-frame + last-frame mode).
Veo will receive the starting frame and ending frame from the REAL video, then interpolate
motion between them. The video should show:
- The SAME person (or someone who looks very similar) doing the SAME dance
- With {dancers_count} ADDITIONAL backup dancers behind/beside them
- In the EXACT same location/room

YOUR TASK: Write 3 things.

=== 1. VEO_PROMPT ===
Write a 2-3 sentence Veo prompt. Veo works best with this structure:
[Camera/shot description]. [Subject + action]. [Setting + lighting].

RULES FOR GOOD VEO PROMPTS:
- Describe what the camera SEES (not what you want)
- Be SPECIFIC about the environment: "gymnasium with light maple hardwood floor, white drop ceiling with fluorescent panel lights, dark windows on the back wall" NOT "a room"  
- Be SPECIFIC about clothing: "woman in black t-shirt, leopard print Adidas track pants with white stripes, white sneakers, red trucker cap, curly blonde hair" NOT "casual clothes"
- For {dancers_count} backup dancers, describe their EXACT positions: "two dancers flanking on the left and right, two more dancers behind in a second row"
- Say "full body visible from head to toe" — Veo often crops people
- Say "all dancers performing synchronized [style] choreography" 
- Mention the {aspect_ratio} framing implicitly (e.g. "vertical phone video" for 9:16)
- Use "smartphone dance video" or "TikTok style dance video" — NOT "cinematic RED camera"
- Keep it PRACTICAL and CLEAR, not flowery

EXAMPLE GOOD PROMPT:
"Vertical smartphone video, wide full-body shot. A woman in a black crop top and leopard print Adidas track pants with a red cap leads a group of 4 dancers in synchronized hip-hop choreography, all performing energetic moves in a gymnasium with light maple hardwood floor and fluorescent ceiling lights. All 5 dancers fully visible head to toe, natural indoor lighting, casual TikTok dance video aesthetic."

=== 2. NEGATIVE_PROMPT ===
Comma-separated things to avoid. Always include:
cartoon, anime, 3D render, CGI, distorted face, extra limbs, blurry, watermark, text, miniature people, doll-like, cropped body, partial figure

=== 3. ANALYSIS ===
Brief 2-3 line summary of what you observed in the video (dance style, setting, clothing).

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
VEO_PROMPT: [prompt]
NEGATIVE_PROMPT: [negative]
ANALYSIS: [analysis]"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash", contents=[uploaded, instruction]
        )
        raw = resp.text
        print(f"  Gemini response: {len(raw)} chars")

        result = {"veo_prompt": "", "negative_prompt": "", "analysis": "", "raw": raw}
        cur = None
        for line in raw.split("\n"):
            s = line.strip()
            u = s.upper()
            if u.startswith("VEO_PROMPT"):
                cur = "veo_prompt"
                result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif u.startswith("NEGATIVE_PROMPT"):
                cur = "negative_prompt"
                result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif u.startswith("ANALYSIS"):
                cur = "analysis"
                result[cur] = s.split(":", 1)[1].strip() if ":" in s else ""
            elif cur and s:
                result[cur] += " " + s

        for k in result:
            if isinstance(result[k], str):
                result[k] = result[k].strip()

        # Fallback
        if not result["veo_prompt"]:
            result["veo_prompt"] = (
                f"Smartphone dance video, wide full-body shot. A dancer leads "
                f"a group of {dancers_count + 1} performers in synchronized "
                f"choreography in an indoor space. All dancers fully visible "
                f"from head to toe, natural lighting, casual dance video aesthetic."
            )
        if not result["negative_prompt"]:
            result["negative_prompt"] = (
                "cartoon, anime, 3D render, CGI, distorted face, extra limbs, "
                "blurry, watermark, text, miniature, doll-like, cropped body"
            )

        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

        return result

    except Exception as e:
        print(f"  Gemini failed: {e}")
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
        return {
            "veo_prompt": (
                f"Smartphone dance video, wide full-body shot. "
                f"A group of {dancers_count + 1} dancers performing synchronized "
                f"energetic dance choreography in an indoor room. "
                f"All dancers fully visible head to toe, "
                f"natural lighting, casual TikTok dance video feel."
            ),
            "negative_prompt": (
                "cartoon, anime, 3D render, distorted, blurry, "
                "watermark, text, miniature, doll-like, cropped body"
            ),
            "analysis": "Gemini analysis failed — using generic prompt.",
            "raw": ""
        }


# ──────────────────────────────────────────────────────────
#  Veo 3.1 Video Generation (first_frame + last_frame mode)
# ──────────────────────────────────────────────────────────

def generate_veo_clip(client, prompt: str, negative: str,
                       first_frame_image, last_frame_image,
                       aspect_ratio: str,
                       clip_index: int, total_clips: int) -> str:
    """Generate a single 8s Veo 3.1 clip using first + last frame anchoring."""
    os.makedirs("processed", exist_ok=True)
    max_retries = 3
    operation = None
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"  Clip {clip_index+1}/{total_clips} — attempt {attempt+1}...")
            print(f"  Prompt: {prompt[:200]}...")

            config = types.GenerateVideosConfig(
                person_generation="allow_adult",
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                negative_prompt=negative,
                last_frame=last_frame_image,
            )

            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=first_frame_image,
                config=config,
            )

            print(f"  Operation: {operation.name}")
            break

        except Exception as e:
            last_error = str(e)
            print(f"  Attempt {attempt+1} failed: {last_error[:200]}")

            # Rate limit
            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue

            # Negative prompt issue
            if "negative_prompt" in last_error.lower():
                print("  Retrying without negative prompt...")
                try:
                    config_no_neg = types.GenerateVideosConfig(
                        person_generation="allow_adult",
                        aspect_ratio=aspect_ratio,
                        number_of_videos=1,
                        last_frame=last_frame_image,
                    )
                    operation = client.models.generate_videos(
                        model="veo-3.1-generate-preview",
                        prompt=prompt,
                        image=first_frame_image,
                        config=config_no_neg,
                    )
                    print(f"  Started (no neg): {operation.name}")
                    break
                except Exception as e2:
                    last_error = str(e2)
                    print(f"  Still failed: {last_error[:150]}")

            # last_frame issue → try with first_frame only
            if "last_frame" in last_error.lower() or "frame" in last_error.lower():
                print("  Retrying without last_frame (first frame only)...")
                try:
                    config_no_last = types.GenerateVideosConfig(
                        person_generation="allow_adult",
                        aspect_ratio=aspect_ratio,
                        number_of_videos=1,
                        negative_prompt=negative,
                    )
                    operation = client.models.generate_videos(
                        model="veo-3.1-generate-preview",
                        prompt=prompt,
                        image=first_frame_image,
                        config=config_no_last,
                    )
                    print(f"  Started (no last frame): {operation.name}")
                    break
                except Exception as e3:
                    last_error = str(e3)
                    print(f"  Still failed: {last_error[:150]}")

            # Image-to-video fail → text-only
            print("  Trying text-only fallback...")
            try:
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
                print(f"  Text-only started: {operation.name}")
                break
            except Exception as e4:
                last_error = str(e4)
                print(f"  Text-only failed: {last_error[:150]}")
            break

    if operation is None:
        raise Exception(f"Clip {clip_index+1} failed: {last_error}")

    # Poll until done
    print(f"  Waiting for Veo 3.1...")
    polls = 0
    while not operation.done:
        polls += 1
        time.sleep(10)
        operation = client.operations.get(operation)
        elapsed = polls * 10
        if polls % 3 == 0:
            print(f"  Still generating... ({elapsed}s)")
        if polls > 60:
            raise Exception("Veo timed out after 10 minutes")

    print(f"  ✓ Clip {clip_index+1} generated!")

    vid = operation.response.generated_videos[0]
    clip_path = f"processed/clip_{clip_index}_{int(time.time())}.mp4"
    client.files.download(file=vid.video)
    vid.video.save(clip_path)

    sz = os.path.getsize(clip_path)
    print(f"  Saved: {clip_path} ({sz/1024/1024:.1f} MB)")
    return clip_path


# ──────────────────────────────────────────────────────────
#  Video Concatenation
# ──────────────────────────────────────────────────────────

def concatenate_videos(clip_paths: list, output_path: str) -> str:
    if not clip_paths:
        raise ValueError("No clips")
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path)
        return output_path

    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        print("  Concatenating with moviepy...")
        clips = [VideoFileClip(p) for p in clip_paths]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, codec="libx264",
                              audio_codec="aac", logger=None)
        for c in clips:
            c.close()
        final.close()
        return output_path
    except Exception as e:
        print(f"  Moviepy failed: {e}")

    # ffmpeg fallback
    try:
        import subprocess
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = "ffmpeg"

        concat_file = output_path + ".txt"
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{os.path.abspath(cp).replace(chr(92), '/')}'\n")
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", concat_file, "-c", "copy", output_path],
            capture_output=True, text=True, timeout=120
        )
        if os.path.exists(concat_file):
            os.remove(concat_file)
        if result.returncode == 0:
            return output_path
    except Exception as e:
        print(f"  ffmpeg failed: {e}")

    shutil.copy2(clip_paths[0], output_path)
    return output_path


# ──────────────────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────────────────

def generate_group_dance_video(video_path: str, dancers_count: int,
                                output_path: str) -> dict:
    """
    Generate a group dance video using Veo 3.1 frame-by-frame chaining.

    1. Gemini watches video → writes optimal Veo prompt
    2. Extract keyframes every 8s from your video
    3. Each keyframe pair → first_frame + last_frame → Veo 3.1 interpolates
    4. Chain clips + concatenate for longer videos

    Returns: {status, analysis, generated_video_path}
    """
    if not GOOGLE_API_KEY:
        return {"status": "failed",
                "analysis": "GEMINI_API_KEY not set.",
                "generated_video_path": None}

    clip_paths = []
    temp_files = []

    try:
        client = get_client()

        # Step 1: Video info
        print("[1/4] Reading video metadata...")
        vinfo = get_video_info(video_path)
        is_vertical = vinfo["height"] > vinfo["width"]
        aspect = "9:16" if is_vertical else "16:9"
        duration = vinfo["duration"]
        print(f"  {vinfo['width']}x{vinfo['height']}, {duration:.1f}s, {aspect}")

        # Step 2: Gemini writes the Veo prompt
        print("[2/4] Gemini analyzing video & writing Veo prompt...")
        gresult = gemini_write_veo_prompt(client, video_path, dancers_count, aspect)

        veo_prompt = gresult["veo_prompt"]
        negative = gresult["negative_prompt"]
        analysis = gresult["analysis"]

        print(f"\n  ╔══ VEO PROMPT ══════════════════════════")
        print(f"  ║ {veo_prompt}")
        print(f"  ╠══ NEGATIVE ═══════════════════════════")
        print(f"  ║ {negative[:120]}")
        print(f"  ╠══ ANALYSIS ═══════════════════════════")
        print(f"  ║ {analysis[:120]}")
        print(f"  ╚═════════════════════════════════════════\n")

        # Step 3: Extract keyframes every 8s
        print("[3/4] Extracting keyframes every 8s...")
        num_clips = min(MAX_CLIPS, max(1, math.ceil(duration / VEO_CLIP_DURATION)))
        print(f"  Need {num_clips} clip(s) for ~{num_clips * VEO_CLIP_DURATION}s")

        # Keyframe times: 0s, 8s, 16s, 24s, ... + end
        # We need num_clips + 1 keyframes (pairs for each clip)
        keyframe_times = []
        for i in range(num_clips + 1):
            t = min(i * VEO_CLIP_DURATION, duration - 0.1)
            keyframe_times.append(max(0.1, t))

        key_frames = []
        for i, t in enumerate(keyframe_times):
            out_path = f"processed/kf_{i}_{int(time.time())}.png"
            os.makedirs("processed", exist_ok=True)
            extract_frame_at_time(video_path, t, out_path)
            temp_files.append(out_path)
            enh = enhance_frame(out_path)
            if enh != out_path:
                temp_files.append(enh)
            key_frames.append(enh)
            print(f"  Keyframe {i}: t={t:.1f}s")

        # Step 4: Generate Veo clips between consecutive keyframe pairs
        print(f"[4/4] Generating {num_clips} clip(s) — first/last frame chaining...")

        for i in range(num_clips):
            # Load first frame (keyframe i)
            with open(key_frames[i], "rb") as f:
                first_bytes = f.read()
            first_image = types.Image(image_bytes=first_bytes, mime_type="image/png")

            # Load last frame (keyframe i+1)
            last_idx = min(i + 1, len(key_frames) - 1)
            with open(key_frames[last_idx], "rb") as f:
                last_bytes = f.read()
            last_image = types.Image(image_bytes=last_bytes, mime_type="image/png")

            clip = generate_veo_clip(
                client, veo_prompt, negative,
                first_image, last_image,
                aspect, i, num_clips
            )
            clip_paths.append(clip)

            if i < num_clips - 1:
                print(f"  Waiting 10s before next clip...")
                time.sleep(10)

        # Concatenate
        if len(clip_paths) > 1:
            print(f"  Concatenating {len(clip_paths)} clips...")
        concatenate_videos(clip_paths, output_path)

        # Stats
        out = get_video_info(output_path)
        fsize = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        final_analysis = (
            f"Frame-chaining: poses anchored every {VEO_CLIP_DURATION}s from your video.\n"
            f"Generated group dance with {dancers_count} backup dancer(s).\n"
            f"Original: {vinfo['width']}x{vinfo['height']}, {duration:.1f}s\n"
            f"Output: {len(clip_paths)} clip(s), {out['duration']:.1f}s, "
            f"{fsize/1024/1024:.1f} MB\n"
            f"Aspect: {aspect}\n\n"
            f"Gemini analysis:\n{analysis}\n\n"
            f"Veo prompt:\n{veo_prompt}"
        )

        print(f"\n  ✓ DONE: {output_path} ({out['duration']:.1f}s, {fsize/1024/1024:.1f} MB)")

        # Cleanup temp files
        for fp in temp_files:
            if os.path.exists(fp):
                os.remove(fp)
        for cp in clip_paths:
            if os.path.exists(cp) and os.path.abspath(cp) != os.path.abspath(output_path):
                os.remove(cp)

        return {
            "status": "completed",
            "analysis": final_analysis,
            "generated_video_path": output_path
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        for fp in temp_files:
            if os.path.exists(fp):
                os.remove(fp)
        for cp in clip_paths:
            if os.path.exists(cp):
                os.remove(cp)
        return {
            "status": "failed",
            "analysis": f"Video generation failed: {e}",
            "generated_video_path": None
        }