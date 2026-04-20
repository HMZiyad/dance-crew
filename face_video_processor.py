"""
face_video_processor.py — Face-Consistent Image → Video Generator (Gemini/Veo only)
=====================================================================================

Pipeline:
  Step 1: Gemini analyzes the photo → describes the person's FULL appearance
          (face, clothing, body) to create a precise Veo prompt
  Step 2: Veo 3.1 generates the first clip, using the original photo as the
          seed image (image anchor helps preserve appearance/face)
  Step 3: For multi-chunk videos, last frame of each clip seeds the next chunk
          → maintains visual continuity across all chunks
  Step 4: FFmpeg concatenates all chunks → final MP4
"""

import os
import time
import uuid
import cv2
import subprocess
import shutil
from dotenv import load_dotenv

from google import genai
from google.genai import types
import choreography_library

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file.")
    return genai.Client(api_key=GEMINI_API_KEY)


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_last_frame(video_path: str, out_img: str):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 2))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(out_img, frame)
    cap.release()


def image_bytes_from_path(path: str) -> types.Image:
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(path, "rb") as f:
        return types.Image(image_bytes=f.read(), mime_type=mime)


def concatenate_videos(clip_paths: list, output_path: str):
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], output_path)
        return output_path
    ffmpeg = get_ffmpeg_exe()
    concat_txt = output_path + "_list.txt"
    try:
        with open(concat_txt, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{os.path.abspath(cp).replace(chr(92), '/')}'\n")
        args = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", concat_txt, "-c", "copy", "-movflags", "+faststart", output_path
        ]
        result = subprocess.run(args, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr[-300:]}")
    finally:
        if os.path.exists(concat_txt):
            os.remove(concat_txt)
    return output_path


def gemini_write_face_prompt(client, image_path: str, action: str) -> str:
    """
    Gemini looks at the photo and writes a Veo prompt describing the person's
    full appearance (face + clothing) AND injects specific choreographic moves
    for the requested action/dance style so Veo generates real choreography.
    """
    print("[Step 1] Gemini analyzing photo for detailed appearance + choreography prompt...")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)

    # Get genre-specific move descriptions
    choreo_moves = choreography_library.get_choreo(action)

    instruction = f"""You are a Veo 3.1 prompt engineer. Look at this photo and write a precise
video prompt so the generated video person looks EXACTLY like the person in this photo
AND performs SPECIFIC recognisable choreography.

Describe in this order:
1. Facial features: face shape, skin tone, hair color + style, facial hair if present, glasses if present
2. Body build: height impression, build
3. Exact outfit: every garment, color, pattern, footwear, accessories
4. Then insert EXACTLY these choreographic moves (do not paraphrase — use these specific terms):
   {choreo_moves}
5. Location/setting: a real venue appropriate for: {action}
6. End with: "photorealistic, cinematic lighting, full body visible head to toe, masterpiece quality, professional dance performance"

CRITICAL: The prompt MUST include the specific move names from step 4.
Write ONLY the final prompt. No labels, no explanation. Under 900 characters.
"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, instruction]
        )
        prompt = resp.text.strip()
        if len(prompt) > 950:
            prompt = prompt[:950]
        print(f"[Step 1] Prompt ready ({len(prompt)} chars): {prompt[:100]}...")
        return prompt
    except Exception as e:
        print(f"[Step 1] Gemini failed, using fallback prompt: {e}")
        return (
            f"A person {choreo_moves}. "
            "Photorealistic, cinematic lighting, smooth motion, full body visible, masterpiece quality."
        )


def generate_veo_clip(client, prompt: str, seed_image, clip_index: int, total: int) -> str:
    """Generate one Veo clip, using seed_image as the anchor frame."""
    os.makedirs("processed", exist_ok=True)
    operation = None
    last_error = None

    for attempt in range(2):
        try:
            print(f"  [Veo] Clip {clip_index+1}/{total} — attempt {attempt+1}...")
            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=seed_image,
                config=types.GenerateVideosConfig(
                    person_generation="allow_adult",
                    aspect_ratio="9:16",
                    number_of_videos=1,
                    negative_prompt=(
                        "cartoon, anime, illustration, CGI, 3d render, blurry, "
                        "distorted face, extra limbs, watermark, text"
                    ),
                ),
            )
            print(f"  [Veo] Operation: {operation.name}")
            break
        except Exception as e:
            last_error = str(e)
            print(f"  [Veo] Attempt {attempt+1} failed: {last_error[:200]}")
            if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                time.sleep(20 * (attempt + 1))
                continue
            operation = None; break

    if operation is None:
        raise Exception(f"Veo clip {clip_index+1} failed: {last_error}")

    print("  [Veo] Waiting for generation...")
    polls = 0
    while not operation.done:
        time.sleep(10)
        polls += 1
        operation = client.operations.get(operation)
        if polls % 3 == 0:
            print(f"  [Veo] Generating... ({polls * 10}s elapsed)")
        if polls > 72:
            raise Exception("Veo timed out after 12 minutes.")

    if not operation.response or not operation.response.generated_videos:
        err = operation.error if operation.error else "No video returned (possibly blocked by safety filters)."
        raise Exception(f"Veo generation failed: {err}")

    vid = operation.response.generated_videos[0]
    vid_path = os.path.join("processed", f"facevid_{clip_index}_{uuid.uuid4()}.mp4")
    client.files.download(file=vid.video)
    vid.video.save(vid_path)
    print(f"  [Veo] Clip {clip_index+1} saved: {vid_path}")
    return vid_path


def generate_face_video(
    image_path: str,
    action: str,
    duration: int,
    output_path: str,
) -> dict:
    """
    Full pipeline:
      1. Gemini writes a full-appearance Veo prompt (face + clothes + action)
      2. Veo generates clip 1, anchored to the original reference photo
      3. For multi-chunk: last frame of clip N → seed for clip N+1
      4. Concatenate all clips → final MP4
    """
    os.makedirs("processed", exist_ok=True)
    temp_files   = []
    video_chunks = []

    # Veo generates ~8 second clips natively
    num_chunks = max(1, round(duration / 8.0))

    try:
        client = get_client()

        # ── Step 1: Gemini writes the full-appearance prompt ──────────────────
        veo_prompt = gemini_write_face_prompt(client, image_path, action)

        # ── Steps 2+: Veo generates each chunk ────────────────────────────────
        # Chunk 1  → seed = original photo (best face anchor)
        # Chunk N  → seed = last frame of chunk N-1 (visual continuity)
        current_seed_path = image_path

        for i in range(num_chunks):
            print(f"\n[Step {i+2}] Generating Veo chunk {i+1}/{num_chunks}...")

            seed_image = image_bytes_from_path(current_seed_path)
            chunk_prompt = veo_prompt if i == 0 else f"SEAMLESS CONTINUATION, same person, same outfit: {veo_prompt}"

            vid_path = generate_veo_clip(client, chunk_prompt, seed_image, i, num_chunks)
            video_chunks.append(vid_path)

            # Extract last frame → seed for next chunk
            if i < num_chunks - 1:
                lf_path = os.path.join("processed", f"facevid_lf_{i}_{uuid.uuid4()}.jpg")
                extract_last_frame(vid_path, lf_path)
                temp_files.append(lf_path)
                current_seed_path = lf_path

        # ── Final: concatenate all chunks ─────────────────────────────────────
        print(f"\nConcatenating {num_chunks} chunk(s) into final output...")
        concatenate_videos(video_chunks, output_path)
        print(f"✓ Done → {output_path}")

        return {
            "status":           "completed",
            "output_file_path": output_path,
            "prompt_used":      veo_prompt,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {
            "status":           "failed",
            "output_file_path": None,
            "error_msg":        str(e),
            "prompt_used":      locals().get("veo_prompt", ""),
        }
    finally:
        safe_to_delete = temp_files + video_chunks
        for f in safe_to_delete:
            if f and os.path.exists(f) and f != output_path:
                try: os.remove(f)
                except: pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python face_video_processor.py <image_path> <action> [duration] [output.mp4]")
        sys.exit(1)
    img    = sys.argv[1]
    action = sys.argv[2]
    dur    = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out    = sys.argv[4] if len(sys.argv) > 4 else "face_video_output.mp4"
    result = generate_face_video(img, action, dur, out)
    print(result)
