import os
import time
import uuid
import cv2
import base64
import requests
import subprocess
import shutil
from dotenv import load_dotenv

from google import genai
from google.genai import types
from runwayml import RunwayML

load_dotenv()

def get_runway_client():
    api_key = os.environ.get("RUNWAY_API_KEY") or os.environ.get("RUNWAYML_API_SECRET")
    if not api_key:
        raise ValueError("RUNWAY_API_KEY or RUNWAYML_API_SECRET not set in .env file.")
    os.environ["RUNWAYML_API_SECRET"] = api_key
    return RunwayML()

def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)

def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def extract_last_frame(video_path: str, out_img: str):
    """ Extract the last frame of a video using cv2. """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(out_img, frame)
    cap.release()

def get_base64_data_uri(image_path: str) -> str:
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

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
            raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr[-200:]}")
    finally:
        if os.path.exists(concat_txt):
            os.remove(concat_txt)
    return output_path


def gemini_write_anime_prompt(client, image_path: str, dance_style: str) -> str:
    """ Analyze the photo with Gemini to build a 2D anime video prompt for RunwayML. """
    print("[Step 1] Uploading image to Gemini for outfit analysis...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

    instruction = f"""You are an expert anime director and prompt engineer for a generative video model.
Look at the character in this photo and concisely describe their exact outfit, clothing colors, hair, and appearance.
Then fill in [Character Description] with your description in this template:

"High-octane anime choreography, fluid sakuga style. [Character Description] performing a complex {dance_style} dance routine. Pure 2D flat Japanese anime, sharp line art, vibrant cel-shading, hand-drawn aesthetic, NO photorealism, trail effects on limbs, masterpiece studio ghibli quality."

CRITICAL INSTRUCTION: Output ONLY the final completed prompt. Do not add conversational text. Must be under 500 characters.
"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, instruction]
        )
        prompt = resp.text.strip()
        if len(prompt) > 900:
            prompt = prompt[:900]
        print(f"[Step 1] Anime Prompt: {prompt[:120]}...")
        return prompt
    except Exception as e:
        print(f"[Step 1] Gemini prompt generation failed: {e}")
        return (
            f"High-octane anime choreography, fluid sakuga style. A character performing a complex "
            f"{dance_style} dance routine. Pure 2D flat Japanese anime, sharp line art, vibrant "
            f"cel-shading, hand-drawn aesthetic, masterpiece studio ghibli quality."
        )


def imagen_generate_anime_sketch(g_client, anime_prompt: str, dance_style: str) -> str | None:
    """
    Uses Imagen 3 to generate a 2D anime sketch. Uses a generic anime prompt
    to avoid safety filter blocks that occur when describing real human faces.
    Returns the saved sketch file path, or None if it fails.
    """
    print("[Step 2] Generating 2D anime sketch with Imagen 3...")
    try:
        # Use a generic anime character prompt to avoid safety filter blocks on real faces.
        imagen_prompt = (
            f"A full-body 2D flat Japanese anime character in studio ghibli style, "
            f"vibrant cel-shaded illustration, sharp clear lineart, standing pose ready to dance, "
            f"NO photorealism, NO real humans. Dance style: {dance_style}. "
            f"Outfit context: {anime_prompt[:300]}"
        )
        img_result = g_client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=imagen_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="9:16"
            )
        )
        if img_result.generated_images:
            sketch_path = os.path.join("processed", f"anime_sketch_{uuid.uuid4()}.jpg")
            # Write raw bytes directly — .save() is not reliable on all SDK versions
            img_bytes = img_result.generated_images[0].image.image_bytes
            with open(sketch_path, "wb") as f:
                f.write(img_bytes)
            print(f"[Step 2] Anime sketch saved: {sketch_path}")
            return sketch_path
        else:
            print("[Step 2] Imagen returned no images.")
            return None
    except Exception as e:
        print(f"[Step 2] Imagen sketch failed: {e}")
        return None


def generate_anime_video(image_path: str, dance_style: str, duration: int, output_path: str) -> dict:
    os.makedirs("processed", exist_ok=True)
    temp_files      = []   # intermediate files to clean up
    video_chunks    = []
    anime_sketch_path = None

    # RunwayML Gen-3 Alpha supports 5 or 10-second clips
    chunk_size = 10 if duration > 5 else 5
    num_chunks = max(1, round(duration / float(chunk_size)))

    try:
        r_client = get_runway_client()
        g_client = get_gemini_client()

        # ── STEP 1: Gemini builds the anime prompt ────────────────────────────────
        anime_prompt = (
            f"High-octane anime choreography, fluid sakuga style. A character performing a "
            f"complex {dance_style} dance routine. Pure 2D flat Japanese anime, sharp line art, "
            f"vibrant cel-shading, hand-drawn aesthetic, masterpiece studio ghibli quality."
        )
        if g_client:
            anime_prompt = gemini_write_anime_prompt(g_client, image_path, dance_style)

        # ── STEP 2: Imagen 3 generates a 2D anime sketch ─────────────────────────
        if g_client:
            anime_sketch_path = imagen_generate_anime_sketch(g_client, anime_prompt, dance_style)

        if not anime_sketch_path:
            print("[Step 2] Falling back to original uploaded image as seed.")
            anime_sketch_path = image_path

        # ── STEP 3: RunwayML animates the sketch in chunks ────────────────────────
        # • Chunk 1   → seed = Imagen 3 anime sketch
        # • Chunk N+1 → seed = last frame of chunk N
        current_seed_img = anime_sketch_path

        for i in range(num_chunks):
            print(f"[Step 3] Generating RunwayML chunk {i+1}/{num_chunks} "
                  f"(seed: {os.path.basename(current_seed_img)})...")

            final_prompt = anime_prompt if i == 0 else f"SEAMLESS CONTINUATION: {anime_prompt}"
            if len(final_prompt) > 900:
                final_prompt = final_prompt[:900]

            try:
                task = r_client.image_to_video.create(
                    model='gen3a_turbo',
                    prompt_image=get_base64_data_uri(current_seed_img),
                    prompt_text=final_prompt,
                    duration=chunk_size,
                    ratio='768:1280'
                )
                print(f"[Step 3] Runway task submitted: {task.id}")

                polls = 0
                while True:
                    time.sleep(10)
                    polls += 1
                    task_status = r_client.tasks.retrieve(task.id)
                    if task_status.status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                        break
                    if polls % 3 == 0:
                        print(f"  Waiting... ({polls * 10}s elapsed)")

                if task_status.status != 'SUCCEEDED':
                    err = getattr(task_status, 'failure', None) or "Unknown failure"
                    raise Exception(f"Runway generation failed: {err}")

                print(f"[Step 3] Chunk {i+1} done!")
                video_url = task_status.output[0]
                vid_path = os.path.join("processed", f"runway_anime_{uuid.uuid4()}.mp4")
                resp = requests.get(video_url, timeout=120)
                resp.raise_for_status()
                with open(vid_path, "wb") as f:
                    f.write(resp.content)

            except Exception as e:
                raise RuntimeError(f"RunwayML chunk {i+1} failed: {str(e)}")

            video_chunks.append(vid_path)

            # Extract last frame → seed for next chunk
            if i < num_chunks - 1:
                last_frame_path = os.path.join("processed", f"lf_{i+1}_{uuid.uuid4()}.jpg")
                extract_last_frame(vid_path, last_frame_path)
                temp_files.append(last_frame_path)
                current_seed_img = last_frame_path

        # ── Concatenate all chunks ─────────────────────────────────────────────────
        print(f"Concatenating {num_chunks} chunk(s) into final output...")
        concatenate_videos(video_chunks, output_path)

        # The Imagen sketch is used as the preview in the UI
        return {
            "status":           "completed",
            "anime_image_path": anime_sketch_path,
            "output_file_path": output_path,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {
            "status":           "failed",
            "anime_image_path": anime_sketch_path,
            "output_file_path": None,
            "error_msg":        str(e),
        }
    finally:
        # Clean up intermediate last-frame images and raw runway clips
        # Do NOT delete the anime sketch (shown in UI) or the final output
        safe_to_delete = temp_files + video_chunks
        for f in safe_to_delete:
            if f and os.path.exists(f) and f != output_path and f != anime_sketch_path:
                try: os.remove(f)
                except: pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python anime_processor_runway.py <image_path> <dance_style>")
        sys.exit(1)
    img   = sys.argv[1]
    style = sys.argv[2]
    out   = "test_runway_anime_output.mp4"
    result = generate_anime_video(img, style, duration=10, output_path=out)
    print(result)
