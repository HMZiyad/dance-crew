import os
import time
import uuid
import cv2
import requests
import subprocess
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

def extract_first_frame(video_path: str, out_img: str):
    """ Extract the first frame of a video using cv2. """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(out_img, frame)
    cap.release()

def extract_last_frame(video_path: str, out_img: str):
    """ Extract the last frame of a video using cv2. """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(out_img, frame)
    cap.release()

def image_bytes_from_path(path: str):
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(path, "rb") as f:
        return types.Image(image_bytes=f.read(), mime_type=mime)

def concatenate_videos(clip_paths: list, output_path: str):
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
    """ Analyze photo with Gemini to build a 2D anime dance video prompt. """
    print("Uploading image to Gemini for outfit analysis...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

    choreo_moves = choreography_library.get_choreo(dance_style)

    instruction = f"""You are an expert anime director and prompt engineer for a generative video model.
Look at the character in this photo and concisely describe their exact outfit, clothing colors, hair, and appearance.
Then, adapt those details into the following prompt template by replacing [Character Description] with your visual description:

"FLAT 2D ANIME STYLE: High-octane anime choreography, fluid sakuga style. [Character Description] is {choreo_moves[:220]}. Pure 2D flat Japanese anime character, sharp line art, vibrant cel-shading, 2D hand-drawn aesthetic. NO realism. Trail effects on limbs, masterpiece studio ghibli cel-shaded animation quality."

CRITICAL INSTRUCTION: Output ONLY the final completed prompt. Do not add any conversational text. Must be under 500 characters. Keep the specific move names.
"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, instruction]
        )
        prompt = resp.text.strip()
        print(f"Generated Anime Prompt: {prompt[:100]}...")
        print(prompt)
        return prompt
    except Exception as e:
        print(f"Gemini prompt generation failed: {e}")
        return f"FLAT 2D ANIME STYLE: A character performing a complex {dance_style} dance routine. Pure 2D flat Japanese anime character, sharp line art, vibrant cel-shading, masterpiece studio ghibli animation quality."


def generate_anime_video(image_path: str, dance_style: str, duration: int, output_path: str) -> dict:
    os.makedirs("processed", exist_ok=True)
    temp_files      = []   # intermediate files to clean up later
    video_chunks    = []
    anime_sketch_path = None

    # Veo natively generates ~8 second clips
    num_chunks = max(1, round(duration / 8.0))

    try:
        client = get_client()

        # ── STEP 1: Gemini analyzes uploaded photo → builds anime prompt ──────────
        print("[Step 1] Generating anime prompt via Gemini...")
        anime_prompt = gemini_write_anime_prompt(client, image_path, dance_style)

        # ── STEP 2: Imagen 3 generates a 2D anime sketch from the original photo ──
        print("[Step 2] Generating 2D anime sketch with Imagen 3...")
        try:
            # Use a generic anime character description to avoid safety filter blocks on real faces
            imagen_prompt = (
                f"A full-body 2D flat Japanese anime character in studio ghibli style, "
                f"vibrant cel-shaded illustration, sharp clear lineart, standing pose ready to dance, "
                f"NO photorealism, NO real humans. Dance style: {dance_style}. "
                f"Outfit context: {anime_prompt[:300]}"
            )
            img_result = client.models.generate_images(
                model='imagen-4.0-generate-001',
                prompt=imagen_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="9:16"
                )
            )
            if img_result.generated_images:
                anime_sketch_path = os.path.join("processed", f"anime_sketch_{uuid.uuid4()}.jpg")
                # Write raw bytes directly — .save() is not reliable on all SDK versions
                img_bytes = img_result.generated_images[0].image.image_bytes
                with open(anime_sketch_path, "wb") as f:
                    f.write(img_bytes)
                print(f"[Step 2] Anime sketch saved: {anime_sketch_path}")
            else:
                raise Exception("Imagen returned no images.")
        except Exception as e:
            print(f"[Step 2] Imagen sketch failed, falling back to original image: {e}")
            anime_sketch_path = image_path

        # ── STEP 3: Veo animates the sketch in chunks ─────────────────────────────
        # • Chunk 1   → seed = the Imagen 3 anime sketch
        # • Chunk N+1 → seed = last frame of chunk N
        current_seed_img = anime_sketch_path

        for i in range(num_chunks):
            print(f"[Step 3] Generating Veo chunk {i+1}/{num_chunks} "
                  f"(seed: {os.path.basename(current_seed_img)})...")

            veo_prompt = anime_prompt if i == 0 else f"SEAMLESS CONTINUATION: {anime_prompt}"

            try:
                operation = client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=veo_prompt,
                    image=image_bytes_from_path(current_seed_img),
                    config=types.GenerateVideosConfig(
                        person_generation="allow_adult",
                        aspect_ratio="9:16",
                        number_of_videos=1,
                        negative_prompt=(
                            "photorealistic, realism, real human, real person, 3d render, "
                            "live-action, photography, CGI, human skin, natural lighting, "
                            "realistic face, realistic body"
                        )
                    )
                )

                polls = 0
                while not operation.done:
                    time.sleep(10)
                    polls += 1
                    operation = client.operations.get(operation)
                    if polls % 3 == 0:
                        print(f"  Waiting... ({polls * 10}s elapsed)")
                    if polls > 72:
                        raise Exception("Veo timed out after 12 minutes.")

                if not operation.response or not operation.response.generated_videos:
                    err = operation.error if operation.error else "No video returned (possibly blocked by safety filters)."
                    raise Exception(f"Veo generation failed: {err}")

                vid = operation.response.generated_videos[0]
                vid_path = os.path.join("processed", f"veo_anime_{uuid.uuid4()}.mp4")
                client.files.download(file=vid.video)
                vid.video.save(vid_path)

            except Exception as e:
                raise RuntimeError(f"Veo chunk {i+1} failed: {str(e)}")

            video_chunks.append(vid_path)

            # Extract last frame → seed for next chunk
            if i < num_chunks - 1:
                last_frame_path = os.path.join("processed", f"lf_{i+1}_{uuid.uuid4()}.jpg")
                extract_last_frame(vid_path, last_frame_path)
                temp_files.append(last_frame_path)
                current_seed_img = last_frame_path

        # ── Concatenate all chunks into the final output ───────────────────────────
        print(f"Concatenating {num_chunks} chunk(s) into final output...")
        concatenate_videos(video_chunks, output_path)

        # The Imagen anime sketch is the preview image shown in the UI
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
        # Clean up only intermediate last-frame images and raw video chunks.
        # DO NOT delete anime_sketch_path (shown in UI) or the final output.
        safe_to_delete = temp_files + video_chunks
        for f in safe_to_delete:
            if f and os.path.exists(f) and f != output_path and f != anime_sketch_path:
                try: os.remove(f)
                except: pass
