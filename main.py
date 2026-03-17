from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from typing import List, Optional
from datetime import datetime

from database import engine, get_db, init_db, Video, ImageJob
from models import VideoCreate, VideoResponse, VideoUpdate, ImageJobResponse
import video_processor
import imgevideo

# Initialize DB
init_db()

app = FastAPI(title="Dance Crew Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR    = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="."), name="static")


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def build_video_response(video: Video, base_url: str) -> dict:
    original_path  = video.original_file_path.replace("\\", "/")  if video.original_file_path  else None
    processed_path = video.processed_file_path.replace("\\", "/") if video.processed_file_path else None
    return {
        "id":                  video.id,
        "title":               video.title,
        "description":         video.description,
        "dancers_count":       video.dancers_count,
        "filename":            video.filename,
        "processed_status":    video.processed_status,
        "uploaded_at":         video.uploaded_at,
        "original_file_path":  video.original_file_path,
        "processed_file_path": video.processed_file_path,
        "gemini_analysis":     video.gemini_analysis,
        "original_video_url":  f"{base_url}/static/{original_path}"  if original_path  else None,
        "generated_video_url": f"{base_url}/static/{processed_path}" if processed_path else None,
    }


def build_imagejob_response(job: ImageJob, base_url: str) -> dict:
    img_path = job.image_file_path.replace("\\", "/") if job.image_file_path else None
    out_path = job.output_file_path.replace("\\", "/") if job.output_file_path else None
    return {
        "id":                job.id,
        "title":             job.title,
        "dance_style":       job.dance_style,
        "dancers_count":     job.dancers_count,
        "status":            job.status,
        "created_at":        job.created_at,
        "image_file_path":   job.image_file_path,
        "output_file_path":  job.output_file_path,
        "analysis":          job.analysis,
        "image_url":         f"{base_url}/static/{img_path}" if img_path else None,
        "generated_video_url": f"{base_url}/static/{out_path}" if out_path else None,
    }


# ══════════════════════════════════════════════════════════
#  ── VIDEO ROUTES (existing) ──
# ══════════════════════════════════════════════════════════

@app.post("/videos/", response_model=VideoResponse)
async def upload_video(
    request: Request,
    title: str = Form(...),
    dancers_count: int = Form(0),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    file_ext        = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path       = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_video = Video(
        filename=unique_filename, title=title,
        dancers_count=dancers_count, original_file_path=file_path
    )
    db.add(db_video); db.commit(); db.refresh(db_video)

    if dancers_count > 0:
        background_tasks.add_task(process_video_task, db_video.id, dancers_count, None)

    base_url = str(request.base_url).rstrip("/")
    return build_video_response(db_video, base_url)


@app.get("/videos/", response_model=List[VideoResponse])
def read_videos(request: Request, skip: int = 0, limit: int = 100,
                db: Session = Depends(get_db)):
    videos   = db.query(Video).offset(skip).limit(limit).all()
    base_url = str(request.base_url).rstrip("/")
    return [build_video_response(v, base_url) for v in videos]


@app.get("/videos/{video_id}", response_model=VideoResponse)
def read_video(video_id: int, request: Request, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    base_url = str(request.base_url).rstrip("/")
    return build_video_response(video, base_url)


@app.delete("/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    for p in [video.original_file_path, video.processed_file_path]:
        if p and os.path.exists(p):
            os.remove(p)
    db.delete(video); db.commit()
    return {"ok": True}


def process_video_task(video_id: int, dancers_count: int, _db_session):
    db = next(get_db())
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video: return
        video.processed_status = "processing"
        video.dancers_count    = dancers_count
        db.commit()

        output_filename = f"{uuid.uuid4()}.mp4"
        output_path     = os.path.join(PROCESSED_DIR, output_filename)
        print(f"Processing video {video_id} with {dancers_count} dancers...")
        result = video_processor.generate_group_dance_video(
            video.original_file_path, dancers_count, output_path
        )
        video.gemini_analysis    = result.get("analysis", "")
        video.processed_status   = result.get("status", "failed")
        video.processed_file_path = result.get("generated_video_path")
        db.commit()
    except Exception as e:
        print(f"Error processing video: {e}")
        video.processed_status = "failed"
        video.gemini_analysis  = str(e)
        db.commit()
    finally:
        db.close()


@app.post("/videos/{video_id}/process")
async def process_video(video_id: int, dancers_count: int,
                        background_tasks: BackgroundTasks,
                        db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    background_tasks.add_task(process_video_task, video_id, dancers_count, None)
    return {"message": "Processing started", "video_id": video_id}


@app.get("/generated-videos/", response_model=List[VideoResponse])
def list_generated_videos(request: Request, skip: int = 0, limit: int = 100,
                          db: Session = Depends(get_db)):
    videos = db.query(Video).filter(
        Video.processed_file_path.isnot(None),
        Video.processed_status == "completed"
    ).offset(skip).limit(limit).all()
    base_url = str(request.base_url).rstrip("/")
    return [build_video_response(v, base_url) for v in videos]


@app.get("/generated-videos/{video_id}/download")
def download_generated_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.processed_file_path or not os.path.exists(video.processed_file_path):
        raise HTTPException(status_code=404, detail="Generated video file not found")
    return FileResponse(
        path=video.processed_file_path, media_type="video/mp4",
        filename=f"{video.title}_generated.mp4"
    )


@app.delete("/generated-videos/{video_id}")
def delete_generated_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.processed_file_path:
        raise HTTPException(status_code=404, detail="No generated video to delete")
    if os.path.exists(video.processed_file_path):
        os.remove(video.processed_file_path)
    video.processed_file_path = None
    video.processed_status    = "pending"
    video.gemini_analysis     = None
    db.commit()
    return {"ok": True, "message": "Generated video deleted. Original preserved."}


# ══════════════════════════════════════════════════════════
#  ── IMAGE-TO-VIDEO ROUTES (new) ──
# ══════════════════════════════════════════════════════════

@app.get("/dance-styles/")
def get_dance_styles():
    """Return the list of available dance styles for the frontend selector."""
    return {"styles": imgevideo.DANCE_STYLES}


@app.post("/image-video/")
async def create_image_video_job(
    request: Request,
    title: str = Form(...),
    dance_style: str = Form(...),
    dancers_count: int = Form(0),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Upload a photo + choose a dance style → generate a 16-second dance video
    where the person in the photo dances in that style (face preserved).
    """
    # Save image
    file_ext        = file.filename.rsplit(".", 1)[-1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path       = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Create DB record
    job = ImageJob(
        title         = title,
        dance_style   = dance_style,
        dancers_count = dancers_count,
        image_file_path = file_path,
        status        = "pending",
    )
    db.add(job); db.commit(); db.refresh(job)

    # Start background generation
    background_tasks.add_task(image_video_task, job.id)

    base_url = str(request.base_url).rstrip("/")
    return build_imagejob_response(job, base_url)


@app.get("/image-video/", response_model=List[ImageJobResponse])
def list_image_jobs(request: Request, skip: int = 0, limit: int = 100,
                    db: Session = Depends(get_db)):
    jobs     = db.query(ImageJob).order_by(ImageJob.id.desc()).offset(skip).limit(limit).all()
    base_url = str(request.base_url).rstrip("/")
    return [build_imagejob_response(j, base_url) for j in jobs]


@app.get("/image-video/{job_id}", response_model=ImageJobResponse)
def get_image_job(job_id: int, request: Request, db: Session = Depends(get_db)):
    job = db.query(ImageJob).filter(ImageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    base_url = str(request.base_url).rstrip("/")
    return build_imagejob_response(job, base_url)


@app.delete("/image-video/{job_id}")
def delete_image_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ImageJob).filter(ImageJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    for p in [job.image_file_path, job.output_file_path]:
        if p and os.path.exists(p):
            try: os.remove(p)
            except Exception: pass
    db.delete(job); db.commit()
    return {"ok": True}


@app.get("/image-video/{job_id}/download")
def download_image_video(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ImageJob).filter(ImageJob.id == job_id).first()
    if not job or not job.output_file_path or not os.path.exists(job.output_file_path):
        raise HTTPException(status_code=404, detail="Output video not found")
    return FileResponse(
        path=job.output_file_path, media_type="video/mp4",
        filename=f"{job.title}_{job.dance_style}.mp4"
    )


# ── Background task ──────────────────────────────────────
def image_video_task(job_id: int):
    db = next(get_db())
    try:
        job = db.query(ImageJob).filter(ImageJob.id == job_id).first()
        if not job: return

        job.status = "processing"; db.commit()

        output_filename = f"imgvid_{uuid.uuid4()}.mp4"
        output_path     = os.path.join(PROCESSED_DIR, output_filename)

        print(f"[ImgVideo] Job {job_id}: {job.dance_style}, {job.dancers_count} backup dancers")
        result = imgevideo.generate_dance_from_image(
            job.image_file_path,
            job.dance_style,
            job.dancers_count,
            output_path,
        )
        job.status          = result.get("status", "failed")
        job.analysis        = result.get("analysis", "")
        job.output_file_path = result.get("generated_video_path")
        db.commit()
    except Exception as e:
        print(f"[ImgVideo] Job {job_id} error: {e}")
        job.status   = "failed"
        job.analysis = str(e)
        db.commit()
    finally:
        db.close()