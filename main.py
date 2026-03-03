from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from typing import List

from database import engine, get_db, init_db, Video
from models import VideoCreate, VideoResponse, VideoUpdate
import video_processor

# Initialize DB
init_db()

app = FastAPI(title="Dance Crew Backend")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Mount static files to serve videos
app.mount("/static", StaticFiles(directory="."), name="static")

def build_video_response(video: Video, base_url: str) -> dict:
    """Build a response dict with full video URLs."""
    # Convert Windows backslashes to forward slashes for URLs
    original_path = video.original_file_path.replace("\\", "/") if video.original_file_path else None
    processed_path = video.processed_file_path.replace("\\", "/") if video.processed_file_path else None
    
    data = {
        "id": video.id,
        "title": video.title,
        "description": video.description,
        "dancers_count": video.dancers_count,
        "filename": video.filename,
        "processed_status": video.processed_status,
        "uploaded_at": video.uploaded_at,
        "original_file_path": video.original_file_path,
        "processed_file_path": video.processed_file_path,
        "gemini_analysis": video.gemini_analysis,
        "original_video_url": f"{base_url}/static/{original_path}" if original_path else None,
        "generated_video_url": f"{base_url}/static/{processed_path}" if processed_path else None,
    }
    return data

@app.post("/videos/", response_model=VideoResponse)
async def upload_video(
    request: Request,
    title: str = Form(...),
    dancers_count: int = Form(0),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    # Save file locally
    file_ext = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create DB entry
    db_video = Video(
        filename=unique_filename,
        title=title,
        dancers_count=dancers_count,
        original_file_path=file_path
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    
    # Auto-trigger processing if dancers_count > 0
    if dancers_count > 0:
        background_tasks.add_task(process_video_task, db_video.id, dancers_count, None)
    
    base_url = str(request.base_url).rstrip("/")
    return build_video_response(db_video, base_url)

@app.get("/videos/", response_model=List[VideoResponse])
def read_videos(request: Request, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    videos = db.query(Video).offset(skip).limit(limit).all()
    base_url = str(request.base_url).rstrip("/")
    return [build_video_response(v, base_url) for v in videos]

@app.get("/videos/{video_id}", response_model=VideoResponse)
def read_video(video_id: int, request: Request, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    base_url = str(request.base_url).rstrip("/")
    return build_video_response(video, base_url)

@app.delete("/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete original file
    if video.original_file_path and os.path.exists(video.original_file_path):
        os.remove(video.original_file_path)
    # Delete processed file
    if video.processed_file_path and os.path.exists(video.processed_file_path):
        os.remove(video.processed_file_path)
        
    db.delete(video)
    db.commit()
    return {"ok": True}

def process_video_task(video_id: int, dancers_count: int, db_session: Session):
    """Background task: generates a group dance video using Veo."""
    db = next(get_db()) 
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return

        video.processed_status = "processing"
        video.dancers_count = dancers_count
        db.commit()

        # Generate output path
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        # Call the full video generation pipeline
        print(f"Processing video {video_id} with {dancers_count} dancers...")
        result = video_processor.generate_group_dance_video(
            video.original_file_path, dancers_count, output_path
        )
        
        video.gemini_analysis = result.get("analysis", "")
        video.processed_status = result.get("status", "failed")
        video.processed_file_path = result.get("generated_video_path")
        
        db.commit()
    except Exception as e:
        print(f"Error processing video: {e}")
        video.processed_status = "failed"
        video.gemini_analysis = str(e)
        db.commit()
    finally:
        db.close()

@app.post("/videos/{video_id}/process")
async def process_video(
    video_id: int, 
    dancers_count: int, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    background_tasks.add_task(process_video_task, video_id, dancers_count, None)
    
    return {"message": "Processing started", "video_id": video_id}


# ============================================================
# Generated Videos — CRUD Routes
# ============================================================

@app.get("/generated-videos/", response_model=List[VideoResponse])
def list_generated_videos(request: Request, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all videos that have a generated (processed) video."""
    videos = db.query(Video).filter(
        Video.processed_file_path.isnot(None),
        Video.processed_status == "completed"
    ).offset(skip).limit(limit).all()
    base_url = str(request.base_url).rstrip("/")
    return [build_video_response(v, base_url) for v in videos]

@app.get("/generated-videos/{video_id}", response_model=VideoResponse)
def get_generated_video(video_id: int, request: Request, db: Session = Depends(get_db)):
    """Get details of a specific generated video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.processed_file_path or video.processed_status != "completed":
        raise HTTPException(status_code=404, detail="No generated video available for this video")
    base_url = str(request.base_url).rstrip("/")
    return build_video_response(video, base_url)

@app.get("/generated-videos/{video_id}/download")
def download_generated_video(video_id: int, db: Session = Depends(get_db)):
    """Download the generated video file."""
    from fastapi.responses import FileResponse
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.processed_file_path or not os.path.exists(video.processed_file_path):
        raise HTTPException(status_code=404, detail="Generated video file not found")
    return FileResponse(
        path=video.processed_file_path,
        media_type="video/mp4",
        filename=f"{video.title}_generated.mp4"
    )

@app.delete("/generated-videos/{video_id}")
def delete_generated_video(video_id: int, db: Session = Depends(get_db)):
    """Delete only the generated video (keeps the original upload)."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.processed_file_path:
        raise HTTPException(status_code=404, detail="No generated video to delete")
    
    # Delete the generated file
    if os.path.exists(video.processed_file_path):
        os.remove(video.processed_file_path)
    
    # Clear DB fields
    video.processed_file_path = None
    video.processed_status = "pending"
    video.gemini_analysis = None
    db.commit()
    
    return {"ok": True, "message": "Generated video deleted. Original video preserved."}
