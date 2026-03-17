from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class VideoBase(BaseModel):
    title: str
    description: Optional[str] = None
    dancers_count: int = 0

class VideoCreate(VideoBase):
    pass

class VideoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    dancers_count: Optional[int] = None
    processed_status: Optional[str] = None

class VideoResponse(VideoBase):
    id: int
    filename: str
    processed_status: str
    uploaded_at: datetime
    original_file_path: str
    processed_file_path: Optional[str] = None
    gemini_analysis: Optional[str] = None
    original_video_url: Optional[str] = None
    generated_video_url: Optional[str] = None

    class Config:
        from_attributes = True

class ImageJobResponse(BaseModel):
    id: int
    title: str
    dance_style: str
    dancers_count: int
    status: str
    created_at: datetime
    image_file_path: Optional[str] = None
    output_file_path: Optional[str] = None
    analysis: Optional[str] = None
    image_url: Optional[str] = None
    generated_video_url: Optional[str] = None
 
    class Config:
        from_attributes = True
 