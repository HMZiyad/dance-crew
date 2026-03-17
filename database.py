from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./dance_crew.db"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class Video(Base):
    __tablename__ = "videos"

    id               = Column(Integer, primary_key=True, index=True)
    filename         = Column(String, index=True)
    title            = Column(String, index=True)
    description      = Column(String, nullable=True)
    processed_status = Column(String, default="pending")   # pending/processing/completed/failed
    dancers_count    = Column(Integer, default=0)
    uploaded_at      = Column(DateTime, default=datetime.utcnow)
    original_file_path  = Column(String)
    processed_file_path = Column(String, nullable=True)
    gemini_analysis     = Column(String, nullable=True)


class ImageJob(Base):
    """Tracks image-to-dance-video generation jobs."""
    __tablename__ = "image_jobs"

    id               = Column(Integer, primary_key=True, index=True)
    title            = Column(String, index=True)
    dance_style      = Column(String)                      # e.g. "hip-hop"
    dancers_count    = Column(Integer, default=0)          # backup dancers
    status           = Column(String, default="pending")   # pending/processing/completed/failed
    created_at       = Column(DateTime, default=datetime.utcnow)
    image_file_path  = Column(String)                      # uploaded reference photo
    output_file_path = Column(String, nullable=True)       # generated MP4
    analysis         = Column(String, nullable=True)        # Gemini analysis + prompt used


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()