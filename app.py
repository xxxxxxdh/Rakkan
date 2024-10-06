from fastapi import FastAPI, UploadFile, File, Request, Query, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, MetaData, Table, Column, String, insert, func, desc, extract, Integer, DateTime, LargeBinary, select, text
from sqlalchemy.orm import sessionmaker, Session
from google.cloud import storage
from urllib.parse import quote_plus
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import uvicorn
import base64
from datetime import datetime, timedelta
import logging
from DoubleParkingViolation import main
import json
from google.oauth2 import service_account

import os
import json
from google.oauth2 import service_account
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the JSON string from environment variable
credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

# If the credentials are not in an environment variable, use the file path
if not credentials_json:
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    with open(credentials_path, 'r') as file:
        credentials_json = file.read()

# Parse the JSON string
credentials_dict = json.loads(credentials_json)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

# Use the credentials
client = storage.Client(credentials=credentials)


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the templates directory to serve static files
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Templates configuration
templates = Jinja2Templates(directory="templates")

# Database configurations
videos_db_user = "postgres"
videos_db_pass = "@Noora1234"
videos_db_name = "videos"
videos_db_host = "35.225.36.176"
videos_db_port = "5432"

violations_db_user = "postgres"
violations_db_pass = "@Noora1234"
violations_db_name = "violations"
violations_db_host = "35.225.36.176"
violations_db_port = "5432"

videos_db_pass_encoded = quote_plus(videos_db_pass)
videos_db_string = f"postgresql://{videos_db_user}:{videos_db_pass_encoded}@{videos_db_host}:{videos_db_port}/{videos_db_name}"

violations_db_pass_encoded = quote_plus(violations_db_pass)
violations_db_string = f"postgresql://{violations_db_user}:{violations_db_pass_encoded}@{violations_db_host}:{violations_db_port}/{violations_db_name}"

# Database setups
videos_engine = create_engine(videos_db_string)
videos_metadata = MetaData()
videos_table = Table('videos', videos_metadata,
    Column('storage_url', String)
)

violations_engine = create_engine(violations_db_string)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=violations_engine)

violations_metadata = MetaData()
violations_table = Table('violations', violations_metadata,
    Column('time_date', DateTime),
    Column('pic', LargeBinary),
    Column('plate', String)
)

# Google Cloud Storage configuration
bucket_name = "bucket-capstonet5"
chunk_size = 1 * 1024 * 1024  # 1 MiB

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Pydantic models
class OverviewStats(BaseModel):
    total_violations: int
    unique_plates: int
    peak_hour: Optional[str]
    total_violations_today: int
    violation_times: dict
    top_violating_times: List[dict]

class ViolationTime(BaseModel):
    hour: str
    count: int

class DailyViolation(BaseModel):
    time_date: str
    plate: str
    pic: Optional[str]

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection error")
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    bucket = storage_client.bucket(bucket_name)
    if bucket.exists():
        print(f"Using existing bucket: {bucket_name}")
    else:
        print(f"Bucket {bucket_name} does not exist. Please create it manually or update the bucket name.")

@app.get("/", response_class=HTMLResponse)
async def read_home():
    with open("templates/home.html", "r") as f:
        return f.read()

@app.get("/upload", response_class=HTMLResponse)
async def read_upload(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard():
    with open("templates/dashboard.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        filename = f"{uuid.uuid4()}_{file.filename}"
        storage_url = await upload_to_gcs_and_db(file, filename)
        return JSONResponse(content={"message": "Upload completed", "filename": filename, "storage_url": storage_url})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/start_processing/")
async def start_processing(request: Request):
    try:
        data = await request.json()
        storage_url = data.get('storage_url')
        if not storage_url:
            return JSONResponse(content={"error": "No storage_url provided"}, status_code=400)
        main_result = main(storage_url)
        return JSONResponse(content={"message": "Processing completed", "result": main_result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/overview_stats", response_model=OverviewStats)
def overview_stats(db: Session = Depends(get_db)):
    try:
        total_violations = db.scalar(select(func.count(violations_table.c.time_date)))
        unique_plates = db.scalar(select(func.count(func.distinct(violations_table.c.plate))))
        
        peak_hour = db.execute(
            select(
                func.date_trunc('hour', violations_table.c.time_date).label('hour'),
                func.count(violations_table.c.time_date).label('count')
            ).group_by(text('1'))
            .order_by(desc('count'))
            .limit(1)
        ).first()
        
        today = datetime.now().date()
        total_violations_today = db.scalar(
            select(func.count(violations_table.c.time_date))
            .where(func.date(violations_table.c.time_date) == today)
        )
        
        violation_times = db.execute(
            select(
                extract('hour', violations_table.c.time_date).label('hour'),
                func.count(violations_table.c.time_date).label('count')
            ).group_by(text('1'))
            .order_by(text('1'))
        ).all()
        
        violation_times_dict = {int(row.hour): row.count for row in violation_times}
        
        top_violating_times = db.execute(
            select(
                func.date_trunc('hour', violations_table.c.time_date).label('time'),
                func.count(violations_table.c.time_date).label('count')
            ).group_by(text('1'))
            .order_by(desc('count'))
            .limit(5)
        ).all()
        
        return {
            'total_violations': total_violations,
            'unique_plates': unique_plates,
            'peak_hour': peak_hour.hour.strftime('%H:%M') if peak_hour else None,
            'total_violations_today': total_violations_today,
            'violation_times': violation_times_dict,
            'top_violating_times': [{'time': t.time.strftime('%H:%M'), 'count': t.count} for t in top_violating_times]
        }
    except Exception as e:
        logger.error(f"Error in overview_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/violation_times", response_model=List[ViolationTime])
def violation_times(range: str = Query(..., description="Time range: day, week, or month"), db: Session = Depends(get_db)):
    try:
        if range == 'day':
            start_date = datetime.now() - timedelta(days=1)
            group_by = func.date_trunc('hour', violations_table.c.time_date)
        elif range == 'week':
            start_date = datetime.now() - timedelta(weeks=1)
            group_by = func.date_trunc('day', violations_table.c.time_date)
        else:  # month
            start_date = datetime.now() - timedelta(days=30)
            group_by = func.date_trunc('day', violations_table.c.time_date)
        
        violations = db.execute(
            select(
                group_by.label('hour'),
                func.count(violations_table.c.time_date).label('count')
            ).where(violations_table.c.time_date >= start_date)
            .group_by(text('1'))
            .order_by(text('1'))
        ).all()
        
        return [{'hour': v.hour.isoformat(), 'count': v.count} for v in violations]
    except Exception as e:
        logger.error(f"Error in violation_times: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/daily_violations", response_model=List[DailyViolation])
def daily_violations(date: str = Query(..., description="Date in YYYY-MM-DD format"), db: Session = Depends(get_db)):
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        violations = db.execute(
            select(violations_table).where(
                func.date(violations_table.c.time_date) == date_obj.date()
            ).order_by(violations_table.c.time_date)
        ).all()
        
        return [{
            'time_date': v.time_date.isoformat(),
            'plate': v.plate,
            'pic': base64.b64encode(v.pic).decode('utf-8') if v.pic else None
        } for v in violations]
    except ValueError as ve:
        logger.error(f"Error parsing date in daily_violations: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid date format. Please use YYYY-MM-DD.")
    except Exception as e:
        logger.error(f"Error in daily_violations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/test_db")
def test_db(db: SessionLocal = Depends(get_db)):
    try:
        result = db.execute("SELECT 1").scalar()
        return {"status": "success", "message": "Database connection successful"}
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database test failed: {str(e)}")

async def upload_to_gcs_and_db(file: UploadFile, filename: str):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.chunk_size = chunk_size
        with blob.open("wb") as f:
            while chunk := await file.read(chunk_size):
                f.write(chunk)
        storage_url = f"https://storage.googleapis.com/{bucket_name}/{filename}"
        with videos_engine.connect() as conn:
            stmt = insert(videos_table).values(storage_url=storage_url)
            conn.execute(stmt)
            conn.commit()
        print(f"Video uploaded successfully: {storage_url}")
        return storage_url
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)