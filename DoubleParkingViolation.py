import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from google.cloud import vision
from google.oauth2 import service_account
import io
import json
import time
from sqlalchemy import create_engine, MetaData, Table, Column, DateTime, String , select , delete , LargeBinary
from urllib.parse import quote_plus


# Get the JSON string from environment variable
credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')

# If the credentials are not in an environment variable, use the file path
if not credentials_json:
    credentials_path = "capstone-t5-6e8ba9f61a31.json"
    with open(credentials_path, 'r') as file:
        credentials_json = file.read()

# Parse the JSON string
credentials_dict = json.loads(credentials_json)

# Create credentials object
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

# Use these credentials to create the Vision client
client = vision.ImageAnnotatorClient(credentials=credentials)

def setup_database_connection():
    # Violations database
    violations_db_user = "postgres"
    violations_db_pass = "@Noora1234"
    violations_db_name = "violations"
    violations_db_host = "35.225.36.176"
    violations_db_port = "5432"

    violations_db_pass_encoded = quote_plus(violations_db_pass)
    violations_db_string = f"postgresql://{violations_db_user}:{violations_db_pass_encoded}@{violations_db_host}:{violations_db_port}/{violations_db_name}"
    print(f"Database connection string: {violations_db_string}")
    
    violations_engine = create_engine(violations_db_string)

    # Videos database
    videos_db_user = "postgres"
    videos_db_pass = "@Noora1234"
    videos_db_name = "videos"
    videos_db_host = "35.225.36.176"
    videos_db_port = "5432"

    videos_db_pass_encoded = quote_plus(videos_db_pass)
    videos_db_string = f"postgresql://{videos_db_user}:{videos_db_pass_encoded}@{videos_db_host}:{videos_db_port}/{videos_db_name}"
    
    videos_engine = create_engine(videos_db_string)

    return violations_engine, videos_engine

violations_engine, videos_engine = setup_database_connection()

# Metadata and table for violations database
violations_metadata = MetaData()
violations_table = Table('violations', violations_metadata,
    Column('time_date', DateTime),
    Column('pic', LargeBinary), 
    Column('plate', String)
)

# Metadata and table for videos database
videos_metadata = MetaData()
videos_table = Table('videos', videos_metadata,
    Column('storage_url', String)
)

def get_latest_video_url():
    try:
        with videos_engine.connect() as connection:
            # Select the last row from the videos table
            stmt = select(videos_table.c.storage_url).order_by(videos_table.c.storage_url.desc()).limit(1)
            result = connection.execute(stmt)
            latest_url = result.scalar()
            
            if latest_url:
                print(f"Retrieved latest video URL: {latest_url}")
                return latest_url
            else:
                print("No video URLs found in the database.")
                return None
    except Exception as e:
        print(f"Error retrieving latest video URL: {e}")
        return None


def clean_database_violations():
    try:
        with violations_engine.connect() as connection:
            delete_stmt = delete(violations_table)
            connection.execute(delete_stmt)
            connection.commit()
        print("Database violations cleaned successfully.")
    except Exception as e:
        print(f"Error cleaning database violations: {e}")

def clean_database_videos():
    try:
        with videos_engine.connect() as connection:
            delete_stmt = delete(videos_table)
            connection.execute(delete_stmt)
            connection.commit()
        print("Database videos cleaned successfully.")
    except Exception as e:
        print(f"Error cleaning database videos: {e}")


def delete_processed_video_url(url):
    try:
        with videos_engine.connect() as connection:
            delete_stmt = delete(videos_table).where(videos_table.c.storage_url == url)
            connection.execute(delete_stmt)
            connection.commit()
        print(f"Deleted processed video URL: {url}")
    except Exception as e:
        print(f"Error deleting processed video URL: {e}")


def crop_image(img, bbox, target_size):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
    return resized

def detect_license_plate(image_path):
    license_plate_model = YOLO("license_plate_detect.pt")
    results = license_plate_model(image_path, conf=0.6, iou=0.3)
    if len(results) > 0:
        boxes = results[0].boxes
        if len(boxes) > 0:
            box = boxes[0]
            x, y, w, h = box.xywh[0]
            return {
                'x': x.item(),
                'y': y.item(),
                'width': w.item(),
                'height': h.item()
            }
    return None

def show_image(img, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def detect_english_text(cropped_plate):
    success, encoded_image = cv2.imencode('.png', cropped_plate)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text detected in the plate.")
        return ""

    all_text = texts[0].description if texts else ""
    filtered_text = ''.join(char for char in all_text if char.isascii() and (char.isalnum() or char.isspace()))
    parts = filtered_text.split()

    if "KSA" in parts:
        plate_text = parts[-1]
    else:
        plate_text = ' '.join(part for part in parts if len(part) >= 3 and len(part) <= 4)

    if not (any(c.isalpha() for c in plate_text) and any(c.isdigit() for c in plate_text)):
        letters = ''.join(c for c in filtered_text if c.isalpha())
        numbers = ''.join(c for c in filtered_text if c.isdigit())
        plate_text = f"{letters} {numbers}"

    plate_text = plate_text.replace("KSA", "").strip()
    letters = ''.join(c for c in plate_text if c.isalpha())
    numbers = ''.join(c for c in plate_text if c.isdigit())

    if len(numbers) > 4:
        numbers = numbers[:4]

    plate_text = f"{numbers} {letters}"

    if response.error.message:
        raise Exception(f'{response.error.message}\nFor more info on error messages, check: '
                        'https://cloud.google.com/apis/design/errors')

    return plate_text.strip()

def detect_arabic_text(image, client):
    try:
        success, encoded_image = cv2.imencode('.png', image)
        content = encoded_image.tobytes()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            full_text = texts[0].description
            arabic_text = ''.join(char for char in full_text if '\u0600' <= char <= '\u06FF' or char.isdigit())

            numbers = ''.join(char for char in arabic_text if char.isdigit())
            letters = ''.join(char for char in arabic_text if not char.isdigit())

            numbers = numbers[:4]
            letters = letters[:3]

            if len(numbers) == 4 and len(letters) == 3:
                formatted_text = f"{' '.join(letters)} {' '.join(numbers)}"
                return formatted_text
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error in Arabic text detection: {str(e)}")
        return None

def compare_and_refine_plate(arabic_text, english_text):
    arabic_to_english = {'ا': 'A', 'ب': 'B', 'ح': 'J', 'د': 'D', 'ر': 'R', 'س': 'S', 'ص': 'X', 'ط': 'T', 'ع': 'E', 'ق': 'G', 'ك': 'K', 'ل': 'L', 'م': 'Z', 'ن': 'N', 'ه': 'H', 'و': 'U', 'ى': 'V', 'ي': 'V', 'ی': 'V'}
    english_to_arabic = {v: k for k, v in arabic_to_english.items()}
    number_to_arabic = {'0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'}
    arabic_to_number = {v: k for k, v in number_to_arabic.items()}

    arabic_chars = arabic_text.replace(" ", "")
    english_chars = english_text.replace(" ", "")

    if len(arabic_chars) != 7 or len(english_chars) != 7:
        print("Warning: Input doesn't have 7 characters. Attempting to correct...")
        base = english_chars if len(english_chars) == 7 else arabic_chars
        if len(base) != 7:
            print("Error: Neither input has 7 characters. Cannot refine.")
            return arabic_text, english_text

    refined_arabic = [''] * 7
    refined_english = [''] * 7

    for i in range(7):
        eng_char = english_chars[i]
        ara_char = arabic_chars[6-i]

        if i < 4:
            if eng_char.isdigit():
                refined_english[i] = eng_char
                refined_arabic[6-i] = number_to_arabic.get(eng_char, ara_char)
            elif ara_char in arabic_to_number:
                refined_arabic[6-i] = ara_char
                refined_english[i] = arabic_to_number[ara_char]
            else:
                refined_english[i] = eng_char
                refined_arabic[6-i] = number_to_arabic.get(eng_char, ara_char)
        else:
            if eng_char.upper() in english_to_arabic:
                refined_english[i] = eng_char.upper()
                refined_arabic[6-i] = english_to_arabic[eng_char.upper()]
            elif ara_char in arabic_to_english:
                refined_arabic[6-i] = ara_char
                refined_english[i] = arabic_to_english[ara_char]
            else:
                refined_english[i] = eng_char.upper()
                refined_arabic[6-i] = english_to_arabic.get(eng_char.upper(), ara_char)

    refined_arabic = ' '.join(refined_arabic[:4]) + ' ' + ' '.join(refined_arabic[4:])
    refined_english = ''.join(refined_english[:4]) + ' ' + ''.join(refined_english[4:])

    return refined_arabic, refined_english

def process_license_plate(car_image):
    try:
        if car_image is None:
            print("Failed to load the image.")
            return None, None, None, None, None

        prediction = detect_license_plate(car_image)

        if prediction:
            height, width = car_image.shape[:2]
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)

            cropped_plate = crop_image(car_image, (x1, y1, x2, y2), (width, height))

            arabic_text = detect_arabic_text(cropped_plate, client)
            english_text = detect_english_text(cropped_plate)

            print("Detected license plate:")
            if arabic_text:
                print("Arabic:", arabic_text)
            if english_text:
                print("English:", english_text)

            try:
                if arabic_text is None:
                    arabic_text = "ا ب ت ١ ٢ ٣ ٤"
                    print("Warning: Arabic text detection failed. Using placeholder.")

                refined_arabic, refined_english = compare_and_refine_plate(arabic_text, english_text)
                print("Refined license plate:")
                print("Arabic:", refined_arabic)
                print("English:", refined_english)

                if not refined_arabic or not refined_english:
                    print("Warning: Refinement process produced empty results. Using original detections.")
                    refined_arabic = arabic_text if arabic_text else refined_arabic
                    refined_english = english_text if english_text else refined_english

                arabic_parts = refined_arabic.split()
                english_parts = refined_english.split()

                if len(arabic_parts) > 1:
                    refined_arabic = ''.join(arabic_parts[:4]) + ' ' + ' '.join(arabic_parts[4:])
                if len(english_parts) > 1:
                    refined_english = ''.join(english_parts[:4]) + ' ' + ''.join(english_parts[4:])

            except Exception as e:
                print(f"Error refining license plate: {str(e)}")
                refined_arabic, refined_english = arabic_text, english_text

            return car_image, cropped_plate, refined_arabic, refined_english, (x1, y1, x2, y2)
        else:
            print("No license plate detected.")
            return car_image, None, None, None, None
    except Exception as e:
        print(f"Error processing license plate: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def process_video(video_path, process_every_n_frames=5, violation_threshold_seconds=30):
    car_model = YOLO("vehicles_detect.pt")
    classes = [2, 3, 6]
    FIXED_WIDTH = 3840
    FIXED_HEIGHT = 2160
    points = np.array([
        [0, 1050],
        [3800, 960],
        [3800, 1650],
        [0, 1650],
    ], np.int32)
    points = points.reshape((-1, 1, 2))

    def point_inside_polygon(x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0][0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n][0]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_dir = 'violated_objects'
    os.makedirs(save_dir, exist_ok=True)

    # Create video_output folder if it doesn't exist
    video_output_dir = 'video_output'
    os.makedirs(video_output_dir, exist_ok=True)

    # Set up video writer for the output video
    output_video_path = os.path.join(video_output_dir, 'processed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (FIXED_WIDTH, FIXED_HEIGHT))

    objects = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to fixed size
        frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))

        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            out.write(frame)  # Write the unprocessed frame to the output video
            continue

        start_time = time.time()

        clean_frame = frame.copy()

        results = car_model.track(frame, persist=True, classes=classes, tracker="botsort.yaml")

        cv2.polylines(frame, [points], True, (0, 255, 0), 2)

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if point_inside_polygon(center_x, center_y, points):
                    if track_id not in objects:
                        objects[track_id] = {'start_time': time.time(), 'violated': False}

                    duration = time.time() - objects[track_id]['start_time']

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ID: {track_id}, Time: {duration:.1f}s", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if duration >= violation_threshold_seconds and not objects[track_id]['violated']:
                        objects[track_id]['violated'] = True

                        full_screenshot_path = os.path.join(save_dir, f"violated_{track_id}_full.jpg")
                        cv2.imwrite(full_screenshot_path, clean_frame)

                        enhanced_cropped = crop_image(clean_frame, box, (frame_width, frame_height))
                        car_image, cropped_plate, refined_arabic, refined_english, bbox = process_license_plate(enhanced_cropped)

                        if refined_english:
                            plate_text = refined_english
                        else:
                            plate_text = None
                        
                        if plate_text: #then if there is plate insert 
                            clean_screenshot_path = os.path.join(save_dir, f"violated_{track_id}_clean_frame.jpg")
                            cv2.imwrite(clean_screenshot_path, clean_frame)
                            print(f"Saved new violated object: {track_id}")
                        # Add to violations database (time , pic , plate in english)
                            try:
                                current_time = datetime.now()
                                with open(clean_screenshot_path, 'rb') as image_file:
                                    image_binary = image_file.read()
                                with violations_engine.connect() as connection:
                                    ins = violations_table.insert().values(
                                        time_date=current_time,
                                        pic=image_binary,
                                        plate=plate_text
                                        )
                                    connection.execute(ins)
                                    connection.commit()
                                print(f"Added to database: Plate {plate_text} at {current_time}")
                            except Exception as e:
                                print(f"Error adding to database: {e}")
                        else:
                            print("No valid license plate detected. Skipping database entry.")

                    if objects[track_id]['violated']:
                        cv2.putText(frame, "VIOLATED", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        out.write(frame)  # Write the processed frame to the output video

        elapsed_time = time.time() - start_time
        if elapsed_time < frame_time * process_every_n_frames:
            time.sleep(frame_time * process_every_n_frames - elapsed_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  
    cv2.destroyAllWindows()

    print(f"Processed video saved to: {output_video_path}")

def main(latest_video_url):
    latest_video_url = get_latest_video_url()

    if latest_video_url:
        process_video(latest_video_url, process_every_n_frames=5, violation_threshold_seconds=30)
    else:
        print("No video URL available to process.")
    
    delete_processed_video_url(latest_video_url)


