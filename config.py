# config.py

import cv2
import logging
import os
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Camera Buffer Configuration ---
BUFFER_SECONDS = 35

# --- Video Codec Configuration ---
MP4_FOURCC = cv2.VideoWriter_fourcc(*'mp4v') 

# --- Directory for saved videos ---
VIDEO_OUTPUT_DIR = "recorded_videos"
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True) # Ensure directory exists

# --- Firebase Authentication Configuration ---
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
DISERCOIN_API_EMAIL = os.getenv("DISERCOIN_API_EMAIL")
DISERCOIN_API_PASSWORD = os.getenv("DISERCOIN_API_PASSWORD")

# --- Remote Server Configuration ---
REMOTE_SERVER_URL = "http://10.0.0.127:3003"