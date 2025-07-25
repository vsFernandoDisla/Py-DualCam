# video_utils.py

import cv2
import numpy as np
import os
import httpx # Import httpx for async HTTP requests
import time # For tracking token expiry
import slugify # For slugifying filenames

from config import (
    logger, MP4_FOURCC, VIDEO_OUTPUT_DIR,
    FIREBASE_API_KEY, DISERCOIN_API_EMAIL, DISERCOIN_API_PASSWORD,
    REMOTE_SERVER_URL
)

# --- Global Token Storage (in-memory for this example) ---
# In a production environment, consider more robust token storage (e.g., Redis, database)
# if multiple workers or restarts could cause issues.
_current_id_token: str | None = None
_refresh_token: str | None = None
_token_expiry_time: float = 0.0 # Unix timestamp (seconds) when ID token expires

# --- Authentication Endpoints ---
FIREBASE_SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
FIREBASE_REFRESH_TOKEN_URL = "https://securetoken.googleapis.com/v1/token"

async def _get_firebase_id_token() -> str:
    """
    Authenticates with Firebase using email/password and retrieves an ID token.
    Updates the global token storage.
    """
    global _current_id_token, _refresh_token, _token_expiry_time

    if not all([FIREBASE_API_KEY, DISERCOIN_API_EMAIL, DISERCOIN_API_PASSWORD]):
        logger.error("❌ Error: Missing Firebase API Key, Email, or Password environment variables.")
        raise ValueError("Firebase authentication credentials are not set.")

    async with httpx.AsyncClient() as client:
        try:
            logger.info("🔑 Authenticating with Firebase for Disercoin API token...")
            response = await client.post(
                FIREBASE_SIGN_IN_URL,
                json={
                    "email": DISERCOIN_API_EMAIL,
                    "password": DISERCOIN_API_PASSWORD,
                    "returnSecureToken": True,
                },
                params={"key": FIREBASE_API_KEY},
                timeout=10.0
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            data = response.json()
            _current_id_token = data["idToken"]
            _refresh_token = data["refreshToken"]
            # Calculate expiry time: current time (seconds) + expires_in (seconds)
            _token_expiry_time = time.time() + int(data["expiresIn"]) - 60 # 60 sec buffer

            logger.info(f"✅ Firebase ID Token obtained. Expires in {int(data['expiresIn'])} seconds.")
            return _current_id_token

        except httpx.RequestError as e:
            logger.error(f"❌ Network error during Firebase authentication: {e}")
            raise RuntimeError(f"Failed to connect to Firebase auth: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Firebase authentication failed: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Firebase authentication error: {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during Firebase authentication: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Firebase auth error: {e}")


async def _refresh_firebase_id_token() -> str:
    """
    Refreshes the Firebase ID token using the refresh token.
    Updates the global token storage.
    """
    global _current_id_token, _refresh_token, _token_expiry_time

    if not _refresh_token:
        logger.warning("No refresh token available. Attempting full re-authentication.")
        return await _get_firebase_id_token() # Fallback to full authentication

    if not FIREBASE_API_KEY:
        logger.error("❌ Error: Missing Firebase API Key environment variable for token refresh.")
        raise ValueError("Firebase API Key is not set for token refresh.")

    async with httpx.AsyncClient() as client:
        try:
            logger.info("🔄 Refreshing Firebase ID Token...")
            response = await client.post(
                FIREBASE_REFRESH_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refreshToken": _refresh_token,
                },
                params={"key": FIREBASE_API_KEY},
                timeout=10.0
            )
            response.raise_for_status()

            data = response.json()
            _current_id_token = data["id_token"] # Note 'id_token' for refresh response
            _refresh_token = data.get("refresh_token", _refresh_token) # refresh_token might not always be returned
            _token_expiry_time = time.time() + int(data["expires_in"]) - 60 # 60 sec buffer

            logger.info(f"✅ Firebase ID Token refreshed. New token expires in {int(data['expires_in'])} seconds.")
            return _current_id_token

        except httpx.RequestError as e:
            logger.error(f"❌ Network error during Firebase token refresh: {e}")
            raise RuntimeError(f"Failed to connect to Firebase token refresh: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Firebase token refresh failed: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Firebase token refresh error: {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during Firebase token refresh: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Firebase refresh error: {e}")


async def get_valid_firebase_token() -> str:
    """
    Retrieves a valid Firebase ID token, refreshing it if expired.
    """
    global _current_id_token, _token_expiry_time

    # Check if token exists and is not expired (considering buffer)
    if _current_id_token and time.time() < _token_expiry_time:
        return _current_id_token
    
    # If expired or doesn't exist, try to refresh
    try:
        return await _refresh_firebase_id_token()
    except Exception as e:
        logger.warning(f"Failed to refresh token: {e}. Attempting initial authentication.")
        # If refresh fails, try to get a new token from scratch
        return await _get_firebase_id_token()


# --- Video Processing Functions (No changes here, just for context) ---
def combine_videos_vertically(cam1_filename: str, cam2_filename: str, output_filename: str) -> bool:
    # ... (Keep this function as is) ...
    cam1_path = os.path.join(VIDEO_OUTPUT_DIR, cam1_filename)
    cam2_path = os.path.join(VIDEO_OUTPUT_DIR, cam2_filename)
    output_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)

    logger.info(f"Iniciando combinación de videos: '{cam1_path}' y '{cam2_path}' en '{output_path}'")
    cap1 = cv2.VideoCapture(cam1_path)
    cap2 = cv2.VideoCapture(cam2_path)

    if not cap1.isOpened():
        logger.error(f"Error abriendo el primer video: '{cam1_path}'")
        return False
    if not cap2.isOpened():
        logger.error(f"Error abriendo el segundo video: '{cam2_path}'")
        cap1.release() # Release already opened cap
        return False

    try:
        # Get properties for cam1
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps1 = cap1.get(cv2.CAP_PROP_FPS)

        # Get properties for cam2
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps2 = cap2.get(cv2.CAP_PROP_FPS)

        if fps1 <= 0 or fps2 <= 0:
            logger.error(f"FPS inválido detectado: Cam1 FPS={fps1}, Cam2 FPS={fps2}. No se puede combinar.")
            return False

        # Determine if resizing is needed and calculate output dimensions
        resize_needed = (width1 != width2) or (height1 != height2) or (abs(fps1 - fps2) > 0.1)
        
        out_width = max(width1, width2)
        out_height = height1 + height2
        
        output_fps = fps1 

        out = cv2.VideoWriter(output_path, MP4_FOURCC, output_fps, (out_width, out_height))

        if not out.isOpened():
            logger.error(f"Error: No se pudo crear el archivo de video combinado '{output_path}' con FourCC {MP4_FOURCC}")
            logger.error("Asegúrate de que los codecs necesarios (e.g., ffmpeg para H.264) estén instalados y configurados para OpenCV.")
            return False

        frame_count = 0
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 and not ret2:
                break
            
            if not ret1: 
                logger.warning(f"Video '{cam1_filename}' terminó antes que '{cam2_filename}'. Rellenando con frame negro para Cam1.")
                frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)
            if not ret2: 
                logger.warning(f"Video '{cam2_filename}' terminó antes que '{cam1_filename}'. Rellenando con frame negro para Cam2.")
                frame2 = np.zeros((height2, width2, 3), dtype=np.uint8)

            if ret1 or ret2: 
                if resize_needed:
                    frame1 = cv2.resize(frame1, (out_width, height1))
                    frame2 = cv2.resize(frame2, (out_width, height2))

                combined_frame = np.vstack((frame1, frame2))
                out.write(combined_frame)
                frame_count += 1
            else:
                break

        logger.info(f"Video combinado guardado en: '{output_path}' con {frame_count} frames.")
        return True

    except Exception as e:
        logger.error(f"Error durante la combinación de videos: {e}", exc_info=True)
        return False
    finally:
        cap1.release()
        cap2.release()
        out.release()


# --- Modified: upload_video_to_remote_server with Authentication ---
async def upload_video_to_remote_server(local_file_path: str):
    """
    Uploads a local video file to the specified remote server endpoint with authentication.
    """
    try:
        # Get a valid Firebase ID token
        id_token = await get_valid_firebase_token()
        
        # Extract filename and extension for the remote path
        base_filename = os.path.basename(local_file_path)
        name_without_ext, extension = os.path.splitext(base_filename)
        
        sanitized_name = name_without_ext.replace("TEMP-", "")
        slugified_name = slugify.slugify(sanitized_name)

        upload_url = f"{REMOTE_SERVER_URL}/azure-blob/upload/transaction-video/{slugified_name}/{extension.lstrip('.')}"
        
        logger.info(f"Iniciando subida de '{local_file_path}' a '{upload_url}' con token de autenticación.")

        async with httpx.AsyncClient() as client:
            with open(local_file_path, "rb") as f:
                response = await client.post(
                    upload_url,
                    content=f.read(),
                    headers={
                        "Content-Type": "video/mp4",
                        "Authorization": f"Bearer {id_token}" # <-- ADDED AUTHORIZATION HEADER
                    },
                    timeout=300.0
                )
        
        response.raise_for_status()
        logger.info(f"✅ Video '{local_file_path}' subido exitosamente al servidor remoto. Respuesta: {response.status_code}")
        return True
    except RuntimeError as e: # Catch errors specifically from token acquisition
        logger.error(f"❌ Fallo la autenticación o adquisición de token para la subida: {e}")
        return False
    except httpx.RequestError as e:
        logger.error(f"❌ Error de red/conexión al subir '{local_file_path}': {e}")
        return False
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Error HTTP al subir '{local_file_path}': {e.response.status_code} - {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado al subir '{local_file_path}': {e}", exc_info=True)
        return False