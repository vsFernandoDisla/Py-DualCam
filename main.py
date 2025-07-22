import cv2
import time
from collections import deque
import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager
import numpy as np
import os
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
BUFFER_SECONDS = 35
MP4_FOURCC = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4 (MPEG-4 Part 2)
# Consider 'X264' for better compression if ffmpeg is properly configured:
# MP4_FOURCC = cv2.VideoWriter_fourcc(*'X264')

# Global dictionary to hold camera instances
camera_buffers = {}

# === CAMERA BUFFER CLASS ===
class CameraBuffer:
    """
    Manages a circular buffer of video frames from a single camera.
    """
    def __init__(self, index: int):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {index}. Asegúrate de que está conectada y no está en uso.")

        # Get camera properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0 # Ensure float for division
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ensure valid dimensions
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Dimensiones de cámara inválidas para {index}: {self.width}x{self.height}")

        self.buffer = deque(maxlen=int(self.fps * BUFFER_SECONDS))
        self.lock = threading.Lock() # Thread-safety lock for buffer access
        logger.info(f"[Cam {self.index}] Inicializada con FPS: {self.fps}, Dim: {self.width}x{self.height}, Buffer size: {self.buffer.maxlen} frames")

    def record(self):
        """
        Continuously reads frames from the camera and adds them to the buffer.
        Runs in a separate daemon thread.
        """
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.buffer.append(frame)
            else:
                logger.warning(f"[Cam {self.index}] No se pudo leer el frame. Intentando de nuevo...")
                # Consider adding a small delay or re-initialization logic here
                # if frame reading consistently fails.
            time.sleep(1 / self.fps) # Control frame rate

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Retrieves the most recent frame from the buffer in a thread-safe manner.
        Returns None if the buffer is empty.
        """
        with self.lock:
            if self.buffer:
                return self.buffer[-1].copy() # Return a copy to prevent external modification
            return None

    def save_to_file(self, filename: str) -> bool:
        """
        Saves the current buffer content to a video file.
        """
        frames_to_save = []
        with self.lock: # Acquire lock to safely copy the buffer
            if not self.buffer:
                logger.info(f"[Cam {self.index}] Buffer vacío. No se guardará el video '{filename}'.")
                return False
            frames_to_save.extend(list(self.buffer)) # Create a shallow copy of the deque contents

        if not frames_to_save:
            logger.warning(f"[Cam {self.index}] Buffer vacío después de copiar. No se guardará el video '{filename}'.")
            return False

        # Ensure filename has .mp4 extension
        if not filename.lower().endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4" # Robust way to change extension

        out = cv2.VideoWriter(filename, MP4_FOURCC, self.fps, (self.width, self.height))

        if not out.isOpened():
            logger.error(f"[Cam {self.index}] Error: No se pudo crear el archivo de video '{filename}' con FourCC {MP4_FOURCC}")
            logger.error("Asegúrate de que los codecs necesarios (e.g., ffmpeg para H.264) estén instalados y configurados para OpenCV.")
            return False

        try:
            for frame in frames_to_save:
                out.write(frame)
            logger.info(f"[Cam {self.index}] Guardado: '{filename}'")
            return True
        except Exception as e:
            logger.error(f"[Cam {self.index}] Error al escribir frames en '{filename}': {e}")
            return False
        finally:
            out.release() # Ensure the video writer is released

    def release(self):
        """
        Releases the camera capture object.
        """
        self.cap.release()
        logger.info(f"[Cam {self.index}] Cámara liberada.")

# === VIDEO COMBINATION FUNCTION ===
def combine_videos_vertically(cam1_path: str, cam2_path: str, output_path: str) -> bool:
    """
    Combines two video files vertically into a single output video.
    Resizes frames if resolutions differ.
    """
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
        # If resolutions or FPS differ, we'll resize and adjust
        resize_needed = (width1 != width2) or (height1 != height2) or (abs(fps1 - fps2) > 0.1) # Check FPS with tolerance
        
        # Use the maximum width for the combined video
        out_width = max(width1, width2)
        out_height = height1 + height2 # Sum of heights for vertical stack
        
        # Use the FPS of the first video for the output, assuming similar rates
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

            if not ret1 and not ret2: # Both streams ended
                break
            
            # Handle cases where one video is shorter than the other
            if not ret1: 
                logger.warning(f"Video '{cam1_path}' terminó antes que '{cam2_path}'. Rellenando con frame negro para Cam1.")
                frame1 = np.zeros((height1, width1, 3), dtype=np.uint8) # Black frame
            if not ret2: 
                logger.warning(f"Video '{cam2_path}' terminó antes que '{cam1_path}'. Rellenando con frame negro para Cam2.")
                frame2 = np.zeros((height2, width2, 3), dtype=np.uint8) # Black frame

            # Only process if at least one frame was originally read or filled
            if ret1 or ret2: 
                if resize_needed:
                    # Resize frames to match output_width and their original heights
                    frame1 = cv2.resize(frame1, (out_width, height1))
                    frame2 = cv2.resize(frame2, (out_width, height2))

                # Stack vertically
                combined_frame = np.vstack((frame1, frame2))
                out.write(combined_frame)
                frame_count += 1
            else:
                # This case should ideally not be reached if both not ret1 and not ret2 is handled above
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

# === FASTAPI LIFESPAN EVENTS ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes cameras and starts recording threads on startup.
    Releases camera resources on shutdown.
    """
    # --- Startup Events ---
    try:
        logger.info("Iniciando cámaras...")
        # Initialize camera buffers with their respective indices
        cam1_instance = CameraBuffer(0)
        cam2_instance = CameraBuffer(2) # Assuming camera 1 is available

        camera_buffers["cam1"] = cam1_instance
        camera_buffers["cam2"] = cam2_instance

        # Start recording threads as daemon threads
        threading.Thread(target=cam1_instance.record, daemon=True).start()
        threading.Thread(target=cam2_instance.record, daemon=True).start()
        logger.info("Servidor iniciado. POST /save para guardar y combinar buffers. GET /latest_frames para obtener imágenes.")
    except RuntimeError as e:
        logger.critical(f"Error fatal al iniciar cámaras: {e}. El servidor no puede continuar.")
        exit(1) # Exit if cameras cannot be opened

    yield # Application runs

    # --- Shutdown Events ---
    logger.info("Apagando servidor. Liberando recursos de la cámara...")
    for cam_name, cam_buffer in camera_buffers.items():
        cam_buffer.release()
    logger.info("Recursos de la cámara liberados. Servidor apagado.")

# === FASTAPI APPLICATION ===
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. Change to specific origins in production!
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post("/save")
async def save_and_combine_video(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger saving camera buffers and combining them into a single video.
    The video processing is done in a background task to keep the API responsive.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cam1_path = f"cam1_{timestamp}.mp4"
    cam2_path = f"cam2_{timestamp}.mp4"
    output_path = f"combined_{timestamp}.mp4"

    # Define the background task function
    def process_videos_in_background():
        logger.info(f"Iniciando tarea en segundo plano para guardar y combinar videos: {timestamp}")
        
        # Save individual camera buffers
        success_cam1 = camera_buffers["cam1"].save_to_file(cam1_path)
        success_cam2 = camera_buffers["cam2"].save_to_file(cam2_path)

        if not success_cam1 and not success_cam2:
            logger.error(f"No se pudo guardar ningún video para combinar para el timestamp: {timestamp}")
            return {"status": "error", "message": "No se pudo guardar ningún video para combinar."}
        
        # Only attempt to combine if both individual saves were successful
        if success_cam1 and success_cam2:
            success_combine = combine_videos_vertically(cam1_path, cam2_path, output_path)
            if success_combine:
                logger.info(f"Videos combinados y guardados exitosamente: '{output_path}'")
                # Clean up individual camera files
                try:
                    os.remove(cam1_path)
                    os.remove(cam2_path)
                    logger.info(f"Archivos temporales '{cam1_path}' y '{cam2_path}' eliminados.")
                except OSError as e:
                    logger.error(f"Error al eliminar archivos temporales: {e}")
                # This return is for the background task, not directly sent to client
                return {"status": "ok", "message": "Videos combinados", "file": output_path}
            else:
                logger.error(f"Fallo al combinar videos para el timestamp: {timestamp}")
                return {"status": "error", "message": "No se pudo combinar los videos."}
        else:
            logger.warning(f"Uno o ambos videos individuales no se pudieron guardar. No se intentará combinar para: {timestamp}")
            # This return is for the background task, not directly sent to client
            return {"status": "warning", "message": "Uno o ambos videos individuales no se pudieron guardar. No se combinaron."}


    # Add the video processing function to background tasks
    background_tasks.add_task(process_videos_in_background)

    # Return immediate response to the client
    return {"status": "accepted", "message": "Procesando videos en segundo plano. El archivo estará disponible pronto.", "timestamp": timestamp}

@app.get("/latest_frames")
async def get_latest_frames():
    """
    Returns the latest available frame from each camera as base64-encoded JPEG images.
    """
    response_data = {}
    
    for cam_name, cam_buffer in camera_buffers.items():
        frame = cam_buffer.get_latest_frame()
        if frame is not None:
            # Encode frame to JPEG
            # imencode returns (boolean, numpy.ndarray) where ndarray is the encoded image bytes
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]) # Quality 90
            if ret:
                # Convert to base64 string
                base64_image = base64.b64encode(buffer).decode('utf-8')
                response_data[cam_name] = f"data:image/jpeg;base64,{base64_image}"
            else:
                logger.error(f"No se pudo codificar el último frame de {cam_name} a JPEG.")
                response_data[cam_name] = None # Indicate failure for this camera
        else:
            logger.info(f"Buffer de {cam_name} vacío. No hay frames disponibles.")
            response_data[cam_name] = None # Indicate no frame available

    if not any(response_data.values()): # Check if all cameras returned None
        raise HTTPException(status_code=404, detail="No hay frames disponibles de ninguna cámara.")
        
    return response_data

# === START SERVER ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

