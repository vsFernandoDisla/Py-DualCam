# main.py

import time
import threading
import base64
import os
from contextlib import asynccontextmanager

import uvicorn
import cv2 # Only needed for imencode here
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import logger, VIDEO_OUTPUT_DIR
from camera_buffer import CameraBuffer
from video_utils import combine_videos_vertically, upload_video_to_remote_server

# Global dictionary to hold camera instances
camera_buffers = {}
class VideoData(BaseModel):
    videoName: str

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
        cam2_instance = CameraBuffer(2) # Assuming camera 2 is available

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
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. Change to specific origins in production!
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/save")
async def save_and_combine_video(background_tasks: BackgroundTasks, data: VideoData):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cam1_filename = f"cam1_{timestamp}.mp4"
    cam2_filename = f"cam2_{timestamp}.mp4"
    output_filename = f"{data.videoName}.mp4"
    output_full_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)

    async def process_videos_in_background():
        logger.info(f"Iniciando tarea en segundo plano para guardar, combinar y subir videos: {timestamp}")
        
        success_cam1 = camera_buffers["cam1"].save_to_file(cam1_filename)
        success_cam2 = camera_buffers["cam2"].save_to_file(cam2_filename)

        if not success_cam1 and not success_cam2:
            logger.error(f"No se pudo guardar ningún video para combinar para el timestamp: {timestamp}")
            return {"status": "error", "message": "No se pudo guardar ningún video para combinar."}
        
        if success_cam1 and success_cam2:
            success_combine = combine_videos_vertically(cam1_filename, cam2_filename, output_filename)
            if success_combine:
                logger.info(f"Videos combinados y guardados exitosamente: '{output_full_path}'")
                
                logger.info(f"Iniciando subida de '{output_filename}' al servidor remoto...")
                upload_success = await upload_video_to_remote_server(output_full_path) # This call now includes auth
                
                if upload_success:
                    logger.info(f"✅ Video '{output_filename}' subido exitosamente al servidor remoto.")
                    # try:
                    #     os.remove(os.path.join(VIDEO_OUTPUT_DIR, cam1_filename))
                    #     os.remove(os.path.join(VIDEO_OUTPUT_DIR, cam2_filename))
                    #     os.remove(output_full_path)
                    #     logger.info(f"Archivos temporales '{cam1_filename}', '{cam2_filename}' y '{output_filename}' eliminados.")
                    # except OSError as e:
                    #     logger.error(f"Error al eliminar archivos temporales: {e}")
                    return {"status": "ok", "message": "Videos combinados y subidos", "file": output_filename}
                else:
                    logger.error(f"❌ Fallo la subida del video combinado '{output_filename}' al servidor remoto.")
                    return {"status": "error", "message": "No se pudo subir el video combinado al servidor remoto."}
            else:
                logger.error(f"Fallo al combinar videos para el timestamp: {timestamp}")
                return {"status": "error", "message": "No se pudo combinar los videos."}
        else:
            logger.warning(f"Uno o ambos videos individuales no se pudieron guardar. No se intentará combinar ni subir para: {timestamp}")
            return {"status": "warning", "message": "Uno o ambos videos individuales no se pudieron guardar. No se combinaron ni subieron."}

    background_tasks.add_task(process_videos_in_background)

    return {"status": "accepted", "message": "Procesando videos en segundo plano. El archivo se subirá al servidor remoto.", "timestamp": timestamp}

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
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)