# camera_buffer.py

import cv2
import time
from collections import deque
import threading
import numpy as np
import os

from config import logger, BUFFER_SECONDS, MP4_FOURCC, VIDEO_OUTPUT_DIR

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

        full_path = os.path.join(VIDEO_OUTPUT_DIR, filename)
        # Ensure filename has .mp4 extension
        if not full_path.lower().endswith(".mp4"):
            full_path = os.path.splitext(full_path)[0] + ".mp4" # Robust way to change extension

        out = cv2.VideoWriter(full_path, MP4_FOURCC, self.fps, (self.width, self.height))

        if not out.isOpened():
            logger.error(f"[Cam {self.index}] Error: No se pudo crear el archivo de video '{full_path}' con FourCC {MP4_FOURCC}")
            logger.error("Asegúrate de que los codecs necesarios (e.g., ffmpeg para H.264) estén instalados y configurados para OpenCV.")
            return False

        try:
            for frame in frames_to_save:
                out.write(frame)
            logger.info(f"[Cam {self.index}] Guardado: '{full_path}'")
            return True
        except Exception as e:
            logger.error(f"[Cam {self.index}] Error al escribir frames en '{full_path}': {e}")
            return False
        finally:
            out.release() # Ensure the video writer is released

    def release(self):
        """
        Releases the camera capture object.
        """
        self.cap.release()
        logger.info(f"[Cam {self.index}] Cámara liberada.")