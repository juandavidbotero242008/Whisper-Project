import os
import sys
import logging

# Nivel de logs: cambia a logging.DEBUG para ver el detalle completo
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Agregar las rutas de las DLLs de NVIDIA al PATH de Windows
venv_base = sys.prefix
nvidia_bin_path = os.path.join(venv_base, "Lib", "site-packages", "nvidia")

if os.path.exists(nvidia_bin_path):
    logger.info("Carpeta NVIDIA encontrada: %s", nvidia_bin_path)
    for root, dirs, files in os.walk(nvidia_bin_path):
        if "bin" in dirs:
            try:
                os.add_dll_directory(os.path.join(root, "bin"))
                os.environ["PATH"] += os.path.pathsep + os.path.join(root, "bin")
                logger.debug("DLL agregada al PATH: %s", os.path.join(root, "bin"))
            except Exception:
                logger.exception("No se pudo agregar la ruta de DLL: %s", os.path.join(root, "bin"))
else:
    logger.warning("No existe la carpeta NVIDIA en %s; no se agregaron DLLs al PATH", nvidia_bin_path)


from queue import Queue
import threading
import torch
from faster_whisper import WhisperModel
from orquestator import orquestator
from traductor import traducir



def main():
    queue = Queue()

    print("[MAIN]: Cargando modelos en GPU...")
    logger.info("Cargando modelos...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Dispositivo seleccionado: %s", device)
    if device == "cpu":
        logger.warning("CUDA no disponible: los modelos correrán en CPU")

    try:
        model_vad, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad"
        )
        model_vad.to(device)
        logger.info("Modelo VAD (Silero) cargado en %s", device)
    except Exception:
        logger.exception("No se pudo cargar el modelo VAD")
        raise

    try:
        model_whisper = WhisperModel("small", device=device, compute_type="int8")
        logger.info("Modelo Whisper cargado en %s", device)
    except Exception:
        logger.exception("No se pudo cargar el modelo Whisper")
        raise

    thread1 = threading.Thread(target=orquestator, args=(queue, model_vad), name="orquestador")
    thread2 = threading.Thread(target=traducir, args=(queue, model_whisper), name="traductor")

    try:
        thread1.start()
        logger.info("Hilo '%s' iniciado", thread1.name)
        thread2.start()
        logger.info("Hilo '%s' iniciado", thread2.name)

        thread1.join()
        thread2.join()
    except KeyboardInterrupt:
        logger.warning("Interrupción por teclado (Ctrl+C) recibida")
        raise
    except Exception:
        logger.exception("Error inesperado al ejecutar los hilos")
        raise

    print("Hilos terminados")
    logger.info("Hilos terminados")

    pass

if __name__ == "__main__":
    main()