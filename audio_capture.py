import logging
import soundcard as sc
import numpy as np
import torch
import numpy as np
from queue import Queue

logger = logging.getLogger(__name__)


def audio_capture():
    try:
        # Obtiene el altavoz principal del sistema
        speaker = sc.default_speaker()
        logger.info("Altavoz por defecto: %s", speaker.name)

        # Activa el modo loopback (escuchar lo que sale por el altavoz)
        mic = sc.get_microphone(id=speaker.id, include_loopback=True)
        logger.info("Dispositivo loopback obtenido: %s", mic.name)

        # Definir parámetros
        SAMPLE_RATE = 16000  # Frecuencia ideal para Whisper (16 kHz)
        NUM_FRAMES = 80000   # Cantidad de muestras a capturar (16000 = 1 segundo de audio)

        with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
            logger.info("Grabación iniciada (samplerate=%s, numframes=%s)", SAMPLE_RATE, NUM_FRAMES)
            # Graba un bloque de audio de la duración definida por numframes
            while True:

                data = recorder.record(numframes=NUM_FRAMES)


                print("[DATA]: Datos de audio puros recolectados\n")

                print(f"Dimension: {data.shape} \n")
                print(f"Max: {data.max()} \n")
                print(f"Min: {data.min()} \n")


                data = data.flatten()
                logger.debug("Bloque capturado: %d muestras", data.size)

                yield data
    except Exception:
        logger.exception("Error en la captura de audio")
        raise
    finally:
        logger.info("Captura de audio finalizada")