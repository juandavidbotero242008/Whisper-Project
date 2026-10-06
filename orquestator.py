import logging
from queue import Queue
from audio_capture import audio_capture
from audio_procesor import audio_procesor

logger = logging.getLogger(__name__)

def orquestator(queue: Queue, model_vad):

    logger.info("Orquestador iniciado")

    try:
        while(True):
            for data in audio_capture():
                logger.debug("Bloque recibido de la captura (%d muestras)", data.size)
                try:
                    audio_procesor(data, queue, model_vad)
                except Exception:
                    # El detalle del error ya lo registra audio_procesor
                    logger.error("Falló el procesamiento de un bloque; se omite y se continúa con el siguiente")
    except Exception as e:
        # El detalle del error ya lo registra audio_capture
        logger.error("La captura de audio falló; el orquestador se detiene: %s", e)
    finally:
        logger.info("Orquestador finalizado")