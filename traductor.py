import logging
import time
from faster_whisper import WhisperModel
from queue import Queue

logger = logging.getLogger(__name__)

def traducir(queue: Queue, model: WhisperModel):
    logger.info("Traductor iniciado; esperando audio en la cola")

    try:
        while True:
            try:
                inicio = time.perf_counter()
                segments, info =  model.transcribe(queue.get(), language="es")
                logger.debug("Audio recibido de la cola (elementos pendientes: %d)", queue.qsize())
                print("Traduciendo \n")
                for segment in segments:
                    print(segment.text)
                    logger.debug("Segmento [%.2fs - %.2fs]: %s", segment.start, segment.end, segment.text)
                logger.info("Transcripción completada en %.2fs (elementos pendientes en cola: %d)", time.perf_counter() - inicio, queue.qsize())
            except Exception:
                logger.exception("Error al transcribir un audio; se omite y se continúa con el siguiente")
    except BaseException:
        logger.exception("El hilo del traductor terminó por un error inesperado")
        raise
    finally:
        logger.info("Traductor finalizado")