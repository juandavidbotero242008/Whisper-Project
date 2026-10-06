import logging
import soundcard as sc
import numpy as np
import numpy as np
from numpy.typing import NDArray
from queue import Queue
import torch
from silero_vad import get_speech_timestamps

logger = logging.getLogger(__name__)

def audio_procesor(data: NDArray[np.float32], queue: Queue, model: torch.nn.Module):

    logger.debug("Procesando bloque de %d muestras", len(data))

    try:
        audio = torch.from_numpy(data).float()

        print("[TENSOR]: Audio apuntando de Numpy a Tensor")

        sensibilidad = 0.2

        print(f"[SENSIBILIDAD]: Intentando con sensibilidad: {sensibilidad}")

        timestamps = get_speech_timestamps(audio, model, sampling_rate=16000, threshold=0.2)

        print(f"[TIMESTAMP]: Timestamp conseguido: {timestamps} \n")
        logger.debug("VAD detectó %d segmento(s) de voz", len(timestamps))
    except Exception:
        logger.exception("Error al convertir el audio o ejecutar el VAD")
        raise

    try:
        bloque = []

        print("[BLOQUE]: Bloque vacio creado")

        print("Momento antes del bucle \n")

        for ts in timestamps:

            print("[BUCLE: Vuelta al bucle]")

            inicio = ts['start']
            final = ts['end']

            corte = data[inicio : final]
            print("[SLICING]: Arreglo Cortado \n")

            bloque.append(corte)
            print("[BLOQUE]: Segmento añadido al bloque \n")

        if len(bloque)>0:

            print("[BLOQUE]: Bloque contiene segmentos\n")

            audioLimpio = np.concatenate(bloque)

            print("[CONCATENAR]: Uniendo segmentos\n")

            print("[COLA]: Enviando a cola\n")

            queue.put(audioLimpio)
            logger.info("Audio con voz enviado a la cola (%d muestras, tamaño de cola: %d)", len(audioLimpio), queue.qsize())
        else:
            logger.debug("No se detectó voz en el bloque; no se envía a la cola")
    except Exception:
        logger.exception("Error al recortar, unir o encolar los segmentos de voz")
        raise