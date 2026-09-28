import soundcard as sc
import numpy as np
from queue import Queue

def capturar(queue: Queue):
    # Obtiene el altavoz principal del sistema
    speaker = sc.default_speaker()

    # Activa el modo loopback (escuchar lo que sale por el altavoz)
    mic = sc.get_microphone(id=speaker.id, include_loopback=True)

    # Definir parámetros
    SAMPLE_RATE = 16000  # Frecuencia ideal para Whisper (16 kHz)
    NUM_FRAMES = 80000   # Cantidad de muestras a capturar (16000 = 1 segundo de audio)

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        # Graba un bloque de audio de la duración definida por numframes
        while True:
            data = recorder.record(numframes=NUM_FRAMES)
            data = data.flatten()
            queue.put(data)
             
