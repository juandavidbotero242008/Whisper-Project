import soundcard as sc
import numpy as np
import torch
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

    model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad"
        )
    
    (get_speech_timestamps, _, read_audio, _, _) = utils

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        # Graba un bloque de audio de la duración definida por numframes
        while True: 

            data = recorder.record(numframes=NUM_FRAMES)
            data = data.flatten()

            print("[DATA]: Datos de audio puros recolectados\n")

            print(f"Dimension: {data.shape} \n")
            print(f"Max: {data.max()} \n")
            print(f"Min: {data.min()} \n")
            

            data = data.flatten()

            
            audio = torch.from_numpy(data).float()

            

            print("[TENSOR]: Audio convertido de Numpy a Tensor")

            

            sensibilidad = 0.2

            print(f"[SENSIBILIDAD]: Intentando con sensibilidad: {sensibilidad}")

            timestamps = get_speech_timestamps(audio, model, sampling_rate=16000, threshold=0.2)

            print(f"[TIMESTAMP]: Timestamp conseguido: {timestamps} \n")
            
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

                print("[COLA]: Audio limpio en cola \n")

            else:
                pass



           
             
