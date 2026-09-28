from faster_whisper import WhisperModel
from queue import Queue

def traducir(queue: Queue):
    model_size = "tiny"
    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16") 
    while True:
        segments, info =  model.transcribe(queue.get(), language="es", vad_filter=True)
        for segment in segments:
            print(segment.text)

    
   

