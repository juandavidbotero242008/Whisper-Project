from queue import Queue
from audio_capture import audio_capture
from audio_procesor import audio_procesor

def orquestator(queue: Queue, model_vad):

    while(True):
        for data in audio_capture():
            audio_procesor(data, queue, model_vad)

