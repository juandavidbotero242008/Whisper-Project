import os
import sys

# Agregar las rutas de las DLLs de NVIDIA al PATH de Windows
venv_base = sys.prefix
nvidia_bin_path = os.path.join(venv_base, "Lib", "site-packages", "nvidia")

if os.path.exists(nvidia_bin_path):
    for root, dirs, files in os.walk(nvidia_bin_path):
        if "bin" in dirs:
            os.add_dll_directory(os.path.join(root, "bin"))
            os.environ["PATH"] += os.path.pathsep + os.path.join(root, "bin")



from faster_whisper import WhisperModel
from queue import Queue

def traducir(queue: Queue):
    model_size = "small"
    model = WhisperModel(model_size, device="CUDA", compute_type="float16") 
    while True:
        segments, info =  model.transcribe(queue.get(), language="es")
        print("Traduciendo \n")
        for segment in segments:
            print(segment.text)

    
   

