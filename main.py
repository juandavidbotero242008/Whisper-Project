from queue import Queue
from capturador import capturar
from traductor import traducir
import threading
import time
import os
import sys
import site



def main():
    site_packages = site.getsitepackages()[0]

    # Construye la ruta hacia las DLLs de cuBLAS y cuDNN
    cublas_path = os.path.join(site_packages, "nvidia", "cuBLAS", "bin") # Ten en cuenta las mayúsculas "cuBLAS"
    cudnn_path = os.path.join(site_packages, "nvidia", "cudnn", "bin")

    if os.path.exists(cublas_path):
        os.add_dll_directory(cublas_path)
    if os.path.exists(cudnn_path):
        os.add_dll_directory(cudnn_path)
    queue = Queue()

    thread1 = threading.Thread(target=capturar, args=(queue,))
    thread2 = threading.Thread(target=traducir, args=(queue,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print("Hilos terminados")

    pass

if __name__ == "__main__":
    main()