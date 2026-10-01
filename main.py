from queue import Queue
from capturador import capturar
from traductor import traducir
import threading
import time
import os
import sys
import site



def main():

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