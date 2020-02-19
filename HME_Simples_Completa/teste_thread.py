import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)
    return 1

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    total = 0
    for i in range(4):
        logging.info("Main    : before creating thread")
        x = threading.Thread(target=thread_function, args=(i,))
        logging.info("Main    : before running thread")
        total = x.start() + 1
        logging.info("Main    : wait for the thread to finish")
        # x.join()
        logging.info("Main    : all done")
    
    cont = 0
    while(total <= 4):
        cont = cont + 1
    
    print("Contador: %s",)
