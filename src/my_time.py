import time

start_time = 0

def start():
    global start_time
    start_time = time.time()

def stop():
    return time.time() - start_time

def stop_print():
    print("--- %s seconds ---" % (time.time() - start_time))
