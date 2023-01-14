import multiprocessing
import time
import psutil


def cpu_load():
    """Function to simulate CPU load"""
    while True:
        pass


if __name__ == '__main__':
    # Start 4 processes that will run the cpu_load function
    processes = [multiprocessing.Process(target=cpu_load) for i in range(4)]
    [p.start() for p in processes]

    # Monitor server's memory and disk usage during the load test
    while True:
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        print(f'Memory usage: {memory_usage}%, Disk usage: {disk_usage}%')
        time.sleep(1)
