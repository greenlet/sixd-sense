import multiprocessing as mpr
import os
import queue
import time
from typing import Tuple, Any


MAX_QUEUE_SIZE = 5


def queue_pop(q: mpr.Queue, timeout_sec: float = 0.1) -> Tuple[Any, bool]:
    try:
        res = q.get(True, timeout_sec)
        return res, True
    except queue.Empty:
        return None, False


def queue_push(q: mpr.Queue, item: Any, timeout_sec: float = 0.1) -> bool:
    try:
        q.put(item, True, timeout_sec)
        return True
    except queue.Full:
        return False


def worker_fn(name: str, q_data: mpr.Queue, q_cmd: mpr.JoinableQueue):
    print(f'Worker {name} starting')
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

    i = -1
    can_gen = True
    item = None
    while True:
        if can_gen:
            i += 1
            item = f'worker {name}. item {i}'
        can_gen = queue_push(q_data, item)

        cmd, received = queue_pop(q_cmd)
        if received and cmd == 'stop':
            break

    print(f'Worker {name} stopping')


def main():
    print('Main')
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

    ps = {}
    q_data = mpr.Queue(MAX_QUEUE_SIZE)
    for i in range(3):
        name = f'wr_{i}'
        q_cmd = mpr.JoinableQueue()
        print(f'Starting process {name} from the main process')
        p = mpr.Process(target=worker_fn, args=(name, q_data, q_cmd))
        p.start()
        ps[name] = {
            'q_cmd': q_cmd,
            'process': p,
        }

    time.sleep(3)
    for d in ps.values():
        assert queue_push(d['q_cmd'], 'stop')
        d['process'].join()
        d['q_cmd'].close()

    i = 0
    while i < MAX_QUEUE_SIZE:
        res, ret = queue_pop(q_data)
        if ret:
            print(res)
            i += 1

    q_data.close()
    print('Exiting main process')


if __name__ == '__main__':
    main()

