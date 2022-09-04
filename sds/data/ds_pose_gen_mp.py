import multiprocessing as mpr
import queue
import sys
import time
from enum import Enum
import multiprocessing as mpr
from pathlib import Path
import threading as thr
from typing import Tuple, Dict, Optional, Any, List

import cv2
from imgaug import augmenters as iaa
import numpy as np

from sds.data.ds_pose_gen import DsPoseGen
from sds.data.utils import extract_pose, resize_imgs, DsPoseItem, ds_pose_item_to_numbers
from sds.synth.renderer import Renderer, OutputType
from sds.utils.common import IntOrTuple, int_to_tuple
from sds.utils.utils import canonical_cam_mat_from_img, gen_rot_vec, make_transform
from sds.utils.ds_utils import load_objs


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


def mp_run(q_data: mpr.Queue, q_cmd: mpr.Queue, objs: Dict, obj_glob_id: str, img_out_size: int,
           img_base_size: IntOrTuple, aug_enabled: bool):
    q_data.cancel_join_thread()
    ds_gen = DsPoseGen(objs, obj_glob_id, img_out_size, img_base_size, aug_enabled, hide_window=True)
    i = -1
    can_gen = True
    nums = None
    while True:
        if can_gen:
            i += 1
            item = ds_gen.gen_item()
            nums = ds_pose_item_to_numbers(item)
        can_gen = queue_push(q_data, nums)

        cmd, received = queue_pop(q_cmd)
        if received and cmd == 'stop':
            return


class DsPoseGenMp:
    def __init__(self, objs: Dict, obj_glob_id: str, img_out_size: int,
                 img_base_size: IntOrTuple, aug_enabled: bool, batch_size: int, n_workers: int):
        objs = {obj_glob_id: objs[obj_glob_id]}
        self.img_out_size = int_to_tuple(img_out_size)
        self.batch_size = batch_size
        self.q_data = mpr.Queue(batch_size * 3)
        procs: List[Tuple[mpr.Process, mpr.Queue]] = []
        for i in range(n_workers):
            q_cmd = mpr.Queue(1)
            proc = mpr.Process(target=mp_run, args=(self.q_data, q_cmd, objs, obj_glob_id, img_out_size, img_base_size, aug_enabled))
            procs.append((proc, q_cmd))
            proc.start()
        self.procs = procs
        self.buf = []
        self.batch_size = batch_size

    def get_batch(self):
        while len(self.buf) < self.batch_size:
            item, ret = queue_pop(self.q_data)
            if ret:
                self.buf.append(item)
        res, self.buf = self.buf[:self.batch_size], self.buf[self.batch_size:]
        return res

    def stop(self):
        for proc, q_cmd in self.procs:
            # print('Proc:', proc.name, 'stop')
            queue_push(q_cmd, 'stop')
        for proc, q_cmd in self.procs:
            # print('Proc:', proc.name, 'wait')
            proc.join()
            q_cmd.close()
        self.q_data.close()


def _test_ds_pose_gen_mp():
    ds_name = 'itodd'
    ds_path = Path('/ws/data/sds') / ds_name
    objs = load_objs(ds_path.parent, ds_name, load_meshes=True)
    # img_size = 128
    img_size = 400
    obj_num = 1
    num_to_obj_id = {obj['id_num']: obj_id for obj_id, obj in objs.items()}
    print(num_to_obj_id)
    obj_id = num_to_obj_id[obj_num]
    dsgen = DsPoseGenMp(objs, obj_id, img_size, (1280, 1024), True, 500, 50)
    while True:
        t = time.time()
        batch = dsgen.get_batch()
        print(f'Batch size: {len(batch)}. Time: {time.time() - t:.3f}')
        i = np.random.randint(0, len(batch))
        (img, params_in), target = batch[i]
        img_noc, img_norms = img[..., :3], img[..., 3:]
        cv2.imshow('img_noc', img_noc)
        cv2.imshow('img_norms', img_norms)
        if cv2.waitKey(1) in (27, ord('q')):
            break
    dsgen.stop()


if __name__ == '__main__':
    _test_ds_pose_gen_mp()

