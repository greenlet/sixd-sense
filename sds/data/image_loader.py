from pathlib import Path
import cv2
import numpy as np


class ImageLoader:
    def __init__(self, images_dir_path: Path):
        self.dir_path = images_dir_path
        self.file_paths = list(p for p in self.dir_path.iterdir() if p.is_file())

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, i: int) -> np.ndarray:
        img = cv2.imread(self.file_paths[i].as_posix())
        return img

