# examples/Python/Basic/pointcloud.py

import numpy as np
import open3d as o3d

from sds.utils.utils import gen_rot_vec

if __name__ == "__main__":

    n = 20000
    pts = np.zeros((n, 3), np.float64)
    for i in range(n):
        rv, _ = gen_rot_vec()
        pts[i] = rv

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd])


