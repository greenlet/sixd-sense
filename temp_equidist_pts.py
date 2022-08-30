from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree


def calc2d(num_pts: int):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    r = np.sqrt(indices / num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y = r * np.cos(theta), r * np.sin(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.set_aspect('equal')
    plt.show()


def calc_3d_pts_uniform(num_pts: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return x, y, z


def graph_from_sphere_pts(pts: np.ndarray) -> List[Tuple[int, int]]:
    n = len(pts)
    r = np.sqrt(4 * np.pi / n) * 1.5
    kdt = KDTree(pts)
    # edges = set()
    edges = []
    for i, pt in enumerate(pts):
        inds = kdt.query_ball_point(pt, r)
        for ind in inds:
            # edges.add((i, ind))
            edges.append((i, ind))
    # edges = list(edges)
    edges = list(set(edges))
    return edges


def calc3d(num_pts: int):
    x, y, z = calc_3d_pts_uniform(num_pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    # ax.set_aspect('equal')
    plt.show()


def calc_3d_graph(num_pts: int):
    x, y, z = calc_3d_pts_uniform(num_pts)
    pts = np.stack([x, y, z], axis=1)
    edges = graph_from_sphere_pts(pts)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # o3d.visualization.draw_geometries([pcd])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    o3d.visualization.draw_geometries([pcd, line_set])


def main():
    # calc2d(1000)
    # calc3d(1000)
    calc_3d_graph(1000)


if __name__ == '__main__':
    main()

