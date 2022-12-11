import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from sds.utils.utils import calc_3d_pts_uniform, graph_from_sphere_pts, calc_3d_graph_elevated


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
    t = time.time()
    edges = graph_from_sphere_pts(pts)
    print(f'Calculated {len(edges)} edges from {num_pts} points for {time.time() - t:.3f} sec.')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # o3d.visualization.draw_geometries([pcd])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    o3d.visualization.draw_geometries([pcd, line_set])


def show_3d_graph(pts: np.ndarray, n_levels: int, edges: np.ndarray):
    n_pts = len(pts)
    expand_coeff = np.linspace(1, 1.5, n_levels)
    pts = np.tile(pts, (n_levels, 1)) * np.repeat(expand_coeff, n_pts)[..., None]

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
    # calc_3d_graph(10000)

    t1 = time.time()
    # n_pts, n_levels = 10000, 360
    n_pts, n_levels = 100, 4
    pts, n_verts, edges = calc_3d_graph_elevated(n_pts, n_levels)
    t2 = time.time()
    n_edges = len(edges)
    print(f'3d graph vertices: {n_verts}, edges: {n_edges}, time: {t2 - t1:.3f}')
    if n_verts <= 1000:
        show_3d_graph(pts, n_levels, edges)


if __name__ == '__main__':
    main()

