import time
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
    edges = set()
    for i, pt in enumerate(pts):
        inds = kdt.query_ball_point(pt, r)
        for ind in inds:
            if ind == i:
                continue
            e = (i, ind) if i < ind else (ind, i)
            edges.add(e)
    edges = list(edges)
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


def calc_3d_graph_elevated(n_pts: int, n_levels: int) -> Tuple[np.ndarray, int, np.ndarray]:
    x, y, z = calc_3d_pts_uniform(n_pts)
    pts = np.stack([x, y, z], axis=1)
    inds_pts, levels = np.arange(n_pts), np.arange(n_levels)
    edges_pts = graph_from_sphere_pts(pts)
    n_edges = len(edges_pts)

    edges_mul = np.tile(edges_pts, (n_levels, 1))
    levels_mul = np.repeat(levels * n_pts, n_edges)
    edges = edges_mul + levels_mul[..., None]
    edges_prev = edges_mul + np.stack([levels_mul, np.roll(levels_mul, -n_edges, axis=0)], axis=1)
    edges_next = edges_mul + np.stack([levels_mul, np.roll(levels_mul, n_edges, axis=0)], axis=1)

    n_verts = n_pts * n_levels
    inds_verts = np.arange(n_verts)
    edges_up = np.stack([inds_verts, np.roll(inds_verts, -n_pts)], axis=1)
    edges_down = np.stack([inds_verts, np.roll(inds_verts, n_pts)], axis=1)

    edges = np.concatenate([edges, edges_prev, edges_next, edges_up, edges_down])

    return pts, n_verts, edges


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

