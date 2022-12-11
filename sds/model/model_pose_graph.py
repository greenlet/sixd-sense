from typing import Any, Tuple

import numpy as np
import tensorflow as tf

from sds.model.model_pose import build_layers_down
from sds.utils.utils import calc_3d_graph_elevated


# `maps_features`: N_components x Feature_dimension. Feature represents embedding extracted by
# backbone CNN from object's maps.
# `graph_pts`: N_points_in_one_level x 3. Contains set of points distributed on sphere which represents
# one level of vertices in the graph.
# `n_verts` = N_points_in_one_level * `n_levels`. Number of vertices in the graph.
# `n_levels` - number of descrete steps of interval: [0, pi), i.e. given h = pi / `n_levels`,
# levels (0, 1, ..., `n_levels` - 1) correspond to values (0, h, ..., (`n_levels` - 1) * h) respectively.
# `edges_dir`: N_components x N_directed_edges. Directed edges of graph with `n_verts` vertices.
# `batch_size` = N_components.
# `inp_pose_size` - Size of pose parameters vector needed to restore 3d position of object' center.
def build_head_manual(maps_features: Any, graph_pts: np.ndarray, n_verts: int, n_levels: int, edges_dir: np.ndarray, batch_size: int, inp_pose_size: int = 6,
        normalize: bool = True, add_self_loops: bool = True):
    inp_pose = tf.keras.Input((inp_pose_size,), batch_size=batch_size, dtype=tf.float32)
    n_pts = len(graph_pts)
    vert_pts = tf.convert_to_tensor(graph_pts, dtype=tf.float32)

    # Build adjacency matrix
    n_edges_dir = len(edges_dir)
    n_edges = n_edges_dir * 2
    edges_1 = tf.convert_to_tensor(edges_dir, dtype=tf.int64)
    edges_2 = tf.reverse(edges_1, axis=[1])
    edges = tf.concat([edges_1, edges_2], axis=0)
    edges = tf.tile(edges, (batch_size, 1))
    shift = tf.range(0, batch_size * n_edges, n_edges, dtype=tf.int64)
    shift = tf.repeat(shift, n_edges)[..., tf.newaxis]
    edges += shift

    A = tf.sparse.SparseTensor(edges, tf.ones(batch_size * n_edges, tf.float32), (batch_size * n_verts, batch_size * n_verts))
    A = tf.sparse.reorder(A)

    angle_step = np.pi / n_levels
    angles = tf.range(n_levels, dtype=tf.float32) * angle_step
    angles = tf.reshape(angles, (n_levels, 1))

    # batch_size x Feature_dimension --> batch_size * n_verts x Feature_dimension
    vert_feat_maps = tf.repeat(maps_features, n_verts, axis=0)

    # N_points_in_one_level x 3 --> n_verts x 3
    vert_feat_pts = tf.tile(vert_pts, (n_levels, 1))
    # n_verts x 3 --> batch_size * n_verts x 3
    vert_feat_pts = tf.tile(vert_feat_pts, (batch_size, 1))

    # n_levels x 1 --> n_verts x 1
    vert_feat_angle = tf.repeat(angles, n_pts, axis=0)
    # n_verts x 1 --> batch_size * n_verts x 1
    vert_feat_angle = tf.tile(vert_feat_angle, (batch_size, 1))

    # batch_size x inp_pose_size --> batch_size * n_verts x inp_pose_size
    vert_feat_pose = tf.repeat(inp_pose, n_verts, axis=0)

    # batch_size * n_verts x (Feature_dimension + 3 + 1 + inp_pose_size)
    vert_feat = tf.concat([vert_feat_maps, vert_feat_pts, vert_feat_angle, vert_feat_pose], axis=1)
    vert_feat = tf.convert_to_tensor(vert_feat)
    print(vert_feat.shape)

    x = vert_feat
    x = GcnLayer(A, batch_size, n_verts, 128, normalize, add_self_loops)(x)
    x = GcnLayer(A, batch_size, n_verts, 128, normalize, add_self_loops)(x)
    x = GcnLayer(A, batch_size, n_verts, 64, normalize, add_self_loops)(x)
    x = GcnLayer(A, batch_size, n_verts, 32, normalize, add_self_loops)(x)
    x = GcnLayer(A, batch_size, n_verts, 16, normalize, add_self_loops)(x)
    x_pos = x

    x = GcnLayer(A, batch_size, n_verts, 8, normalize, add_self_loops)(x)
    x = GcnLayer(A, batch_size, n_verts, 1, normalize, add_self_loops, 'sigmoid')(x)
    out_rot = x

    x = x_pos
    use_bias = True
    x = tf.keras.layers.Dense(16, 'relu', use_bias=use_bias)(x)
    x = tf.keras.layers.Dense(16, 'relu', use_bias=use_bias)(x)
    x = tf.reshape(x, (batch_size, n_verts, 16))
    x = tf.reduce_mean(x, axis=1)
    x = tf.keras.layers.Dense(16, 'relu', use_bias=use_bias)(x)
    x = tf.keras.layers.Dense(16, 'relu', use_bias=use_bias)(x)
    x = tf.keras.layers.Dense(3, use_bias=use_bias)(x)
    out_pos = x

    return inp_pose, out_pos, out_rot


def build_hybrid_layers(batch_size: int, inp_img_size: int = 256, inp_channels: int = 6, inp_pose_size: int = 6,
            n_graph_pts: int = 1000, n_graph_levels: int = 100) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:

    inp_maps, feat_maps, ch = build_layers_down(inp_img_size, inp_channels, batch_size)

    pts, n_verts, edges = calc_3d_graph_elevated(n_graph_pts, n_graph_levels)
    inp_pose, out_pos, out_rot = build_head_manual(feat_maps, pts, n_verts, n_graph_levels, edges, batch_size, inp_pose_size)

    print('Pos head:', out_pos)
    print('Rot head:', out_rot)

    return (inp_maps, inp_pose), (out_pos, out_rot)


class GcnLayer(tf.keras.layers.Layer):
    def __init__(self, A: tf.sparse.SparseTensor, batch_size: int, n_verts: int, units: int, normalize: bool = True, add_self_loops: bool = True,
            activation: Any = 'relu'):
        super().__init__()
        self.A = A
        self.batch_size = batch_size
        self.n_verts = n_verts
        self.units = units
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.activation = activation

    def call(self, features: Any, *args, **kwargs):
        normalized_values = features
        normalized_values = tf.reshape(normalized_values, (self.batch_size * self.n_verts, features.shape[-1]))
        invsqrt_deg = tf.constant([])
        if self.normalize:
            in_degree = tf.sparse.reduce_sum(self.A, axis=-1)

            if self.add_self_loops:
                in_degree += 1

            invsqrt_deg = tf.math.rsqrt(in_degree)
            normalized_values = invsqrt_deg[:, tf.newaxis] * features

        print(tf.shape(self.A), tf.shape(normalized_values))
        pooled = tf.sparse.sparse_dense_matmul(self.A, normalized_values)

        if self.add_self_loops:
            if self.normalize:
                pooled += invsqrt_deg[:, tf.newaxis] * normalized_values
            else:
                pooled += normalized_values

        # pooled = tf.reshape(pooled, (self.batch_size, self.n_verts, -1))
        out = tf.keras.layers.Dense(units=self.units, activation=self.activation, use_bias=False)(pooled)

        return out


def test_hybrid():
    inp, out = build_hybrid_layers(10)
    print(out)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    print(model.summary())


if __name__ == '__main__':
    test_hybrid()

