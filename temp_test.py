import json
import sys
import time
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Dict, List, Any, Tuple

import cv2
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLUT import *
from OpenGL.GLU import *
import h5py
import yaml

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import pymesh
from scipy.spatial.transform import Rotation as R

vertex_shader = shaders.compileShader("""
#version 300 es
const float M_PI = 3.1415926535897932384626433832795;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
uniform mat4 obj_cam_mat;
uniform mat4 proj_mat;
uniform bool draw_normals;
uniform float max_dist_to_center;

out vec4 color_out;

void main()
{
    gl_Position = proj_mat * obj_cam_mat * vec4(position, 1.0);
    mat3 rot = mat3(obj_cam_mat);
    vec3 col;

    if (draw_normals) {
        vec3 norm_pos = rot * normal;
        col = norm_pos / 2.0 + 0.5;
    } else {
        vec3 cam_pos = rot * position;
        col = -cam_pos / max_dist_to_center / 2.0 + 0.5;
    }

    color_out = vec4(col, 1);
}
""", GL_VERTEX_SHADER)

geometry_shader = shaders.compileShader("""
#version 300 es
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform bool draw_normals;
in vec4 color_out[3];
out vec4 v_color;

void main()
{
    if (draw_normals) {
        vec3 t1 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;        
        vec3 t2 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
        vec3 n = normalize(cross(t1, t2));
        if (n.z < 0.0) {
            n = -n;
        }
        v_color = vec4(n, 1.0);
        for (int i = 0; i < 3; i++) {
            gl_Position = gl_in[i].gl_Position;
            EmitVertex();
        }
        EndPrimitive();
    } else {
        for (int i = 0; i < 3; i++) {
            gl_Position = gl_in[i].gl_Position;
            v_color = color_out[i];
            EmitVertex();
        }
        EndPrimitive();
    }
}
""", GL_GEOMETRY_SHADER)

fragment_shader = shaders.compileShader("""
#version 300 es
precision highp float;
in vec4 v_color;
out vec4 outputColor;
void main()
{
    outputColor = v_color;
}
""", GL_FRAGMENT_SHADER)
program = shaders.compileProgram(vertex_shader, geometry_shader, fragment_shader)
