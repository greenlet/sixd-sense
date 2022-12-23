# SixDSense model

## Prerequisites
### Packages
In a new Conda environment with Python 3.9 install packages:
1. Tensorflow
   ```sh
   conda install -c conda-forge cudatoolkit=11.2 cudatoolkit-dev=11.2 cudnn=8.1.0
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
   python -m pip install tensorflow
   # Verify install:
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Also follow instructions from the first answer to https://stackoverflow.com/questions/46826497/conda-set-ld-library-path-for-env-only

2. Packages for Pyopengl, GLFW
   ```sh
   sudo apt install freeglut3-dev
   ```
3. Pymesh
    ```sh
    sudo apt install libgmp-dev libmpfr-dev libboost-all-dev
    git clone --recurse-submodules -j6 https://github.com/PyMesh/PyMesh.git <pymesh-path>
    cd <pymesh-path> && python setup.py build
    python setup.py install
    ```
4. SDS
   ### Build Cython library (part of EfficeintNet code)
   ```sh
   python setup.py build_ext --inplace
   ```
5. All other packages
   ```sh
   pip install -r requirements.txt
   ```


### BlenderProc2
1. Clone BlenderProc2 and checkout to `main` branch:
    ```sh
#    git clone https://github.com/DLR-RM/BlenderProc.git
    git clone git@github.com:DLR-RM/BlenderProc.git
    cd BlenderProc && git checkout v2.5.0
    pip install -e .
    ```


Packages for Blender environment:
 - pydantic-yaml
 - plyfile
 - ilock

```sh
blenderproc pip install pydantic-yaml plyfile ilock
```

