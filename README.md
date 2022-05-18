# SixDSense model

## Prerequisites
### Packages
In a new Conda environment with Python 3.9 install packages:
1. Tensorflow
   ```sh
#   conda install -c conda-forge tensorflow-gpu
#   conda install -c esri tensorflow-addons
   pip install -r requirements.txt
   ```
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

### BlenderProc2
1. Clone BlenderProc2 and checkout to `main` branch:
    ```sh
    git clone https://github.com/DLR-RM/BlenderProc.git
    cd BlenderProc
    pip install -e .
    ```


Packages for Blender environment:
 - pydantic-yaml
 - plyfile

```sh
blenderproc pip install ...
```

