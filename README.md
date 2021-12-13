# 2021_SuperMeshing_2D_Plain_Strain

This is the implementation of the SuperMeshingNet, for paper: SuperMehsingNet: A Novel Method for Boosting the Mesh Density in Numerical Computation within 2D Domain

The architecture of the SuperMeshingNet is shown in figure.

![SuperMeshingNet architecture](https://s2.loli.net/2021/12/13/mUZhL4EgJqHPy9z.png)

To train the model, please download the project and run:

python code/train.py

To evaluate the model, please run:

python code/test.py

If you want to test different scaling factors, please change the --scale argument in /code/train.py and /code/test.py, replacing 4 as 2 or 8.

The result floder includes the reslt of experiment.

![test result](https://s2.loli.net/2021/12/13/f4AuyX5B7OS1li8.png =400)
