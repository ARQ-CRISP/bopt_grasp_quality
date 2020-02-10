# bopt_grasp_quality
Bayesian Optimisation based Grasp Quality Estimation from Hand Pose



## Dependencies

### Ubuntu 

```
sudo apt-get install libboost-dev cmake cmake-curses-gui g++
sudo apt-get install python-dev python-numpy
```

### Installation of BayesOpt

**Installation**:
```
git clone https://github.com/rmcantin/bayesopt && cd bayesop
cmake -DBAYESOPT_PYTHON_INTERFACE=ON . 
make
sudo make install
```
**Test the installation:**
```python ./python/demo_quad.py```

For more info: [bayesopt installation guide](https://rmcantin.bitbucket.io/html/install.html)