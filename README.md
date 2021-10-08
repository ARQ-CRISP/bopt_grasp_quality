# bopt_grasp_quality
Bayesian Optimisation based Grasp Quality Estimation from Hand Pose

If you are using this Repo, Consider citing the follwing [article](https://www.frontiersin.org/articles/10.3389/frobt.2021.703869/full):

```
@article{siddiqui2021grasp,
  title={Grasp Stability Prediction for a Dexterous Robotic Hand Combining Depth Vision and Haptic Bayesian Exploration},
  author={Siddiqui, Muhammad Sami and Coppola, Claudio and Solak, Gokhan and Jamone, Lorenzo},
  journal={Frontiers in Robotics and AI},
  pages={237},
  year={2021},
  publisher={Frontiers}
}
```

## Dependencies


### Ubuntu 

```bash
sudo apt-get install libboost-dev cmake cmake-curses-gui g++
sudo apt-get install python-dev python-numpy
```
### Installation of SkoptOpt

In this package, a customised version of the scikit-opt library is used. This version includes Unscented Bayesian Optimization.
to install it use the following command

`pip install git+https://github.com/Raziel90/scikit-optimize.git`

The edits have been submitted in a pull request, when those are accepted the official repo can be used.