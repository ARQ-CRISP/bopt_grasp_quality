#!/bin/python
try:
    from bayesopt_quality import BayesOpt_BO
except Exception as e:
    print('WARNING: BayesOpt library may not be installed')
    pass
try:
    from skopt_quality import Skopt_BO
except Exception as e:
    print('WARNING: scikit-opt library may not be properly installed.')
    pass

from random_discrete_quality import Random_Explorer

__all__ = ['BayesOpt_BO', 'Skopt_BO', 'Random_Explorer']