#!/bin/python
# try: LEGACY CODE
try:
    from skopt_quality import Skopt_BO
    from unscented_skopt_quality import Skopt_UBO
except Exception as e:
    print('WARNING: scikit-opt library may not be properly installed.')
    pass

from random_discrete_quality import Random_Explorer

__all__ = ['Skopt_BO', 'Skopt_UBO', 'Random_Explorer']