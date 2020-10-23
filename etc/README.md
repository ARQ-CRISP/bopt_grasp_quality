# Checkpoint File Folder

This is were we hold the checkpoint files of the experiments.

The files generated from the Bayesian optimization must be opened by using `skopt.load`.
The files generated with the random exploration, instead, can be opened with `pickle.load`

Only Real Issue is that those file require `python2.7` to be executed for compatibility issues.
In the `scripts` folder there is some utility function for plotting the Gaussian Processes in the 1D case.

Good Luck,
Claudio Coppola