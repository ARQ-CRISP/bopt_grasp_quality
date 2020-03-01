#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
import skopt
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from rospkg.rospack import RosPack
from skopt.acquisition import gaussian_ei
# from skopt.plots import plot_convergence

matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# plt.rcParams["figure.figsize"] = (8, 14)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']


pack_name = 'bopt_grasp_quality'
pkg_path = RosPack().get_path(pack_name)


def plot_result1D(result, n_samples=400):
    conf95 = 1.96
    result = result
    fig = plt.figure()
    plt.ion()
    plt.title('Estimated Metric')
    history_x = np.array(result.x_iters)
    history_y = np.array(result.func_vals)

    x = np.linspace(
        result.space.bounds[0][0], result.space.bounds[0][1], n_samples).reshape(-1, 1)
    x_gp = result.space.transform(x.tolist())
    gp = result.models[-1]
    y_pred, sigma = gp.predict(x_gp, return_std=True)
    plt.plot(x*100, -y_pred, "g--", label=r"$\mu_{GP}(x)$")
    plt.fill(
        np.concatenate([x, x[::-1]])*100,
        np.concatenate(
            [-y_pred - conf95 * sigma, (-y_pred + conf95 * sigma)[::-1]]),
        alpha=.2, fc="g", ec="None")
    # Plot sampled points
    plt.plot(history_x * 100, -history_y,
             "r.", markersize=8, label="Observations")
    plt.plot(np.array(result.x)*100, -result.fun, '.y', markersize=10, label='best value')
    plt.xlabel('Hand position (cm)')
    plt.ylabel('Grasp Metric')
    plt.legend()
    plt.draw()
    plt.grid()
    plt.pause(0.1)


def plot_history1D(res, iters, n_samples=400):
    x = np.linspace(
        res.space.bounds[0][0], res.space.bounds[0][1], n_samples).reshape(-1, 1)
    x_gp = res.space.transform(x.tolist())
    # fx = np.array([f(x_i, noise_level=0.0) for x_i in x])
    conf95 = 1.96
    r_start = len(res.x_iters) - len(res.models)
    # result = result
    max_iters = len(res.x_iters) + 1
    illegal_iters = filter(lambda x: x < 0 or x >= len(res.models), iters)
    iters = filter(lambda x: x >= 0 and x < len(res.models), iters)
    # print(2.8 * len(iters))
    fig = plt.figure(figsize=(8, 2.8 * len(iters)))
    plt.suptitle('Iteration History')
    plt.ion()
    print('WARNING: iterations {} not existing'.format(illegal_iters))
    for idx, n_iter in enumerate(iters):

        gp = res.models[n_iter]
        plt.subplot(len(iters), 2, 2*idx+1)
        plt.title('Iteration {:d}'.format(n_iter))
        curr_x_iters = res.x_iters[:min(max_iters, r_start + n_iter+1)]
        curr_func_vals = res.func_vals[:min(max_iters, r_start + n_iter+1)]

        y_pred, sigma = gp.predict(x_gp, return_std=True)
        plt.plot(x * 100, -y_pred, "g--", label=r"$\mu_{GP}(x)$")
        plt.fill(np.concatenate(np.array([x, x[::-1]]) * 100),
                 -np.concatenate([y_pred - conf95 * sigma,
                                 (y_pred + conf95 * sigma)[::-1]]),
                 alpha=.2, fc="g", ec="None")

        # Plot sampled points
        plt.plot(np.array(curr_x_iters) * 100, -np.array(curr_func_vals),
                 "r.", markersize=8, label="Observations")

        # Adjust plot layout
        plt.grid()

        if n_iter + 1 == len(res.models):
            plt.plot(np.array(res.x) * 100, -res.fun, 'Xc', markersize=14, label='Best value')

        if idx == len(iters)-1:
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
            # plt.legend(loc="best", prop={'size': 6*4/len(iters)}, numpoints=1)
            plt.xlabel('Hand Position (cm)')

        if idx + 1 != len(iters):
            plt.tick_params(axis='x', which='both', bottom='off',
                            top='off', labelbottom='off')

        # Plot EI(x)
        plt.subplot(len(iters), 2, 2*idx+2)
        acq = gaussian_ei(x_gp, gp, y_opt=np.min(curr_func_vals))
        plt.plot(x*100, acq, "b", label="EI(x)")
        plt.fill_between(x.ravel() *100, -2.0, acq.ravel(), alpha=0.3, color='blue')

        if r_start + n_iter + 1 < max_iters:
            next_x = (res.x_iters + [res.x])[min(max_iters, r_start + n_iter + 1)]
            next_acq = gaussian_ei(res.space.transform([next_x]), gp,
                                y_opt=np.min(curr_func_vals))
            plt.plot(np.array(next_x) * 100, next_acq, "bo", markersize=6,
                    label="Next query point")

        # Adjust plot layout
        plt.ylim(0, 1.1)
        plt.grid()

        if idx == len(iters) - 1:
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
            # plt.legend(loc="best", prop={'size': 6*4/len(iters)}, numpoints=1)

        if idx + 1 != len(iters):
            plt.tick_params(axis='x', which='both', bottom='off',
                            top='off', labelbottom='off')

    plt.show()


def plot_convergence1D(*args, **kwargs):
    from scipy.optimize import OptimizeResult
    """Plot one or several convergence traces.      
        Parameters
        ----------
        args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
                The result(s) for which to plot the convergence trace.

                - if `OptimizeResult`, then draw the corresponding single trace;
                - if list of `OptimizeResult`, then draw the corresponding convergence
                traces in transparency, along with the average convergence trace;
                - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
                an `OptimizeResult` or a list of `OptimizeResult`.

        ax : `Axes`, optional
                The matplotlib axes on which to draw the plot, or `None` to create
                a new one.

        true_minimum : float, optional
                The true minimum value of the function, if known.

        yscale : None or string, optional
                The scale for the y-axis.

        Returns
        -------
        ax : `Axes`
                The matplotlib axes.
        """
    fig = plt.figure()
    plt.ion()
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = plt.cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")
    plt.draw()
    plt.pause(.1)
    return ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='show_result', description="Plots the result of 1D BayesOpt Experiments available types ['history', 'result', 'convergence']")
    parser.add_argument('-f', action='store', dest='file', default='BayesOpt.pkl')
    parser.add_argument('-t', action='store', dest='type', type=str, default='result')
    parser.add_argument('-i', action='store', dest='iters', type=int, default=[0, 1, 2, 3, 4], nargs='+')

    parse_res = parser.parse_args()
    fun_type = {
        'history': lambda res: plot_history1D(res, parse_res.iters),
        'result': lambda res: plot_result1D(res),
        'convergence': lambda res: plot_convergence1D(res)
        }
    
    splitted = parse_res.file.split('/')
    if len(splitted) == 1:
        saved_model = pkg_path + '/etc/' + parse_res.file
    else:
        saved_model = parse_res.file
    print('Loading file: {}'.format(saved_model))
    res = skopt.load(saved_model)
    if parse_res.type in fun_type.keys():
        print('Plot type: {}'.format(parse_res.type))
        fun_type[parse_res.type](res)
    else:
        print('[ERROR] requested plot does not exist!')
    


    print('Minima found in: {:.3f}, {:.3f}'.format(res.x[0], res.fun))
    end = raw_input('Press key to terminate >> ')
