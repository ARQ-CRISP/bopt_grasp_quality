#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
import skopt
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from rospkg.rospack import RosPack
from skopt.acquisition import gaussian_ei
from skopt.plots import plot_convergence

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
        plt.title('Estimated Function')
        history_x = np.array(result.x_iters)
        history_y = np.array(result.func_vals)
        
        x = np.linspace(result.space.bounds[0][0], result.space.bounds[0][1], n_samples).reshape(-1, 1)
        x_gp = result.space.transform(x.tolist())
        gp = result.models[-1]
        y_pred, sigma = gp.predict(x_gp, return_std=True)
        plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
        plt.fill(
            np.concatenate([x, x[::-1]]), 
            np.concatenate([y_pred - conf95 * sigma, (y_pred + conf95 * sigma)[::-1]]), 
            alpha=.2, fc="g", ec="None")
            # Plot sampled points
        plt.plot(history_x, history_y,
                "r.", markersize=8, label="Observations")
        plt.plot(result.x, result.fun, '.y', markersize=10, label='best value')
        plt.xlabel('X position')
        plt.legend()
        plt.draw()
        plt.pause(0.1)

def plot_history(res, iters, n_samples=400):
        x = np.linspace(res.space.bounds[0][0], res.space.bounds[0][1], n_samples).reshape(-1, 1)
        x_gp = res.space.transform(x.tolist())
        # fx = np.array([f(x_i, noise_level=0.0) for x_i in x])
        conf95 = 1.96
        # result = result
        max_iters = len(res.models)
        illegal_iters = filter(lambda x: x < 0 or x >= max_iters, iters)
        iters = filter(lambda x: x >= 0 and x < max_iters, iters)
        print(2.8 * len(iters))
        fig = plt.figure(figsize=(8, 2.8 * len(iters)))
        plt.suptitle('Iteration History')
        plt.ion()
        print('WARNING: iterations {} not existing'.format(illegal_iters))
        for idx, n_iter in enumerate(iters):
                     
                gp = res.models[n_iter]
                plt.subplot(len(iters), 2, 2*idx+1)
                plt.title('Iteration {:d}'.format(n_iter))       
                curr_x_iters = res.x_iters[:min(max_iters, n_iter+1)]
                curr_func_vals = res.func_vals[:min(max_iters, n_iter+1)]

                y_pred, sigma = gp.predict(x_gp, return_std=True)
                plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
                plt.fill(np.concatenate([x, x[::-1]]),
                        np.concatenate([y_pred - conf95 * sigma,
                                        (y_pred + conf95 * sigma)[::-1]]),
                        alpha=.2, fc="g", ec="None")

                # Plot sampled points
                plt.plot(curr_x_iters, curr_func_vals,
                        "r.", markersize=8, label="Observations")

                # Adjust plot layout
                plt.grid()

                if n_iter + 1 == max_iters:
                        plt.plot(res.x, res.fun, 'Xc', markersize=14, label='Best value')

                if idx == len(iters)-1:
                        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
                        # plt.legend(loc="best", prop={'size': 6*4/len(iters)}, numpoints=1)

                if idx + 1 != len(iters):
                        plt.tick_params(axis='x', which='both', bottom='off',
                                        top='off', labelbottom='off')

                # Plot EI(x)
                plt.subplot(len(iters), 2, 2*idx+2)
                acq = gaussian_ei(x_gp, gp, y_opt=np.min(curr_func_vals))
                plt.plot(x, acq, "b", label="EI(x)")
                plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

                next_x = res.x_iters[min(max_iters, n_iter+1)]
                next_acq = gaussian_ei(res.space.transform([next_x]), gp,
                                        y_opt=np.min(curr_func_vals))
                plt.plot(next_x, next_acq, "bo", markersize=6, label="Next query point")

                # Adjust plot layout
                plt.ylim(0, 0.1)
                plt.grid()

                if idx == len(iters) -1:
                        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
                        # plt.legend(loc="best", prop={'size': 6*4/len(iters)}, numpoints=1)

                if idx + 1 != len(iters):
                        plt.tick_params(axis='x', which='both', bottom='off',
                                        top='off', labelbottom='off')

        
        
        plt.show()

def show_convergence(res):
        fig = plt.figure()
        plot_convergence(res)
        plt.ion()
        plt.draw()
        plt.pause(.1)


if __name__ == "__main__":
    

    res = skopt.load(pkg_path + '/etc/' + 'BayesOpt.pkl')

    print(res.x)