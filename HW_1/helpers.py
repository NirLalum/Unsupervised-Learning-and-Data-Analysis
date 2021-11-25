import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
from scipy.stats import norm

# plot estimated F and CI using DKW
def calc_plot_CI(alfa, n, plot=False, ax=None):
    eps = eps = np.sqrt(np.log(2/alfa)/(2*n)) # this the epsilon required for 95% CI according to DKW inequallity
    x_sample = np.random.randn(n)
    H, x_est = np.histogram(x_sample, np.arange(-3,3+0.002,0.001))
    F_n = (1/n)*(np.cumsum(H)) # equals to 1/N*(I{Xi<=x}) which is the estimator of F(x)
    F_ci_up =  F_n + eps
    F_ci_down =  F_n - eps
    
    if plot:
        ax.plot(x_est[1:], F_n, color='deeppink',label='Empirical cdf (95% CI)')
        ax.plot(x_est[1:], F_ci_up, color='pink')
        ax.plot(x_est[1:], F_ci_down, color='pink')
        plt.fill_between(x_est[1:],F_ci_up, F_ci_down, color='lightpink')
        ax.legend()
    
    return F_n, F_ci_down, F_ci_up, x_est


# compute the emperical correlation and std of features
def corr_bootstrap(x:np.ndarray, k, m):
    # first calculate emperical correlation
    emp_corr = np.corrcoef(x, rowvar=False)[0,1] # the diagonal is the variance and the values outside the diagonal are the correlation
    
    # calculate the std of the emperical correlation using bootstrap
    corr_list = []
    for _ in range(m):
        # sample k with replacement
        rows = np.random.choice(x.shape[0], size=k, replace=True)
        xk = x[rows, :]

        # calculate correlation of the sample
        corr_list.append(np.corrcoef(xk, rowvar=False)[0,1])
    
    return emp_corr, np.std(corr_list), corr_list



    


