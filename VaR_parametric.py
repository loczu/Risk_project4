from scipy.stats import genextreme
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vartests

def VaR_GEV(quantile, c=-0.1, loc=0, scale = 1):
    """return VaR form GEV distribution with c param"""

    return genextreme.ppf(quantile, c, loc, scale) * genextreme.std(c) +genextreme.mean(c)

VaR_95 = VaR_GEV(0.05)
VaR_99 = VaR_GEV(0.01)

def plot_VaR(c=-0.1, size=1000):
    """Return plot of distribution and VaR"""

    data = genextreme.rvs(c, size=size)
    x = np.linspace(genextreme.ppf(0.01, c),genextreme.ppf(0.99, c), 100)

    fig, ax = plt.subplots(1, 1)
    ax.hist(data, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.set_xlim([np.min([VaR_99,x[0]])-0.5, x[-1]])
    plt.axvline(x = VaR_95, color = 'b', label = 'VaR 95')
    plt.axvline(x = VaR_99, color = 'r', label = 'VaR 99')
    ax.legend(loc='best', frameon=False)

    plt.show()

print(plot_VaR())