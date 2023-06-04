import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vartests
import pandas as pd
from fitter import Fitter

def straty(table):
    strata = []
    for i in range(1, len(table)):
        x = table[i-1]-table[i]
        if x < 0:
            strata.append(-x)
        
    return strata

def przyrosty(table):
    przyrost = []
    for i in range(1, len(table)):
        x = table[i-1]-table[i]
        if x > 0:
            przyrost.append(x)
        
    return przyrost

def zwroty(table):
    zwrot = []
    for i in range(1, len(table)):
        zwrot.append(np.log(table[i]/table[i-1]) * 100)
        
    return zwrot

data = pd.read_csv("danekawa.csv")
df = pd.DataFrame(data)
cena = df['data']
cena = round((1/0.00045359237) * cena, 2)
strata = straty(cena)
zwrot = zwroty(cena)
przyrost = przyrosty(cena)

f = Fitter(zwrot,
           distributions=['cauchy', 'chi2', 'gamma', 'lognorm', 'norm', 'expon', 't'])
f.fit()
f.summary()

def VaR_t(quantile, df=8.383690833071302, loc=0.004378205763668096, scale = 1.840771049586619):
    """return VaR form GEV distribution with c param"""

    return ss.t.ppf(quantile, df, loc, scale) * ss.t.std(df) + ss.t.mean(df)

VaR_95 = VaR_t(0.05, df = 8.383690833071302, loc =  0.004378205763668096, scale = 1.840771049586619)
VaR_99 = VaR_t(0.01, df = 8.383690833071302, loc =  0.004378205763668096, scale = 1.840771049586619)

def plot_VaR(df=8.383690833071302, loc=0.004378205763668096, scale = 1.840771049586619):
    """Return plot of distribution and VaR"""

    x = np.linspace(-7,7,100)
    data = ss.t.pdf(x, df = df, loc=loc, scale = scale)

    fig, ax = plt.subplots(1, 1)
    #ax.hist(data, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    plt.plot(x, data, color = 'red', label = 'teoretyczna')
    ax.set_xlim([np.min([VaR_99,x[0]])-0.5, x[-1]])
    ax.hist(zwrot, density = True, bins='auto', histtype='stepfilled', color = 'blue', alpha=0.2, label = 'nasz rozkład')
    plt.axvline(x = VaR_95, color = 'black', label = 'VaR 95 teoretyczny')
    plt.axvline(x = VaR_99, color = 'black', label = 'VaR 99 teoretyczny')
    plt.axvline(x = np.quantile(zwrot, 0.05), color = 'gray', label = 'VaR 95 nasz')
    plt.axvline(x = np.quantile(zwrot, 0.01), color = 'gray', label = 'VaR 99 nasz')
    ax.legend(loc='best', frameon=False)

    plt.show()

print(plot_VaR())