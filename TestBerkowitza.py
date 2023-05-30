from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

def data(path:str = "danekawa.csv" ):
    """Return pandas data"""

    return pd.read_csv(path)["data"]

def zwroty(data):
    "Return list of returns [L_1, L_2,...,L_n]"

    L = []
    for idx in range(1,len(data)):
        L.append(np.log(data[idx]/data[idx-1]))

    return L

def Berkowitz(window_size=50, df = zwroty(data())):
    """Return Berkowitz test"""
    U = []
    
    for idx in range(1,len(df) - window_size):
        window_data = df[(idx-1):idx+window_size]
        F =  ECDF(window_data)
        U.append(F(df[(idx)+window_size]))
      
    return U

# W Berkowitz używam ecdf, ale to chyba powinien być nasz rozkład policzony dla każdego okna, żeby to miało sens (?)
# No bo jaki rozkład testujemy? empiryczny? XD

def test_uniform(U = Berkowitz()):
    """Return histogram and p_value of K-S test.
      P-value should be greater than 0.05"""

    plt.hist(U, np.arange(-0.25,1.25,0.01), density=True)
    plt.plot()
    plt.show()

    return stats.kstest(U, stats.uniform(loc=0.0, scale=1).cdf)

print(test_uniform())

#Jak nałoży się odwotną dystrybuntę na U to powinien rozkład normalny wyjść