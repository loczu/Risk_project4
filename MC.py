import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def zwroty(table):
    zwrot = []
    for i in range(1, len(table)):
        zwrot.append(np.log(table[i]/table[i-1]) * 100)
        
    return zwrot

data = pd.read_csv("danekawa.csv")
df = pd.DataFrame(data)
cena = df['data']
cena = round((1/0.00045359237) * cena, 2)
zwrot = zwroty(cena)

p1 = np.arange(1, 5, 1)
q1 = np.arange(1, 5, 1)

zwrot_ = zwrot - np.mean(zwrot)
p = 0
q = 0
coeffs = []
BIC_min = 100000
for i in range(len(p1)):
    for j in range(len(q1)):
        model = ARIMA(zwrot_, order = (p1[i], 0, q1[j])).fit()
        if model.bic < BIC_min:
            coeffs = model.params #wartosci współczynników dobranego modelu (w kolejnosci jak w summary: stała, współczynniki phi, współczynniki theta, sigma^2)
            BIC_min = model.bic
            p = p1[i]
            q = q1[j]
#BIC = 5612.520903142506 p = 1 q = 1

MC = 1000
quantiles = []

for i in range(MC):
    predictions = [zwrot_[-1]]
    noise = ss.norm.rvs(0, np.sqrt(coeffs[3]), size = 251)
    for i in range(1, 251):
        x = coeffs[0] + predictions[i-1] * coeffs[1] + coeffs[2] * noise[i-1] + noise[i]
        predictions.append(x)
    quantiles.append(predictions[-1])
    

fig, ax = plt.subplots(1, 1)
ax.hist(zwrot, density = True, bins='auto', histtype='stepfilled', color = 'blue', alpha=0.2, label = 'nasz rozkład')
plt.axvline(x = np.quantile(quantiles, 0.05), color = 'black', label = 'VaR 95')
plt.axvline(x = np.quantile(quantiles, 0.01), color = 'gray', label = 'VaR 99')
plt.title('Metoda historyczna zwykła')
ax.legend(loc='best', frameon=False)

plt.show()
