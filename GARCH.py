# GARCH(1,1) Model in Python
#   uses maximum likelihood method to estimate (omega,alpha,beta)
# (c) 2014 QuantAtRisk, by Pawel Lachowicz; tested with Python 3.5 only
from arch import arch_model
import matplotlib.pyplot as plt
import pandas as pd
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
 

garch11 = arch_model(zwrot, mean = "Constant", p=1, q=1, vol = "GARCH", dist='StudentsT')
res = garch11.fit()
print(res.summary())

#omega = 0.2202 alpha = 0.0596 beta = 0.8913

#res.plot()
#plt.show()

zwrot_ = []
for i in range(len(zwrot)):
    x = (zwrot[i] - res.params[0])/res._volatility[i]
    zwrot_.append(x)

q99, q95  = np.quantile(np.sort(zwrot_), 0.01), np.quantile(np.sort(zwrot_), 0.05)
i95, i99 = 0, 0

for i in range(len(zwrot_)):
    if np.sort(zwrot_)[i-1] < q99 < np.sort(zwrot_)[i]:
        i99 = i
    if np.sort(zwrot_)[i-1]< q95 < np.sort(zwrot_)[i]:
        i95 = i
        break

sigma99 = res.params[1] + zwrot[i99] * res.params[2] + res._volatility[i99] * res.params[3]
sigma95 = res.params[1] + zwrot[i95] * res.params[2] + res._volatility[i95] * res.params[3]

print(sigma99 * q99 + res.params[0], sigma95 * q95 + res.params[0])

fig, ax = plt.subplots(1, 1)
ax.hist(zwrot, density = True, bins='auto', histtype='stepfilled', color = 'blue', alpha=0.2, label = 'nasz rozkład')
plt.axvline(x = sigma95 * q95 + res.params[0], color = 'black', label = 'VaR 95')
plt.axvline(x = sigma99 * q99 + res.params[0], color = 'gray', label = 'VaR 99')
plt.title('Metoda historyczna GARCH(1, 1)')
ax.legend(loc='best', frameon=False)
plt.show()
#-2.347473198908951 -1.5435777994133173