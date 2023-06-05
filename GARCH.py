# GARCH(1,1) Model in Python
#   uses maximum likelihood method to estimate (omega,alpha,beta)
# (c) 2014 QuantAtRisk, by Pawel Lachowicz; tested with Python 3.5 only
from arch import arch_model
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd

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

sigma = [1.4]
for i in range(1,len(zwrot)):
    x = res.params[1] + zwrot[i-1] * res.params[2] + sigma[i-1] * res.params[3]
    sigma.append(x)

plt.plot(zwrot)
plt.plot(res._volatility)
plt.show()