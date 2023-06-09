import matplotlib.pyplot as plt
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

#BIC = 5612.520903142506
