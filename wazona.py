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

lam = 0.96
w = []
for i in range(len(zwrot)):
    w.append(lam**(i+1))

waga = []
for i in range(len(zwrot)):
    waga.append(1/(1+sum(w))*lam**i)
