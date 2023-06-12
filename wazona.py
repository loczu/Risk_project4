import matplotlib.pyplot as plt
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

def wazona(lam, data, level):
    w = []
    for i in range(len(data)):
        w.append(lam**(i+1))

    waga = []
    for i in range(len(data)):
        waga.append(1/(1+sum(w))*lam**i)

    arr = np.array([data,
                    waga])

    array = arr[:, np.argsort(arr[0, :])]

    x = 0
    for i in range(len(data)):
        x += array[1,i]
        if x > (1-level):
            res = array[0,i]
            break
    
    return res

print(wazona(0.96, zwrot, 0.95), wazona(0.96, zwrot, 0.99))

fig, ax = plt.subplots(1, 1)
ax.hist(zwrot, density = True, bins='auto', histtype='stepfilled', color = 'blue', alpha=0.2, label = 'nasz rozkład')
plt.axvline(x = wazona(0.96, zwrot, 0.95), color = 'black', label = 'VaR 95')
plt.axvline(x = wazona(0.96, zwrot, 0.99), color = 'gray', label = 'VaR 99')
plt.title('Metoda historyczna ważona')
ax.legend(loc='best', frameon=False)

plt.show()
