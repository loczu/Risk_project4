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