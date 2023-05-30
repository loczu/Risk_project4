import pandas as pd
import vartests


def data(path:str = "danekawa.csv" ):
    """Return pandas data"""

    return pd.read_csv(path)["data"]


def Christoffersen(data, VaR=-0.86, conf_level=0.95):
    """Return Christoffersen test"""

    Vec = []
    for sample in data:
        if sample < VaR:
            Vec.append(1)
        else:
            Vec.append(0)

    return vartests.duration_test(Vec, conf_level)

print(Christoffersen(data()))

# troche nie czaje jaki on tu rozkład testuje (?)