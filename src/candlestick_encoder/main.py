import numpy as np
import pandas


from candlestick import Candlestick


def read(filename):
    df = pandas.read_csv(filename, sep='|', header=0, usecols=[0, 1, 2, 3])
    max_value = df.values.max()
    min_value = df.values.min()

    def normalize(x):
        return (x - min_value) / (max_value - min_value)

    df = df.applymap(np.vectorize(normalize))
    return df.round(4)


def main():
    df = read('/Users/renero/Documents/SideProjects/sailboatsfactory/data/ibex_1hr_1y.csv')
    for i in range(df.shape[0]):
        c = Candlestick(df.iloc[i], 'colh')
        print("{}, {}".format(i, c.encode_body()))
    print()


if __name__ == "__main__":
    main()
