import numpy as np
import pandas


from candlestick import Candlestick


def read(filename):
    df = pandas.read_csv(filename, sep='|', header=0, usecols=[0, 1, 2, 3])
    print(df.head(5), "\n--")
    max_value = df.values.max()
    min_value = df.values.min()

    def normalize(x):
        return (x - min_value) / (max_value - min_value)

    df = df.applymap(np.vectorize(normalize))

    return df.round(4)


def main():
    # c = Candlestick(np.array([1.0, 10.0, 0.0, 9.0]), "ohlc")
    # c.info()
    # print('Encoding body as: {}'.format(c.encode_body()))

    df = read('/Users/renero/Documents/SideProjects/sailboatsfactory/data/100.csv')
    print(df.head(5), "\n--")
    c = Candlestick(df.iloc[3], 'colh')
    c.info()
    c.encode_body()


if __name__ == "__main__":
    main()
