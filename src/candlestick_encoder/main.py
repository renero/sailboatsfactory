import numpy as np

from candlestick import Candlestick


def main():
    c = Candlestick(np.array([1.0, 10.0, 0.0, 9.0]), "ohlc")
    c.info()
    print('Encoding body as: {}'.format(c.encode_body()))


if __name__== "__main__":
  main()