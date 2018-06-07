import numpy as np

from candlestick import Candlestick


def main():
    c = Candlestick(np.array([1.0, 10.0, 0.0, 9.0]))
    c.info()


if __name__== "__main__":
  main()