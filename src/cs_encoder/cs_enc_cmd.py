"""Candle Stick Encoder.

Usage:
  cse.py [-i <input_filename] [-o <output_name>] [-e <encoding>]
         [-p <parameters_file>]
  cse.py (-h | --help)
  cse.py --version

Options:
  -i <input>        Input CSV file to be used
  -o <output>       Input CSV file to be used
  -e <encoding>     Encoding of input file (closing, high, low, open)
  -p <params_file>  Path to the parameters file to be used
  -v --version      Show version.
  -h --help         Show this screen.

"""
import sys

import numpy as np
import pandas
import yaml
from docopt import docopt

from cs_encoder import CSEncoder


def read(filename):
    df = pandas.read_csv(filename, sep='|', header=0, usecols=[0, 1, 2, 3])
    max_value = df.values.max()
    min_value = df.values.min()

    def normalize(x):
        return (x - min_value) / (max_value - min_value)

    df = df.applymap(np.vectorize(normalize))
    return df.round(4)


def main(arguments):
    prms_fname = 'params.yaml' if arguments['-p'] is None else arguments['-p']
    with open(prms_fname, 'r') as params_file:
        try:
            params = yaml.load(params_file)
            # Overwrite parameters from file if specified through cmd line.
            if arguments['-i'] is not None:
                params['input_filename'] = arguments['-i']
            if arguments['-o'] is not None:
                params['output_filename'] = arguments['-o']
            if arguments['-e'] is not None:
                params['encoding'] = arguments['-e']
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    df = read(params['input_filename'])
    if params['output_filename'] is None:
        params['output_filename'] = '/dev/stdout'
    out = open(params['output_filename'], 'w')
    print("id,cse", file=out)
    for i in range(df.shape[0]):
        c = CSEncoder(df.iloc[i], params['encoding'])
        print("{},{}".format(i, c.encode_body()), file=out)


# sys.argv = ['cse.py', '-i', '/path/tpo/file.ext']
if __name__ == "__main__":
    arguments = docopt(__doc__, version='Candlestick Encoder 0.1')
    main(arguments)
