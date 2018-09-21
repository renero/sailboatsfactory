"""Candle Stick Console.

Usage:
  cs_console.py [-i <input_filename] [-o <output_name>] [-e <encoding>]
                [-p <parameters_file>]
  cs_console.py (-h | --help)
  cs_console.py --version

Options:
  -i <input>        Input CSV file to be used
  -o <output>       Input CSV file to be used
  -e <encoding>     Encoding of input file (closing, high, low, open)
  -p <params_file>  Path to the parameters file to be used
  -v --version      Show version.
  -h --help         Show this screen.
"""

from docopt import docopt


def main(arguments):
    pass


if __name__ == "__main__":
    arguments = docopt(__doc__, version='Candlestick Encoder 0.1')
    main(arguments)
