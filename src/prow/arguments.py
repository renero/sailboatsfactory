import argparse


class Arguments(object):

    args = None

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', nargs=1, type=str,
                            help='Ticks file to process')
        parser.add_argument('-w', nargs=1, type=int,
                            help='Window size for the LSTM')

        self.args = parser.parse_args()

    @property
    def arg_ticks_file(self):
        if self.args.t is not None:
            return self.args.t[0]
        else:
            return None

    @property
    def arg_window_size(self):
        if self.args.w is not None:
            return self.args.w[0]
        else:
            return None
