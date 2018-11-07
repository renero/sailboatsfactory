import argparse


class Arguments(object):
    args = None

    def __init__(self):
        ActionHelp = """
            Start = Starts the daemon (default)
            Stop = Stops the daemon
            Restart = Restarts the daemon
            """
        parser = argparse.ArgumentParser()

        parser.add_argument('action', nargs='1', default='predict',
                            choices=('train', 'predict', 'ensemble'),
                            help=ActionHelp)
        parser.add_argument('-t', '--ticks', nargs=1, type=str,
                            help='Ticks file to process')
        parser.add_argument('-w', '--window', nargs=1, type=int,
                            help='Window size for the LSTM')

        self.args = parser.parse_args()
        action_name = '_{}'.format(self.args.action)
        setattr(self, action_name, True)


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
