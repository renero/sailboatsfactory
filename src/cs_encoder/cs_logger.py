class CSLogger:

    _DEBUG = 4
    _INFO = 3
    _WARN = 2
    _ERROR = 1

    # https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    _level = 0

    def __init__(self, level=0):
        self._level = level

    def debug(self, msg):
        if self._level < self._DEBUG:
            return
        print('DEBUG: {}'.format(msg))

    def info(self, msg):
        if self._level < self._INFO:
            return
        print('INFO: {}{}{}'.format(self.OKGREEN, msg, self.ENDC))

    def warn(self, msg):
        if self._level < self._WARN:
            return
        print('{}WARN: {}{}'.format(self.WARNING, msg, self.ENDC))

    def error(self, msg):
        if self._level < self._ERROR:
            return
        print('{}ERROR: {}{}'.format(self.FAIL, msg, self.ENDC))
