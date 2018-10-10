class CSLogger:

    _DEBUG = 4
    _INFO = 3
    _WARN = 2
    _ERROR = 1

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
        print('INFO: {}'.format(msg))

    def warn(self, msg):
        if self._level < self._WARN:
            return
        print('WARN: {}'.format(msg))

    def error(self, msg):
        if self._level < self._ERROR:
            return
        print('ERROR: {}'.format(msg))
