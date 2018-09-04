import numpy as np
from keras.utils import to_categorical
from pprint import pprint


class ValidationError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class OnehotEncoder:
    _signed = False
    _states = set()
    _dict = dict()
    _sign_dict = {'p': +1, 'n': -1}

    def __init__(self, signed=False):
        self._signed = signed

    def reset(self):
        self._states = set()
        self._dict = dict()
        return self

    def fit(self, data):
        """Obtain the set of unique strings used in the data to shape
        a dictionary with a numeric mapping for each of them

        ## Arguments:
          - data: Either, a 1D or 2D array of strings. If the attribute
          'signed' is set to True when creating the encoder, the
          first character of every string is suposed to encode the
          sign of the element, so instead of encoding it as
          `[0, 0, ..., 1, 0 ... 0]`, it will be encoded as
          `0, 0, ..., -1, 0 ... 0]` with -1 if the sign is negative.

        ## Return Values:
          - The object, updated.
        """
        # Check if the array is 1D or 2D
        if len(data.shape) == 2:
            if self._signed is True:
                [self._states.update([chr[1:] for chr in l]) for l in data]
            else:
                [self._states.update(l) for l in data]
        elif len(data.shape) == 1:
            if self._signed is True:
                self._states.update([chr[1:] for chr in data])
            else:
                self._states.update(data)
        else:
            raise ValidationError('1D or 2D array expected.', -1)
        # Build the dict.
        self._dict = {k: v for v, k in enumerate(sorted(list(self._states)))}
        return self

    def fit_from_dictionary(self, data):
        if len(data.shape) == 1:
            self._states.update(data)
        else:
            raise ValidationError('1D array expected as dictionary.', -1)
        self._dict = {k: v for v, k in enumerate(sorted(list(self._states)))}
        return self

    def transform(self, data):
        if len(data.shape) == 1 or len(data.shape) == 2:
            num_arrays = data.shape[0] if len(data.shape) == 2 else 1
            num_strings = data.shape[1] if len(
                data.shape) == 2 else data.shape[0]
            num_states = len(self._states)
            data = data.reshape([num_arrays, num_strings])
            if self._signed is True:
                transformed = np.empty([num_arrays, num_strings, num_states])
                for i in range(num_arrays):
                    for j in range(num_strings):
                        code = to_categorical(
                            self._dict[data[i][j][1:]],
                            num_classes=len(self._states))
                        sign = self._sign_dict[data[i][j][0].lower()]
                        transformed[i][j] = np.dot(code, sign)
            else:
                transformed = np.array([
                    to_categorical(
                        [self._dict[x] for x in y],
                        num_classes=len(self._states)) for y in data
                ])
        else:
            raise ValidationError('1D or 2D array expected.', -1)
        return transformed


data = np.array([['a', 'b', 'c'], ['b', 'd', 'e']])
data1d = np.array(['a', 'b', 'c'])
my_dict = np.array(['a', 'b', 'c', 'd', 'e'])
signed_data = np.array([['pa', 'nb', 'pc'], ['nb', 'pd', 'ne']])
signed_data1d = np.array(['pa', 'nb', 'pc'])
signed_dict = np.array(['a', 'b', 'c', 'd', 'e'])

ohe = OnehotEncoder(signed=True)
ohe.fit(signed_data)
st = ohe.transform(signed_data)
pprint(st)
st = ohe.transform(signed_data1d)
pprint(st)

ohe = OnehotEncoder()
ohe.fit(data)
pprint(ohe.transform(data))
ohe.reset()
ohe.fit_from_dictionary(my_dict)
pprint(ohe.transform(data1d))
