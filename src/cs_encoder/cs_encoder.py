import numpy as np


class CSEncoder:
    """Takes as init argument a numpy array with 4 values corresponding
    to the O, H, L, C values, in the order specified by the second argument
    string `encoding` (for instance: 'ohlc').
    """

    min_relative_size = 0.02
    shadow_symmetry_diff_threshold = 0.1
    _diff_tags = ['open', 'close', 'high', 'low', 'min', 'max']
    _def_upper_limits = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    _def_thresholds = [0.02, 0.05, 0.1, 0.1, 0.1, 0.1]
    _def_prcntg_encodings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    def __init__(self, values, encoding="ohlc"):
        """
        Takes as init argument a numpy array with 4 values corresponding
        to the O, H, L, C values, in the order specified by the second argument
        string `encoding` (for instance: 'ohlc').
        """
        self.encoding = encoding.upper()
        self.open = self.close = self.high = self.low = 0.0
        self.min = self.max = self.min_percentile = self.max_percentile = 0.0
        self.mid_body_percentile = self.mid_body_point = 0.0
        self.positive = self.negative = False
        self.has_upper_shadow = self.has_lower_shadow = self.has_both_shadows = False
        self.shadows_symmetric = False
        self.body_in_upper_half = self.body_in_lower_half = self.body_in_center = False
        self.hl_interval_width = self.upper_shadow_len = self.lower_shadow_len = 0.0
        self.upper_shadow_percentile = self.lower_shadow_percentile = 0.0
        self.oc_interval_width = self.body_relative_size = 0.0
        self.shadows_relative_diff = 0.0

        # Assign the proper values to them
        if self.correct_encoding() is False:
            raise ValueError(
                'Could not find all mandatory chars (o, h, l, c) in encoding ({})'.
                format(self.encoding))
        self.open = values[self.encoding.find('O')]
        self.high = values[self.encoding.find('H')]
        self.low = values[self.encoding.find('L')]
        self.close = values[self.encoding.find('C')]
        self.calc_parameters()

        # Assign default encodings for movement
        self.encoded_delta_close = 'pA'
        self.encoded_delta_high = 'pA'
        self.encoded_delta_low = 'pA'
        self.encoded_delta_max = 'pA'
        self.encoded_delta_min = 'pA'
        self.encoded_delta_open = 'pA'


    @staticmethod
    def div(a, b):
        b = b + 0.000001 if b == 0 else b
        return a / b

    def calc_parameters(self):
        # positive or negative movement
        if self.close > self.open:
            self.max = self.close
            self.min = self.open
            self.positive = True
        else:
            self.max = self.open
            self.min = self.close
            self.negative = True

        # Length of the interval between High and Low
        self.hl_interval_width = abs(self.high - self.low)
        self.oc_interval_width = self.max - self.min

        # Mid point of the body (absolute value)
        self.mid_body_point = self.min + (self.oc_interval_width / 2.0)
        # Percentile of the body (relative)
        self.mid_body_percentile = self.div((self.mid_body_point - self.low),
                                            self.hl_interval_width)

        # Calc the percentile position of min and max values
        self.min_percentile = self.div((self.min - self.low),
                                       self.hl_interval_width)
        self.max_percentile = self.div((self.max - self.low),
                                       self.hl_interval_width)

        # total candle interval range width and shadows lengths
        self.upper_shadow_len = self.high - self.max
        self.upper_shadow_percentile = self.div(self.upper_shadow_len,
                                                self.hl_interval_width)
        self.lower_shadow_len = self.min - self.low
        self.lower_shadow_percentile = self.div(self.lower_shadow_len,
                                                self.hl_interval_width)

        # Percentage of HL range occupied by the body.
        self.body_relative_size = self.div(self.oc_interval_width,
                                           self.hl_interval_width)

        # Upper and lower shadows are larger than 2% of the interval range len?
        if self.div(self.upper_shadow_len,
                    self.hl_interval_width) > self.min_relative_size:
            self.has_upper_shadow = True
        if self.div(self.lower_shadow_len,
                    self.hl_interval_width) > self.min_relative_size:
            self.has_lower_shadow = True
        if self.has_upper_shadow and self.has_lower_shadow:
            self.has_both_shadows = True

        # Determine if body is centered in the interval. It must has
        # two shadows with a difference in their lengths less than 5% (param).
        self.shadows_relative_diff = abs(self.upper_shadow_percentile -
                                         self.lower_shadow_percentile)
        if self.has_both_shadows is True:
            if self.shadows_relative_diff < self.shadow_symmetry_diff_threshold:
                self.shadows_symmetric = True

        # Is body centered, or in the upper or lower half?
        if self.min_percentile > 0.5:
            self.body_in_upper_half = True
        if self.max_percentile < 0.5:
            self.body_in_lower_half = True
        if self.shadows_symmetric is True and self.body_relative_size > self.min_relative_size:
            self.body_in_center = True
        # None of the above is fulfilled...
        if any([
                self.body_in_center, self.body_in_lower_half,
                self.body_in_upper_half
        ]) is False:
            if self.lower_shadow_percentile > self.upper_shadow_percentile:
                self.body_in_upper_half = True
            else:
                self.body_in_lower_half = True

    def correct_encoding(self):
        """
        Check if the encoding proposed has all elements (OHLC)
        :return: True or False
        """
        # check that we have all letters
        return all(self.encoding.find(c) != -1 for c in 'OHLC')

    def encode_with(self, encoding_substring):
        if self.body_in_center:
            # print('  centered')
            return encoding_substring[0]
        else:
            if self.has_both_shadows:
                if self.body_in_upper_half:
                    return encoding_substring[1]
                else:
                    if self.body_in_lower_half:
                        return encoding_substring[2]
                    else:
                        raise ValueError(
                            'Body centered & 2 shadows but not in upper or lower halves'
                        )
            else:
                if self.has_lower_shadow:
                    return encoding_substring[3]
                else:
                    return encoding_substring[4]

    def __encode_body(self):
        if self.body_relative_size <= self.min_relative_size:
            return self.encode_with('ABCDE')
        else:
            if self.body_relative_size <= 0.1 + 0.05:
                # print('  10%')
                return self.encode_with('FGHIJ')
            else:
                if self.body_relative_size <= 0.25 + 0.1:
                    # print('  25%')
                    return self.encode_with('KLMNO')
                else:
                    if self.body_relative_size <= 0.5 + 0.1:
                        # print('  50%')
                        return self.encode_with('PQRST')
                    else:
                        if self.body_relative_size <= 0.75 + 0.1:
                            # print('  75%')
                            return self.encode_with('UVWXY')
                        else:
                            # print('  ~ 100%')
                            return 'Z'

    def encode_body(self):
        if self.positive:
            first_letter = 'p'
        else:
            first_letter = 'n'
        encoding = first_letter + self.__encode_body()
        setattr(self, 'encoded_body', encoding)
        # return encoding

    def encode_body_nosign(self):
        encoding = self.__encode_body()
        setattr(self, 'encoded_body', encoding)
        return encoding

    def __encode_movement(self,
                          value,
                          encoding=None,
                          upper_limits=_def_upper_limits,
                          thresholds=_def_thresholds,
                          encodings=_def_prcntg_encodings,
                          pos=0):
        """Tail recursive function to encode a value in one of the possible
        encodings passed as list. The criteria is whether the value is lower
        than a given upper_limit + threshold to use that encoding position.
        If not, jump to the next option until we find the upper limit, or
        move beyond limits.

        Args:
            value(float)  : the value to be encoded
            encoding (str): the resulting encoding found. Must be None in the
                            first call to the function.
            upper_limits(arr[float]): list of the upper limits to consider in
                            the coding schema to be used.
            thresholds(arr[float]): list of threshold to be added to the upper
                            limits.
            encodings (arr[str]): the list of resulting encodings, one for
                            each upper limit, plus one for the beyond last
                            limit case.
            pos:            controls recursion, must be 0 in the first call

        """
        if encoding is not None:
            return encoding
        if pos == len(encodings) - 1:
            # print('>> beyond limimts')
            return encodings[pos]
        if value <= upper_limits[pos] + thresholds[pos]:
            encoding = encodings[pos]
        return self.__encode_movement(value, encoding, upper_limits,
                                      thresholds, encodings, pos + 1)

    def encode_movement_nosign(self, prev_cs):
        """Compute the percentage of change for the OHLC values with respect
        to the relative range of the previous candlestick object (passed as
        argument). This allows to encode the movement of single candlestick.
        """
        # Build a dictionary with ratio of change of the difference between
        # each pair of OHLC values with respect to the range betwween High
        # and low.
        # Each key is in `diff_tags`.
        # deltas = {
        #     attr: self.div((getattr(self, attr) - getattr(prev_cs, attr)),
        #                    prev_cs.hl_interval_width)
        #     for attr in self._diff_tags
        # }
        for attr in self._diff_tags:
            delta = self.div((getattr(self, attr) - getattr(prev_cs, attr)),
                             prev_cs.hl_interval_width)
            enc = self.__encode_movement(delta)
            setattr(self, 'delta_{}'.format(attr), delta)
            setattr(self, 'encoded_delta_{}'.format(attr), enc)

    def encode_movement(self, prev_cs):
        """Compute the percentage of change for the OHLC values with respect
        to the relative range of the previous candlestick object (passed as
        argument). This allows to encode the movement of single candlestick.
        """
        for attr in self._diff_tags:
            delta = self.div((getattr(self, attr) - getattr(prev_cs, attr)),
                             prev_cs.hl_interval_width)
            encoding = self.__encode_movement(delta)
            if delta >= 0.0:
                sign_letter = 'p'
            else:
                sign_letter = 'n'
            setattr(self, 'delta_{}'.format(attr), delta)
            setattr(self, 'encoded_delta_{}'.format(attr), '{}{}'.format(
                sign_letter, encoding))

    def info(self):
        v = vars(self)
        for key, value in sorted(v.items(), key=lambda x: x[0]):
            if isinstance(value, np.float64):
                print('{:.<25}: {:>.3f}'.format(key, value))
            else:
                print('{:.<25}: {:>}'.format(key, value))

    def values(self):
        print('O({:.3f}), H({:.3f}), L({:.3f}), C({:.3f})'.format(
            self.open, self.high, self.low, self.close))
