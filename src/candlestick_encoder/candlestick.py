from typing import Any, Union


class Candlestick:

    min_relative_size = 0.02
    shadow_symmetry_diff_threshold = 0.1

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
        self.mid_body_percentile = (self.mid_body_point - self.low) / self.hl_interval_width

        # Calc the percentile position of min and max values
        self.min_percentile = (self.min - self.low) / self.hl_interval_width
        self.max_percentile = (self.max - self.low) / self.hl_interval_width

        # total candle interval range width and shadows lengths
        self.upper_shadow_len = self.high - self.max
        self.upper_shadow_percentile = self.upper_shadow_len / self.hl_interval_width
        self.lower_shadow_len = self.min - self.low
        self.lower_shadow_percentile = self.lower_shadow_len / self.hl_interval_width

        # Percentage of HL range occupied by the body.
        self.body_relative_size = self.oc_interval_width / self.hl_interval_width

        # Upper and lower shadows are larger than 2% of the interval range len?
        if self.upper_shadow_len / self.hl_interval_width > self.min_relative_size:
            self.has_upper_shadow = True
        if self.lower_shadow_len / self.hl_interval_width > self.min_relative_size:
            self.has_lower_shadow = True
        if self.has_upper_shadow and self.has_lower_shadow:
            self.has_both_shadows = True

        # Determine if body is centered in the interval. It must has two shadows with a difference
        # in their lengths less than 5% (param).
        self.shadows_relative_diff = abs(self.upper_shadow_percentile - self.lower_shadow_percentile)
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
        if any([self.body_in_center, self.body_in_lower_half, self.body_in_upper_half]) is False:
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

    def __init__(self, values, encoding="ohlc"):
        # Set all object variables to 0 or false.
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
            raise ValueError('Could not find all mandatory chars (o, h, l, c) in encoding ({})'.format(self.encoding))
        self.open = values[self.encoding.find('O')]
        self.high = values[self.encoding.find('H')]
        self.low = values[self.encoding.find('L')]
        self.close = values[self.encoding.find('C')]
        self.calc_parameters()

    def encode_with(self, encoding_substring):
        if self.body_in_center:
            print('  centered')
            return encoding_substring[0]
        else:
            if self.has_both_shadows:
                if self.body_in_upper_half:
                    return encoding_substring[1]
                else:
                    if self.body_in_lower_half:
                        return encoding_substring[2]
                    else:
                        raise ValueError('Body centered & 2 shadows but not in upper or lower halves')
            else:
                if self.has_lower_shadow:
                    return encoding_substring[3]
                else:
                    return encoding_substring[4]

    def encode_body(self):
        if self.body_relative_size <= self.min_relative_size:
            return self.encode_with('ABCDE')
        else:
            if self.body_relative_size <= 0.1 + 0.05:
                print('  10%')
                return self.encode_with('FGHIJ')
            else:
                if self.body_relative_size <= 0.25 + 0.1:
                    print('  25%')
                    return self.encode_with('KLMNO')
                else:
                    if self.body_relative_size <= 0.5 + 0.1:
                        print('  50%')
                        return self.encode_with('PPPQR')
                    else:
                        if self.body_relative_size <= 0.75 + 0.2:
                            print('  75%')
                            return self.encode_with('SSSTU')
                        else:
                            print('  ~ 100%')
                            return 'V'


    def info(self):
        print('O({:.3f}), H({:.3f}), L({:.3f}), C({:.3f})'.format(self.open, self.high, self.low, self.close))
        print('mid body point: {:0.2f}'.format(self.mid_body_point))
        print('body center percentile: {:0.2f}'.format(self.mid_body_percentile))
        print('body min/max percentiles: {:0.2f}/{:0.2f}'.format(self.min_percentile, self.max_percentile))
        print('body centered? {}'.format(self.body_in_center))
        print('body in upper half? {}'.format(self.body_in_upper_half))
        print('body in lower half? {}'.format(self.body_in_lower_half))
        print('body relative width: {:0.2f}'.format(self.body_relative_size))
        print('up/lw shdws width: {:0.2f}/{:0.2f}'.format(self.upper_shadow_percentile, self.lower_shadow_percentile))
        print('up/lw relative diff: {:0.2f}'.format(self.shadows_relative_diff))
        print('upper/lower shadows? {}/{}'.format(self.has_upper_shadow, self.has_lower_shadow))
        print('shadows are symmetric: {}'.format(self.shadows_symmetric))
        print('--')

    def values(self):
        print('O({:.3f}), H({:.3f}), L({:.3f}), C({:.3f})'.format(self.open, self.high, self.low, self.close))
