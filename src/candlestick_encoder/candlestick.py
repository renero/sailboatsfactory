from typing import Any, Union


class Candlestick:

    min_relative_size = 0.02
    shadow_symmetry_diff_threshold = 0.05

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
        self.mid_body_point = self.open + (abs(self.open - self.close) / 2.0)
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

    def __init__(self, ohlc):
        # Set all object variables to 0 or false.
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
        self.open = ohlc[0]
        self.high = ohlc[1]
        self.low = ohlc[2]
        self.close = ohlc[3]
        self.calc_parameters()

    def info(self):
        print('body center percentile: {:0.2f}'.format(self.mid_body_percentile))
        print('body centered? {}'.format(self.body_in_center))
        print('body in upper half? {}'.format(self.body_in_upper_half))
        print('body in lower half? {}'.format(self.body_in_lower_half))
        print('body relative width: {:0.2f}'.format(self.body_relative_size))
        print('up/lw shdws wdth: {:0.2f}/{:0.2f}'.format(self.upper_shadow_percentile, self.lower_shadow_percentile))
        print('upper/lower shadows? {}/{}'.format(self.has_upper_shadow, self.has_lower_shadow))
        print('shadows are symmetric: {}'.format(self.shadows_symmetric))
