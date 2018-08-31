from functools import reduce


class Candle:

    def __init__(self, raw, NameColumnMax='tickmaxprice',
                 nameColumnMin='tickminprice'):
        # I compute what is the median of the price movement to later use this
        # values to determine if price fluctuation in a single tick is
        # significative or not.
        self.price_change = abs(
            raw['tickmaxprice'] - raw['tickminprice']).median()

    def encode(self, row):

        bullish = bearish = False
        has_body = has_lower_shadow = has_upper_shadow = False

        open_value = row['tickopenprice']
        close_value = row['tickcloseprice']
        high_value = row['tickmaxprice']
        low_value = row['tickminprice']
        # MIN = min([open_value, high_value, low_value, close_value])
        # MAX = max([open_value, high_value, low_value, close_value])

        # What is the ratio between the |open - close| and the median of
        # price fluctuation
        body_ratio = abs(close_value - open_value) / self.price_change

        # Bullish or Bearish??
        if close_value >= open_value:
            bullish = True
        else:
            bearish = True

        if body_ratio > 0.05:
            has_body = True

        if bullish:
            upper_shadow_ratio = abs(
                high_value - close_value) / self.price_change
            lower_shadow_ratio = abs(
                low_value - open_value) / self.price_change
            if upper_shadow_ratio > 0.05:
                has_upper_shadow = True
            if lower_shadow_ratio > 0.05:
                has_lower_shadow = True
        else:
            upper_shadow_ratio = abs(
                high_value - open_value) / self.price_change
            lower_shadow_ratio = abs(
                low_value - close_value) / self.price_change
            if upper_shadow_ratio > 0.05:
                has_upper_shadow = True
            if lower_shadow_ratio > 0.05:
                has_lower_shadow = True

        encoded = ''
        if bearish:
            encoded += '-'
        encoded += str(int(has_lower_shadow)+1)
        encoded += str(int(has_body)+1)
        encoded += str(int(has_upper_shadow)+1)

        return encoded

    def make_sliding(self, df, N):
        dfs = [df.shift(-i).applymap(lambda x: [x]) for i in range(0, N)]
        return reduce(lambda x, y: x.add(y), dfs)
