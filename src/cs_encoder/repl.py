from cs_encoder.cs_encoder import CSEncoder
# from cs_encoder.onehot_encoder import OnehotEncoder
from cs_encoder.ticks import Ticks
from cs_encoder.cs_plot import CSPlot

# from cs_encoder import CSEncoder
# from ticks import Ticks
# from cs_plot import CSPlot


ticks_file = '../data/100.csv'
cse_file = '../data/100.cse'

target_cols = [
    'tickopenprice', 'tickmaxprice', 'tickminprice', 'tickcloseprice'
]
ohlc_tags = ['o', 'h', 'l', 'c']
cse_tags = ['b', 'o', 'h', 'l', 'c']
ticks = Ticks.read_ohlc(ticks_file, target_cols, ohlc_tags)
encoder = CSEncoder()
encoder.fit(ticks, ohlc_tags)
cse = encoder.ticks2cse(ticks)
CSPlot().plot(ticks, ohlc_names=ohlc_tags)

# Save encodings to CSE file
encoder.save_cse(cse, cse_file)

# --

cse_codes = encoder.read_cse(cse_file, cse_tags)
rec_ticks = encoder.cse2ticks(cse_codes, ohlc_tags)
CSPlot().plot(rec_ticks, ohlc_names=ohlc_tags)

#
# -
#
# One hot encoding.
#
# -
#
# ohe = OnehotEncoder(signed=True).fit_from_dictionary(
#     np.array(CSEncoder._def_prcntg_encodings))
#
# values_to_encode = np.array([cse[i].encoded_delta_open for i in range(0, 4)])
# pprint(values_to_encode)
# encoded = ohe.transform(values_to_encode)
# pprint(encoded[0])
#
# decoded = ohe.decode(encoded[0])
# pprint(decoded)
