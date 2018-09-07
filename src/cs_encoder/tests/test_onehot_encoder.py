import numpy as np
from onehot_encoder import OnehotEncoder
from pprint import pprint


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
