import yaml


class Params(object):
    def __init__(self, params_filepath='./cs_encoder/params.yaml'):
        """
        Init the class with the number of categories used to encode candles
        """
        with open(params_filepath, 'r') as stream:
            try:
                self.params = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        for param_name in self.params.keys():
            attribute_name = '_{}'.format(param_name)
            setattr(self, attribute_name, self.params[param_name])
