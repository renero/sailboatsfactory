import yaml
from cs_logger import CSLogger


class Params(object):
    def __init__(self, params_filepath='./params.yaml'):
        """
        Init a class with all the parameters in the default YAML file.
        For each of them, create a new class attribute, with the same name
        but preceded by '_' character.
        """
        with open(params_filepath, 'r') as stream:
            try:
                self.params = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        for param_name in self.params.keys():
            attribute_name = '_{}'.format(param_name)
            setattr(self, attribute_name, self.params[param_name])

        # Start the logger
        self.log = CSLogger(self._log_level)
