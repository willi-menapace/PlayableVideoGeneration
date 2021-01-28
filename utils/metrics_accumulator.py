
import numpy as np

class MetricsAccumulator:
    '''
    Utility class for metrics accumulation and averaging
    '''

    def __init__(self):
        self.data = {}

    def reset(self):
        self.data = {}

    def add(self, key: str, value: np.ndarray):
        '''
        Adds value to the set of values represented by key

        :param key: The key to associate to the current values
        :param value: (size, ...) tensor with the values to store
        :return:
        '''

        if not key in self.data:
            self.data[key] = []

        self.data[key].append(value)

    def pop(self, key: str, dim=0):
        '''
        Obtains a tensor with all the concatenated values corresponding to a key
        Eliminates the key

        :param key: The key for which to retrieve the values
        :return: tensor with all the values concatenated along dimension dim
        '''

        if key not in self.data:
            raise Exception(f"Key '{key}' is not presetn")

        result = np.concatenate(self.data[key], axis=dim)
        del self.data[key]
        return result