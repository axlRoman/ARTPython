import numpy as np

class ART1:
    def __init__(self, num_features, vigilance):
        self.num_features = num_features
        self.vigilance = vigilance
        self.weights = []

    def train(self, patterns):
        for pattern in patterns:
            self._train_pattern(pattern)

    def _train_pattern(self, pattern):
        if not self.weights:
            self.weights.append(pattern)
            return

        for weight in self.weights:
            if len(weight) == len(pattern):
                match = np.sum(np.minimum(weight, pattern)) / np.sum(pattern)
                if match >= self.vigilance:
                    self._update_weight(weight, pattern)
                    return

        self.weights.append(pattern)

    def _update_weight(self, weight, pattern):
        weight[:] = np.minimum(weight, pattern)

    def predict(self, pattern):
        for i, weight in enumerate(self.weights):
            if len(weight) == len(pattern):
                match = np.sum(np.minimum(weight, pattern)) / np.sum(pattern)
                if match >= self.vigilance:
                    return i
        return -1
