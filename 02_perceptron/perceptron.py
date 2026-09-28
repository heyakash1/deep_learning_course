import numpy as np


class Perceptron:

    def __init__(self, n_inputs: int, lr: float = 1.0, seed: int | None = None):
        self.learning_rate = lr
        self.w = np.random.default_rng(seed).uniform(low=-1.0,high=1.0,size=n_inputs+1)

    def predict(self, x: tuple) -> int:
        aug_input = (1,) + x
        aggregation = np.dot(self.w,aug_input)
        return int(aggregation >= 0)

    def fit(self, truth_table: tuple, max_epochs: int=100) -> tuple:
        pass

def arity(truth_table: tuple) -> int:
    return len(truth_table[0][0])

# print(Perceptron(2).w)
# print(Perceptron(2).w)
p = Perceptron(2)
p.w = np.array([-2,1,1])
print(p.predict((1,0)))