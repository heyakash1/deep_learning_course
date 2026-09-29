import numpy as np
import boolean_functions as bf

def arity(truth_table: tuple) -> int:
    return len(truth_table[0][0])


class Perceptron:

    def __init__(self, n_inputs: int, lr: float = 1.0, seed: int | None = None):
        self.learning_rate = lr
        self.w = np.random.default_rng(seed).uniform(low=-1.0,high=1.0,size=n_inputs+1)

    def _augment(self, x: tuple) -> np.ndarray:
        return np.asarray((1,) + x)
    
    def predict(self, x: tuple) -> int:
        aug_input = self._augment(x)
        aggregation = np.dot(self.w,aug_input)
        return int(aggregation >= 0)


    def fit(self, truth_table: tuple, max_epochs: int=100) -> tuple[bool,int]:
        """epochs_used is the number of passes over the data, including the last pass that had no errors."""
        for epochs in range(0,max_epochs):
            errors = 0
            for x,expected in truth_table:
                result = self.predict(x)
                if result != expected:
                    errors += 1
                    self.w += self.learning_rate*(expected-result)*self._augment(x)
            if errors == 0: break
        else:
            return (False, max_epochs)
        return (True,epochs+1)

if __name__ == "__main__":
    p = Perceptron(2)
    print(p.fit(bf.functions["XOR"]))