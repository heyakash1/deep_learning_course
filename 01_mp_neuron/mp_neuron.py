class MPNeuron:
    def __init__(self, b: int):
        self.threshold = b

    def predict(self, x: tuple) -> int:
        """
        """
        aggregation = sum(x)
        if aggregation >= self.threshold :
                return 1
        else :  return 0


if __name__ == "__main__":
    print(MPNeuron(b=2).predict((1,1)))
    print(MPNeuron(b=2).predict((1,0)))