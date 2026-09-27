import boolean_functions
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

def find_threshold(truth_table: tuple, n: int) -> list:
    """
    """

    ans = []
    for b in range(0,n+1):
        flag = False
        for _ in truth_table:
            if MPNeuron(b=b).predict(_[0]) != _[1]:
                flag = True
                break

        if not flag:
            ans.append(b)

    return ans
