HIDDEN_WEIGHTS = {
    (0,0) : (0,0),
    (0,1) : (-1,2),
    (1,0) : (2,-1),
    (1,1) : (1,1),
}
HIDDEN_BIAS = -2


def _fires(w: tuple, x: tuple, bias: int) -> int:
    aggregation = sum(a * b for a, b in zip(w,x))
    aggregation += bias
    if aggregation >= 0:
        return 1
    return 0


def hidden_layer(x: tuple) -> dict:
    output = {}
    for pattern, weights in HIDDEN_WEIGHTS.items():
        output[pattern] = _fires(weights, x, HIDDEN_BIAS)

    return output

if __name__ == "__main__":
    print(hidden_layer((1,1)).values())