import boolean_functions as bf
import numpy as np
from perceptron import Perceptron, arity

SEED = 1


def learns_correctly(p: Perceptron, truth_table: tuple) -> bool:
    for row,expected in truth_table:
        if p.predict(row) != expected:
            return False
    return True


def run_all_functions(functions: dict) -> None:
    converged_count = 0

    for name, truth_table in functions.items():
        p = Perceptron(arity(truth_table), seed=SEED)

        converged, epochs = p.fit(truth_table)
        learned = learns_correctly(p, truth_table)

        status = "converged" if converged else "not converged"
        note = "" if learned == converged else " <-- MISMATCH: fit and re-check disagree"
        print(f"{name:<12} {status:<14} {epochs:>3} epochs w = {np.round(p.w,3)}{note}")

        converged_count += converged

    print(f"\n{converged_count}/{len(functions)} functions converged")


if __name__ == "__main__":
    run_all_functions(bf.functions)