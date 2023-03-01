import time


def deltaT(t1: float, msg: str) -> float:
    delta = time.time() - t1
    t1 = time.time()
    print(f"\n\x1B[32m ∆ {msg} {delta:0.3f}sec \x1b[0m\n")
    return t1
