import numpy as np

# ActGate
def ActGate(n=64):
    Source = np.zeros((n, n, n)).astype(int)
    Source[32][32][32] = 1
    return Source;
