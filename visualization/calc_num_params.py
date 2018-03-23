import numpy as np

layers = [
        [5, 32, 16],
        [16, 64, 32],
        [32, 128, 64],
        [64, 128, 128, 10]]

lParams = list()
for l in layers:
    s = 0
    for j in range(len(l)-1):
        s += (l[j]*l[j+1])
    lParams.append(s)

print(lParams)
print(np.sum(lParams))
