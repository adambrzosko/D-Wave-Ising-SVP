import matplotlib.pyplot as plt
import qutip
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

outputdir = '/home/adam/Desktop/work/2021 Summer Research/code'
dim = 3
qb_per_qd = 3

with open(outputdir + '/resultGood', 'rb') as f:
    result = pickle.load(f)  # unpickle

shape = result.states[0].shape

# get probabilities associated with the outputs (the index maps onto a bitstring f'{index:09b}')
probs = []
for t in range(len(result.states)):
    probs_t = []
    for i in range(shape[0]):
        a = result.states[t][i]
        probs_t.append(abs(a)**2)
    probs.append(probs_t)

# calculate the length of the vectors from bitstrings
vectors = []
for j in range(shape[0]):
    vect = []
    a = f'{j:09b}'
    for i in range(dim):
        vect.append(int(a[i*qb_per_qd:(i+1)*qb_per_qd], 2)) # slice the bitstring and convert from binary to int
    vectors.append(sum([x**2 for x in vect])/len(vect))

def list_duplicates(seq, idx):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key, sum([probs[idx][x] for x in locs])) for key,locs in tally.items())

for dup in sorted(list_duplicates(vectors, 8)):
    print(dup)

data = sorted(list_duplicates(vectors, 8))
x = []
y = []
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1][0][0])
plt.plot(x,y)
plt.show()
