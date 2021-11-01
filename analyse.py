import matplotlib.pyplot as plt
import qutip
import numpy as np
import pickle
from collections import defaultdict

outputdir = '/home/adam/Desktop/work/2021 Summer Research/code/Results'
dim = 3
qb_per_qd = 3

def success_rate(x,y, length, tolerance):
    acc = []
    for i in x:
        if i-length-tolerance <= 0 and i-length+tolerance >= 0:
            acc.append(i)
    freq = 0
    for i in acc:
        freq += y[x.index(i)]
    return freq/sum(y)

def stat(x,y):
    m = max(y)
    print('Most probable value:', x[y.index(m)])
    mean = sum([a*b for a,b in zip(x,y)])
    print('Mean:', mean)
    dev = [((a-mean)**2)*b for a,b in zip(x,y)]
    print('Spread:', np.sqrt(sum(dev)))

with open(outputdir + '/resultGood0', 'rb') as f:
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
    vectors.append(np.sqrt(sum([x**2 for x in vect])))

def list_duplicates(seq, idx):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key, sum([probs[idx][x] for x in locs])) for key,locs in tally.items())

#for dup in sorted(list_duplicates(vectors, 8)):
    #print(dup)

for a in range(2):
    data = sorted(list_duplicates(vectors, a))
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1][0][0])
    plt.bar(x, y)
    plt.xlabel('Length')
    plt.ylabel('Probability')
    plt.title('Shortest vector length at time ' + str(a))
    #plt.show()
    print('Success rate at time', a)
    print(success_rate(x,y,np.sqrt(3),0))
    stat(x,y)