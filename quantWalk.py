import qutip
import numpy as np
import isingify
import pickle
import itertools
import sage.all
from sage.all import ZZ
# from sage.misc.persist import SageUnpickler
outputdir = '/home/adam/Desktop/work/2021 Summer Research/code/Results'
options = qutip.solver.Options(nsteps=10000)

class SVPbyQuantumWalk:
    """Thanks to Jake Lishman for the help with bugs"""
    def __init__(self, dimension, k, times, qudit_mapping, graph, lattice, iteration):
        self.dim = dimension
        self.k = k
        self.times = times
        self.qudit_mapping = qudit_mapping
        self.graph = graph
        self.lattice = lattice
        if self.qudit_mapping == 'bin':
            self.n = self.dim * int(np.ceil(np.log2(self.k)) + 1) # check why
            # say 3 qudits encoded by log2(4) + 1 = 3 qubits each = 9 qubits
        elif self.qudit_mapping == 'ham':
            self.n = self.dim * self.k  # unsure check this
        else:
            raise KeyError('Define encoding type: "bin" or "ham"')
        self.iteration = iteration
        self.N = 2 ** self.n
        self.lam = 1 / self.N

        # Initialise a register which has `n` qubits all in the |+> state.
        plus_state = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        self.reg = qutip.tensor([plus_state] * self.n)

    def SVPtoH(self):

        # graph hamiltonian
        if self.graph == 'FULL':  # (full graph)
            hg = self.lam * self.N * (qutip.qeye([2]*self.n) - self.reg.proj())
        elif self.graph == 'HYPER':  # (hypercube)
            base = self.N * qutip.qeye([2]*self.n)
            for i in range(self.n):
                # Make an object that looks like a sum over every possible
                #   I.I.[...].I.X.I.I.[...].I
                # (i.e. all single-qubit sigma-x operators)
                parts = []
                # Identity on `i` qubits.
                if i > 0:
                    parts.append(qutip.qeye([2] * i))
                # Now put in a sigma-x.
                parts.append(qutip.sigmax())
                # Identity on the rest of the qubits.
                if i < self.n - 1:
                    parts.append(qutip.qeye([2] * (self.n - 1 - i)))
                # Now tensor-product all of them together to make the operator.
                base -= qutip.tensor(parts)
            hg = self.lam * base
        else:
            raise KeyError('Define graph type: FULL or HYPER')

        # problem hamiltonian
        gram = self.lattice @ self.lattice.T
        if self.qudit_mapping == 'bin':
            jmat, hvec, ic = isingify.svp_isingcoeffs_bin(gram, self.k)
        elif self.qudit_mapping == 'ham':
            jmat, hvec, ic = isingify.svp_isingcoeffs_ham(gram, self.k)
        else:
            raise KeyError('Define encoding type: "bin" or "ham"')
        hp = qutip.Qobj(
            isingify.ising_hamiltonian(jmat, hvec, ic),
            dims=[[2]*self.n, [2]*self.n],
        )

        # Full Hamiltonian, with tensor-product structure.
        self.H = hp + hg
        # print('H', self.H)

    def execute(self):
        # This makes a Z.Z.Z.[...] operator, rather than just a single-qubit
        # operator.
        multi_z = qutip.tensor([qutip.sigmaz()] * self.n)
        # Need to check all Z.-Z (0 and 1) combinations!!!
        # all_comb = []
        # for i in range(self.n + 1):
           # operators = [qutip.sigmaz()] * i
           # operators += [-qutip.sigmaz()] * (self.n - i)
            # perm = list(itertools.permutations(operators))
           # all_comb.append(qutip.tensor(operators))
            # for j in range(len(perm)):
               # a = list(perm[j])
               # all_comb.append(qutip.tensor(a[0]))
        result = qutip.sesolve(self.H, self.reg, self.times, e_ops=[], progress_bar=True, options=options)
        # print('result', result)
        # print('states', result.states)
        self.result = result

    def run(self, label):
        # create the lattice
        # lattice = np.random.randint(low=0, high=5, size=(self.dim, self.dim), dtype=int)
        # values in the lattice were restricted to 5 due to the qudit mapping (this is based on k?) i think
        # so with really large values there is a problem of not enough qubits in a qudit to encode the vectors
        self.SVPtoH()
        self.execute()
        with open(outputdir + '/result{}'.format(label) + '{}'.format(self.iteration), 'wb') as f:
            pickle.dump(self.result, f)

def main():
    # use sitek that specifies number of qubits
    dim = 3
    k = 4
    graph = 'FULL'
    qudit_mapping = 'bin'  # exclusively use bin for now
    times = np.linspace(0.0, 1.0, 10) # do for different times!!!
    with open(outputdir + '/bases', 'rb') as f:
        lattice = pickle.load(f) # unpickle
    lattice_good = lattice[1]
    lattice_bad = lattice[0]

    for i in range(10):
        experiment_good = SVPbyQuantumWalk(dim, k, times, qudit_mapping, graph, lattice_good, i)
        experiment_good.run('Good')
        experiment_bad = SVPbyQuantumWalk(dim, k, times, qudit_mapping, graph, lattice_bad, i)
        experiment_bad.run('Bad')


if __name__ == "__main__":
    main()
