import qutip
import numpy as np
import isingify
import pickle
import sage.all
from sage.all import ZZ
# from sage.misc.persist import SageUnpickler
outputdir = '/home/adam/Desktop/work/2021 Summer Research/code'
options = qutip.solver.Options(nsteps=10000)

class SVPbyQuantumWalk:
    """Thanks to Jake Lishman for the help with bugs"""
    def __init__(self, dimension, k, times, qudit_mapping, graph, lattice):
        self.dim = dimension
        self.k = k
        self.times = times
        self.qudit_mapping = qudit_mapping
        self.graph = graph
        self.lattice = lattice
        if self.qudit_mapping == 'bin':
            self.n = self.dim * int(np.ceil(np.log2(self.k)) + 1)
        elif self.qudit_mapping == 'ham':
            self.n = self.dim * self.k  # unsure check this
        else:
            raise KeyError('Define encoding type: "bin" or "ham"')
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
        # Need to check all Z.I combinations
        multi_z = qutip.tensor([qutip.sigmaz()] * self.n)
        result = qutip.sesolve(self.H, self.reg, self.times, e_ops=[multi_z], progress_bar=True, options=options)
        print('result', result)
        print('expect', result.expect)
        self.result = result

    def run(self, label):
        # create the lattice
        # lattice = np.random.randint(low=0, high=5, size=(self.dim, self.dim), dtype=int)
        self.SVPtoH()
        self.execute()
        with open(outputdir + '/result{}'.format(label), 'wb') as f:
            pickle.dump(self.result, f)

def main():
    # use sitek that specifies number of qubits
    dim = 3
    k = 4
    graph = 'FULL'
    qudit_mapping = 'bin'  # exclusively use bin for now
    times = np.linspace(0.0, 1.0, 101)
    with open(outputdir + '/bases', 'rb') as f:
        lattice = pickle.load(f) # unpickle
    lattice_good = lattice[1]
    lattice_bad = lattice[0]

    experiment_good = SVPbyQuantumWalk(dim, k, times, qudit_mapping, graph, lattice_good)
    experiment_good.run('Good')
    experiment_bad = SVPbyQuantumWalk(dim, k, times, qudit_mapping, graph, lattice_bad)
    experiment_bad.run('Bad')


if __name__ == "__main__":
    main()
