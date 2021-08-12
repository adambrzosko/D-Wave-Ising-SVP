from qutip import *
import numpy as np
import isingify


class SVPbyQuantumWalk:

    def __init__(self, dimension, k, times, qudit_mapping, graph):
        self.dim = dimension
        self.k = k
        self.times = times
        self.qudit_mapping = qudit_mapping
        self.graph = graph
        if self.qudit_mapping == 'bin':
            self.n = self.dim * int(np.ceil(np.log2(self.k)) + 1)
        elif self.qudit_mapping == 'ham':
            self.n = self.dim * self.k  #unsure check this
        else:
            raise KeyError('Define encoding type: "bin" or "ham"')
        self.N = 2 ** self.n
        self.lam = 1 / self.N

        # initialise a register
        psi = []
        for i in range(self.N):
            psi.append(basis(self.N, i))
        self.reg = sum(psi).unit()

    def SVPtoH(self, lattice):

        # graph hamiltonian
        if self.graph == 'FULL':  # (full graph)
            hg = self.lam * self.N * (qeye(self.N) - (self.reg * self.reg.dag()))
        elif self.graph == 'HYPER':  # (hypercube)
            x = []
            for i in range(self.N - 1):
                xi = np.zeros((self.N, self.N))
                xi[:i, :i] = qeye(i)
                xi[i:i+2, i:+2] = sigmax()
                xi[i+2:, i+2:] = qeye(self.N - i - 2)
                x.append(xi)
            hg = self.lam * (self.N*qeye(self.N) - Qobj(sum(x)))
        else:
            raise KeyError('Define graph type: FULL or HYPER')

        # problem hamiltonian
        gram = lattice @ lattice.T
        if self.qudit_mapping == 'bin':
            jmat, hvec, ic = isingify.svp_isingcoeffs_bin(gram, self.k)
        elif self.qudit_mapping == 'ham':
            jmat, hvec, ic = isingify.svp_isingcoeffs_ham(gram, self.k)
        else:
            raise KeyError('Define encoding type: "bin" or "ham"')
        hp = isingify.ising_hamiltonian(jmat, hvec, ic)

        # Full hamiltonian
        self.H = hp + hg
        # print('H', H)

    def execute(self):
        result = sesolve(self.H, self.reg, self.times, [sigmaz()], progress_bar=True)
        print('result', result)
        print('expect', result.expect)

    def run(self):
        # create the lattice
        lattice = np.random.randint(low=0, high=5, size=(self.dim, self.dim), dtype=int)
        self.SVPtoH(lattice)
        self.execute()


def main():
    # use sitek that specifies number of qubits
    dim = 3
    k = 4
    graph = 'FULL'
    qudit_mapping = 'bin'  #exclusively use bin for now
    times = np.linspace(0.0, 1.0, 2)
    experiment = SVPbyQuantumWalk(dim, k, times, qudit_mapping, graph)
    experiment.run()


if __name__ == "__main__":
    main()
