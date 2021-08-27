import qutip
import numpy as np
import isingify


class SVPbyQuantumWalk:
    '''Thanks to Jake Lishman for the help with bugs'''
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

        # Initialise a register which has `n` qubits all in the |+> state.
        plus_state = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        self.reg = qutip.tensor([plus_state] * self.n)

    def SVPtoH(self, lattice):

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
        gram = lattice @ lattice.T
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
        # print('H', H)

    def execute(self):
        # This makes a Z.Z.Z.[...] operator, rather than just a single-qubit
        # operator.
        multi_z = qutip.tensor([qutip.sigmaz()] * self.n)
        result = qutip.sesolve(self.H, self.reg, self.times, e_ops=[multi_z], progress_bar=True)
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
    times = np.linspace(0.0, 1.0, 101)
    experiment = SVPbyQuantumWalk(dim, k, times, qudit_mapping, graph)
    experiment.run()


if __name__ == "__main__":
    main()
