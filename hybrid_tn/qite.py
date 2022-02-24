import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import mindquantum as mq
from mindquantum import Hamiltonian
# from mindquantum.core import X, RX
# from mindquantum.core import Circuit
from mindquantum import Circuit, RY, RX, RZ
from mindquantum import X, Z, Y

from mindquantum.algorithm.nisq.chem import generate_uccsd
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian
from mindquantum.core import ParameterResolver
from mindquantum.core.parameterresolver import ParameterResolver as PR
import math
from mindquantum.core.operators import QubitOperator
from timeit import default_timer

class Parameter_manager:
    def __init__(self, key='default'):
        self.parameters = []
        self.count = 0
        self.key = key
        self.grad_key = None
    
    def init_parameter_resolver(self):
        pr = {k:np.random.randn()*2*math.pi for k in self.parameters}
        # pr = {k:0 for k in self.parameters}
        pr = ParameterResolver(pr)
        return pr

    def _replay(self):
        self.count = 0

    def set_grad_key(self, key):
        self.grad_key = key
        self._replay()    

    def create(self):
        param = '{}_theta_{}'.format(self.key, self.count)
        self.count += 1
        self.parameters.append(param)
        if self.grad_key is None or param!=self.grad_key:
            is_grad = False
        else:
            is_grad = True
        return param, is_grad


def RY_gate(circ, i, P):
    ry, is_grad = P.create()
    if not is_grad:
        circ += RY(ry).on(i)
    else:
        circ += Y.on(i)
        circ += RY(ry).on(i)

def RX_gate(circ, i, P):
    rx, is_grad = P.create()
    if not is_grad:
        circ += RX(rx).on(i)
    else:
        circ += X.on(i)
        circ += RX(rx).on(i)

def RZ_gate(circ, i, P):
    rz, is_grad = P.create()
    if not is_grad:
        circ += RZ(rz).on(i)
    else:
        circ += Z.on(i)
        circ += RZ(rz).on(i)

def RZZ_gate(circ, i, j, P):
    circ += X.on(j, i)
    RZ_gate(circ, j, P)
    circ += X.on(j, i)

def RYY_gate(circ, i, j, P):
    circ += X.on(j, i)
    RY_gate(circ, j, P)
    circ += X.on(j, i)

def RXX_gate(circ, i, j, P):
    circ += X.on(j, i)
    RX_gate(circ, j, P)
    circ += X.on(j, i)


def layer(C, P, n_qubits):
    for i in range(n_qubits):
        RZ_gate(C, i, P)
        RY_gate(C, i, P)
        RX_gate(C, i, P)
        RZ_gate(C, i, P)

    for i in range(0, n_qubits-1, 2):
        RZZ_gate(C, i, i+1, P)
        RZZ_gate(C, i, i+1, P)
        RZZ_gate(C, i, i+1, P)

    for i in range(1, n_qubits-1, 2):
        RZZ_gate(C, i, i+1, P)
        RZZ_gate(C, i, i+1, P)
        RZZ_gate(C, i, i+1, P)


def get_system(key='LiH'):
    dist = 1.5
    if key=='LiH':
        geometry = [
            ["Li", [0.0, 0.0, 0.0 * dist]],
            ["H",  [0.0, 0.0, 1.0 * dist]],
        ]
    elif key=='H2':
        geometry = [
            ["H", [0.0, 0.0, 0.0 * dist]],
            ["H",  [0.0, 0.0, 1.0 * dist]],
        ]


    basis = "sto3g"
    spin = 0
    # print("Geometry: \n", geometry)

    molecule_of = MolecularData(
        geometry,
        basis,
        multiplicity=2 * spin + 1
    )
    molecule_of = run_pyscf(
        molecule_of,
        run_scf=1,
        run_ccsd=0,
        run_fci=1
    )

    # print("Hartree-Fock energy: %20.16f Ha" % (molecule_of.hf_energy))
    # molecule_of.save()
    # molecule_file = molecule_of.filename
    # print(molecule_file)

    hamiltonian_QubitOp = get_qubit_hamiltonian(molecule_of)
    # print(hamiltonian_QubitOp)
    # print(molecule_of.n_qubits)
    return molecule_of, Hamiltonian(hamiltonian_QubitOp)


class Optimizer:
    def __init__(self, learn_rate=10, decay_rate=0.01):
        self.diff = np.zeros(1).astype(np.float32)
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        
    def step(self, vector, grad):
        # Performing the gradient descent loop
        vector = vector.astype(np.float32)
        # grad = grad.squeeze(0).squeeze(0)
        grad = grad.astype(np.float32)

        self.diff = self.decay_rate * self.diff - self.learn_rate * grad
        vector += self.diff
        return vector


class VQE:
    def __init__(self, key='LiH'):
        molecule_of, self.ham = get_system(key=key)
        self.n_qubits = molecule_of.n_qubits
        self.fci_energy = molecule_of.fci_energy

        self.P = Parameter_manager()

        self.circ = Circuit()
        layer(self.circ, self.P, self.n_qubits)
        self.pr = self.P.init_parameter_resolver()

        self.optimizer = Optimizer(learn_rate=1.0)

    def pr2array(self, pr):
        parameters = []
        k_list = []
        for k in pr.keys():
            k_list.append(k)
            parameters.append(pr[k])

        parameters = np.array(parameters)
        return parameters, k_list

    def array2pr(self, parameters, k_list):
        _pr = {}
        for k, p in zip(k_list, parameters.tolist()):
            _pr[k] = p
        pr = PR(_pr)
        return pr

    def gradient_descent_step(self):
        parameters, k_list = self.pr2array(self.pr)

        sim = Simulator('projectq', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(self.ham, self.circ)
        f, g = grad_ops(parameters)

        g = g.squeeze(0).squeeze(0)

        parameters = self.optimizer.step(parameters, g).real
        self.pr = self.array2pr(parameters, k_list)


    def manual_preconditional_forwardMode(self, parameters, k_list):
        ham = Hamiltonian(QubitOperator(''))
        sim = Simulator('projectq', self.n_qubits)
        hessian = np.zeros((len(k_list), len(k_list)), dtype=np.complex128)
        for i, ki in enumerate(k_list):
            for j, kj in enumerate(k_list):

                self.P.set_grad_key(ki)
                circ_left = Circuit()
                layer(circ_left, self.P, self.n_qubits)

                self.P.set_grad_key(kj)
                circ_right = Circuit()
                layer(circ_right, self.P, self.n_qubits)

                grad_ops = sim.get_expectation_with_grad(ham, circ_right, circ_left)
                e, _ = grad_ops(parameters)
                hessian[i,j] = e[0][0]

        hessian = hessian * (-1) * (1/4) # this is (-i/2 <0|U)(-i/2 V|0>)
        return hessian.real


    def manual_preconditional_hybridMode(self, parameters, k_list):
        r'''
        use hybrid mode for calculation of Hessian
        '''
        ham = Hamiltonian(QubitOperator(''))
        sim = Simulator('projectq', self.n_qubits)
        hessian = np.zeros((len(k_list), len(k_list)), dtype=np.complex128)
        for i, ki in enumerate(k_list):

            self.P.set_grad_key(ki)
            circ_left = Circuit()
            layer(circ_left, self.P, self.n_qubits)
            circ_left.no_grad()

            self.P.set_grad_key('')
            circ_right = Circuit()
            layer(circ_right, self.P, self.n_qubits)

            grad_ops = sim.get_expectation_with_grad(ham, circ_right, circ_left)
            _, g = grad_ops(parameters)
            hessian[i] = g.squeeze()

        hessian = hessian.imag * (-1/2) # this is (-i/2 <0|U)(-i/2 V|0>)
        return hessian


    def imaginary_time_evolution_step(self, use_hybrid_mode=True):
        parameters, k_list = self.pr2array(self.pr)

        sim = Simulator('projectq', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(self.ham, self.circ)
        f, g = grad_ops(parameters)

        if use_hybrid_mode:
            preconditional = self.manual_preconditional_hybridMode(parameters, k_list)
        else:
            preconditional = self.manual_preconditional_forwardMode(parameters, k_list)

        g = g.squeeze(0).dot(preconditional).squeeze(0) * (-1)

        parameters = self.optimizer.step(parameters, g).real
        self.pr = self.array2pr(parameters, k_list)



    def eval(self):
        sim = Simulator('projectq', self.n_qubits)
        sim.apply_circuit(self.circ, pr=self.pr)
        E = sim.get_expectation(self.ham)
        return E


if __name__ == '__main__':
    V = VQE(key='H2')
    for i in range(100):

        # V.gradient_descent_step()
        V.imaginary_time_evolution_step()

        E = V.eval()
        print(E, V.fci_energy)