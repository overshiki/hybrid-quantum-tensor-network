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


class Ising_like_ham:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def vac_Hamiltonian(self):
        ham = QubitOperator('')
        ham = Hamiltonian(ham)
        return ham

    def local_Hamiltonian(self):
        ham = None
        g, h = 0.5, 0.32
        for i in range(self.n_qubits):
            if ham is None:
                ham = QubitOperator('X{}'.format(i), g)
            else:
                ham += QubitOperator('X{}'.format(i), g)
                
            ham += QubitOperator('Z{}'.format(i), h)
            if i<self.n_qubits-1:
                ham += QubitOperator('Z{} Z{}'.format(i, i+1), 1)

        ham = Hamiltonian(ham)
        return ham


from hybrid_tn.faithful_gradients.utils import pr2array
J = np.complex(0,1)
class Gradient_test:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.P = Parameter_manager()
        self.circ = Circuit()
        layer(self.circ, self.P, self.n_qubits)
        self.pr = self.P.init_parameter_resolver()
        # _, self.k_list = pr2array(self.pr)
        self.Ham = Ising_like_ham(n_qubits)


    def gradient(self, ham):
        coeff = (-1./2) * J
        parameters, k_list = pr2array(self.pr)
        jac = np.zeros(len(parameters)).astype(np.complex)
        hess = np.zeros((len(parameters), len(parameters))).astype(np.complex)

        for i, ki in enumerate(k_list):

            self.P.set_grad_key(ki)
            circ_right = Circuit()
            layer(circ_right, self.P, self.n_qubits)

            sim = Simulator('projectq', self.n_qubits)
            grad_ops = sim.get_expectation_with_grad(ham, circ_right, self.circ)
            e, g = grad_ops(parameters)

            jac[i] = e[0][0] * coeff #this is \partial E/ \partial circ_right
            hess[i] = g.squeeze() * coeff

        jac = jac * 2#+ jac * J # add h.c.
        return jac, hess