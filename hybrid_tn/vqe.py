import numpy as np
import itertools
from mindquantum import Hamiltonian
from mindquantum.core.operators import QubitOperator


import mindquantum
from mindquantum import Circuit, RY, RX, RZ
from mindquantum import X, Z, Y
# from mindquantum.core.gates import X, Z#, ZZ
import numpy as np
from mindquantum.core import ParameterResolver
import math
from mindquantum import Simulator, Measure

class Parameter_manager:
    def __init__(self, key='default'):
        self.parameters = []
        self.count = 0
        self.key = key
        self.grad_key = None
    
    def random_parameter_resolver(self):
        pr = {k:np.random.randn()*2*math.pi for k in self.parameters}
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


def test():
    circ = Circuit()
    circ += RY('ry').on(0)
    ham = Hamiltonian(QubitOperator(''))
    sim = Simulator('projectq', 3)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    e, g = grad_ops(np.array([0.3]))
    print(e)


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


def regular_block(C, n_qubits, P):
    for i in range(n_qubits):
        # C += RY(P.create()).on(i)
        RY_gate(C, i, P)
    for i in range(n_qubits-1):
        C += X.on(i, i+1)
        
def special_block_1(C, n_qubits, P):
    for i in range(n_qubits):
        # C += RX(P.create()).on(i)
        RX_gate(C, i, P)

# def RZZ(C, i, j, P):
#     C += X.on(j, i)
#     C += RZ(P.create()).on(j)
#     C += X.on(j, i)
    
    
def special_block_2(C, n_qubits, P):
    for i in range(n_qubits):
        RZ_gate(C, i, P)

    for i in range(n_qubits-1):
        RZZ_gate(C, i, i+1, P)

class QuantumTensor:
    def __init__(self):
        # self.P = Parameter_manager(count=count)
        
        self.special_block = {
            'first': special_block_1,
            'second': special_block_2
        }
        self.depth = {
            'first': 6,
            'second': 8
        }
    
    # def get_global_param_count(self):
    #     return self.P.count

    # def random_parameter_resolver(self):
        # return self.P.random_parameter_resolver()
        
    def layer(self, P, n_qubits=8, layer_type='first'):
        special_block = self.special_block[layer_type]
        d = self.depth[layer_type]
        
        C = Circuit()
        for i in range(d):
            regular_block(C, n_qubits, P)
            if i==0 or i==d//2: #first and d/2+1'th layer
                special_block(C, n_qubits, P)

        return C


class Hamiltonian_1D:
    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits
    
    def vac_Hamiltonian(self):
        ham = QubitOperator('')
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
        return ham



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


from mindquantum.core.parameterresolver import ParameterResolver as PR
class VQE_1D:
    def __init__(self, n_subsystems=3, n_qubits_per_subsystems=8, lr=0.1):
        self.n_qubits = n_subsystems * n_qubits_per_subsystems

        self.qt = QuantumTensor()
        ''' layer1 construction'''
        # count = self.layer1_qt.get_global_param_count()
        P = Parameter_manager(key='layer1')
        self.circ = self.qt.layer(P, n_qubits=self.n_qubits, layer_type='first')
        self.pr = P.random_parameter_resolver()
        P._replay()
        self.P = P
        self.H = Hamiltonian_1D(n_qubits=self.n_qubits)
        
        self.ham = Hamiltonian(self.H.local_Hamiltonian())
        self.vac_ham = Hamiltonian(self.H.vac_Hamiltonian())

        self.circuit_optimizer = Optimizer(learn_rate=lr)


    def expectation(self):
        sim = Simulator('projectq', self.n_qubits)
        sim.apply_circuit(self.circ, pr=self.pr)
        E = sim.get_expectation(self.ham)
        return E

    def norm(self):
        sim = Simulator('projectq', self.n_qubits)
        sim.apply_circuit(self.circ, pr=self.pr)
        E = sim.get_expectation(self.vac_ham)
        return E

    def pr2array(self):
        parameters = []
        k_list = []
        for k in self.pr.keys():
            k_list.append(k)
            parameters.append(self.pr[k])
        parameters = np.array(parameters)
        return parameters, k_list

    def array2pr(self, parameters, k_list):
        _pr = {}
        for k, p in zip(k_list, parameters.tolist()):
            _pr[k] = p
        pr = PR(_pr)
        self.pr = pr


    def gradient(self):
        sim = Simulator('projectq', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(self.ham, self.circ)
        parameters, k_list = self.pr2array()
        f, g = grad_ops(parameters)
        return g, parameters, k_list

    def step(self):
        g, parameters, k_list = self.gradient()
        g = g.squeeze(0).squeeze(0)
        parameters = self.circuit_optimizer.step(parameters, g).real
        self.array2pr(parameters, k_list)



if __name__ == '__main__':
    m = VQE_1D(n_subsystems=2, n_qubits_per_subsystems=3, lr=0.1)

    for i in range(10000):
        m.step()

        if i%20==0:
            E = m.expectation()
            norm = m.norm()
            print(E.real, norm.real, (E/norm).real)