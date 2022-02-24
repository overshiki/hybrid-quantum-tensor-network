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
    
    def local_Hamiltonian_with_pauli(self, f_first, f_last, pauli_type='X'):
        if pauli_type=='I':
            ham = self._local_Hamiltonian(f_first, f_last)
        else:
            ham = self._local_Hamiltonian_with_pauli(f_first, f_last, pauli_type)
        ham = Hamiltonian(ham)
        return ham
    
    def vac_Hamiltonian_with_pauli(self, pauli_type='X'):
        if pauli_type=='I':
            ham = QubitOperator('')
        else:
            ham = QubitOperator('{}{}'.format(pauli_type, self.n_qubits))
        ham = Hamiltonian(ham)
        return ham

    def _local_Hamiltonian(self, f_first, f_last):
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

        i, j = 0, self.n_qubits-1
        ham += QubitOperator('Z{}'.format(i), f_first)
        ham += QubitOperator('Z{}'.format(j), f_last)
        
        return ham

    
    def _local_Hamiltonian_with_pauli(self, f_first, f_last, pauli_type='X'):
        ham = None
        g, h = 0.5, 0.32
        for i in range(self.n_qubits):
            if ham is None:
                ham = QubitOperator('X{} {}{}'.format(i, pauli_type, self.n_qubits), g)
            else:
                ham += QubitOperator('X{} {}{}'.format(i, pauli_type, self.n_qubits), g)
                
            ham += QubitOperator('Z{} {}{}'.format(i, pauli_type, self.n_qubits), h)
            if i<self.n_qubits-1:
                ham += QubitOperator('Z{} Z{} {}{}'.format(i, i+1, pauli_type, self.n_qubits), 1)
                
        i, j = 0, self.n_qubits-1
        ham += QubitOperator('Z{} {}{}'.format(i, pauli_type, self.n_qubits), f_first)
        ham += QubitOperator('Z{} {}{}'.format(j, pauli_type, self.n_qubits), f_last)
    
        return ham


class QubitOperator_helper:
    def __init__(self):
        self.coeff = {'I': 0.5, 'X': 0.5, 'Y': -0.5, 'Z': 0.5}
        self.verbose_dict = {0:{'I':1,'X':1,'Y':1,'Z':1}}

    def get_coeff(self):
        return self.coeff
    
    def initialize_single(self, ei, ex, ey, ez):
        self._dict = {'I': self.coeff['I']*ei, 'X': self.coeff['X']*ex, 'Y': self.coeff['Y']*ey, 'Z': self.coeff['Z']*ez }
        self.verbose_dict = {0:{'I':ei, 'X':ex, 'Y':ey, 'Z':ez}}
        
    def tensor_product(self, vqo):
        _dict = {}
        for ki, kj in itertools.product(self._dict.keys(), vqo._dict.keys()):
            k = ki+kj
            v = self._dict[ki]*vqo._dict[kj]
            _dict[k] = v
        self._dict = _dict

        assert len(vqo.verbose_dict)==1
        append_index = len(self.verbose_dict)
        self.verbose_dict[append_index] = vqo.verbose_dict[0]

        return self
    
    
    def gradient(self, qubit_index, key='I'):
        _dict = {}
        for k in self._dict.keys():
            if k[qubit_index]==key:
                # we need to / self.verbose_dict[qubit_index][key] because we do not want \alpha to appear
                _dict[k] = self._dict[k] / self.verbose_dict[qubit_index][key]

        return self._to_QubitOperator(_dict)
    
    
    def to_QubitOperator(self):
        return self._to_QubitOperator(self._dict)
    
    def _to_QubitOperator(self, _dict):
        qo = None
        for k, v in _dict.items():
            S = ''
            for i, s in enumerate(k):
                if s!='I':
                    S += '{}{} '.format(s, i)
            
            if qo is None:
                qo = v * QubitOperator(S)
            else:
                qo += v * QubitOperator(S)
        return qo



class VQE_1D:
    def __init__(self, n_subsystems=3, n_qubits_per_subsystems=8):
        self.interaction_coeff = np.random.rand(n_subsystems-1).tolist()
        #we pad 0 around random array to indicate both end of the systems are free of interaction
        interaction_coeff = [0, ] + self.interaction_coeff + [0, ]
        self.boundary_pairs = []
        for i in range(len(interaction_coeff)-1):
            self.boundary_pairs.append([interaction_coeff[i], interaction_coeff[i+1]])
            
        self.n_subsystems = n_subsystems
        self.n_qubits_per_subsystems = n_qubits_per_subsystems

        self.qt = QuantumTensor()
        ''' layer1 construction'''
        # count = self.layer1_qt.get_global_param_count()
        P = Parameter_manager(key='layer1')
        self.layer1_circ = self.qt.layer(P, n_qubits=n_subsystems, layer_type='first')
        self.layer1_pr = P.random_parameter_resolver()
        P._replay()
        self.layer1_P = P
        
        
        ''' layer2 construction'''
        self.subsystem_dict = {}
        self.subsystem_P = {}
        for i in range(n_subsystems):
            # qt = QuantumTensor(count=count)
            # count = qt.get_global_param_count()
            P = Parameter_manager(key='layer2_{}'.format(i))
            circ = self.qt.layer(P, n_qubits=n_qubits_per_subsystems+1, layer_type='second')
            pr = P.random_parameter_resolver()
            P._replay()

            self.subsystem_dict[i] = (circ, pr)
            self.subsystem_P[i] = P



    def norm(self):
        # layer1_operator is resulted from measurement of layer2
        qo = self.vac_layer1_operator().to_QubitOperator()
        qo = qo.real #convert QubitOperator to its real part, since it is constructed using previous measure, there exist small imaginary part, due to numerical issue.
        ham = Hamiltonian(qo)
        sim = Simulator('projectq', self.n_subsystems)
        sim.apply_circuit(self.layer1_circ, pr=self.layer1_pr)

        E = sim.get_expectation(ham)
        return E        


    def layer1_expectation(self):
        # layer1_operator is resulted from measurement of layer2
        qo = self.layer1_operator().to_QubitOperator()
        qo = qo.real #convert QubitOperator to its real part, since it is constructed using previous measure, there exist small imaginary part, due to numerical issue.
        ham = Hamiltonian(qo)
        sim = Simulator('projectq', self.n_subsystems)
        sim.apply_circuit(self.layer1_circ, pr=self.layer1_pr)

        # qs = sim.get_qs()
        # norm = qs.dot(np.conj(qs))
        # print('norm', norm)

        E = sim.get_expectation(ham)
        return E
            
    def layer1_operator(self):
        n_subsystems = self.n_subsystems
        V = None
        for i in range(n_subsystems):

            expectation = self.layer_operator_component(i)

            nv = QubitOperator_helper()
            nv.initialize_single(*[expectation[x] for x in ['I', 'X', 'Y', 'Z']])
            if V is None:
                V = nv
            else:
                V = V.tensor_product(nv)
        return V
            
    
    def layer_operator_component(self, subsystem_index):
        circ, pr = self.subsystem_dict[subsystem_index]
        boundary_condition = self.boundary_pairs[subsystem_index]

        n_qubits = self.n_qubits_per_subsystems
        H = Hamiltonian_1D(n_qubits=n_qubits)

        expectation = {}
        for pauli_key in ['I', 'X', 'Y', 'Z']:
            sim = Simulator('projectq', n_qubits+1)        
            sim.apply_circuit(circ, pr=pr)

            f_first, f_last = boundary_condition
            ham = H.local_Hamiltonian_with_pauli(f_first, f_last, pauli_key)

            E = sim.get_expectation(ham).real
            expectation[pauli_key] = E
            
        return expectation


    def vac_layer1_operator(self):
        n_subsystems = self.n_subsystems
        V = None
        for i in range(n_subsystems):

            expectation = self.vac_layer_operator_component(i)

            nv = QubitOperator_helper()
            nv.initialize_single(*[expectation[x] for x in ['I', 'X', 'Y', 'Z']])
            if V is None:
                V = nv
            else:
                V = V.tensor_product(nv)
        return V


    def vac_layer_operator_component(self, subsystem_index):
        circ, pr = self.subsystem_dict[subsystem_index]
        # boundary_condition = self.boundary_pairs[subsystem_index]

        n_qubits = self.n_qubits_per_subsystems
        H = Hamiltonian_1D(n_qubits=n_qubits)

        expectation = {}
        for pauli_key in ['I', 'X', 'Y', 'Z']:
            sim = Simulator('projectq', n_qubits+1)        
            sim.apply_circuit(circ, pr=pr)

            # f_first, f_last = boundary_condition
            # ham = H.local_Hamiltonian_with_pauli(f_first, f_last, pauli_key)
            ham = H.vac_Hamiltonian_with_pauli(pauli_type=pauli_key)

            E = sim.get_expectation(ham).real
            expectation[pauli_key] = E
            
        return expectation




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
class OPT_VQE1D(VQE_1D):
    def __init__(self, n_subsystems=3, n_qubits_per_subsystems=8, lr=0.1):
        super().__init__(n_subsystems=n_subsystems, n_qubits_per_subsystems=n_qubits_per_subsystems)
        self.layer1_circuit_optimizer = Optimizer(learn_rate=lr)
        
        self.layer2_circuit_optimizer = {}
        for i in range(self.n_subsystems):
            self.layer2_circuit_optimizer[i] = Optimizer(learn_rate=lr)
    
        self.layer1_gradient = {}


    def clear_layer1_gradient(self):
        self.layer1_gradient = {}

    def update_layer1_gradient(self, qubit_index, pauli_key):
        n_subsystems = self.n_subsystems

        qo = self.layer1_gradient_operator(qubit_index, key=pauli_key)
        qo = qo.real #convert QubitOperator to its real part, since it is constructed using previous measure, there exist small imaginary part, due to numerical issue.
        ham = Hamiltonian(qo)
        
        # get expectation with respect to gradient operator will give us gradient we need
        sim = Simulator('projectq', n_subsystems)
        sim.apply_circuit(self.layer1_circ, pr=self.layer1_pr)
        E = sim.get_expectation(ham)
        
        self.layer1_gradient[(qubit_index, pauli_key)] = E #* QubitOperator_helper().get_coeff()[pauli_key]


    def layer1_gradient_operator(self, qubit_index, key='I'):
        # layer1_operator is resulted from measurement of layer2
        V = self.layer1_operator()
        qo = V.gradient(qubit_index, key=key)
        return qo



    def _manual_layer2_preconditional(self, ham, parameters, k_list, P):
        ham = Hamiltonian(QubitOperator(''))
        sim = Simulator('projectq', self.n_qubits_per_subsystems+1)
        hessian = np.zeros((len(k_list), len(k_list)), dtype=np.complex128)
        for i, ki in enumerate(k_list):
            for j, kj in enumerate(k_list):

                P.set_grad_key(ki)
                circ_left = self.qt.layer(P, n_qubits=self.n_qubits_per_subsystems+1, layer_type='second')

                P.set_grad_key(kj)
                circ_right = self.qt.layer(P, n_qubits=self.n_qubits_per_subsystems+1, layer_type='second')

                grad_ops = sim.get_expectation_with_grad(ham, circ_right, circ_left)
                e, _ = grad_ops(parameters)
                # print(ki, kj, e)
                hessian[i,j] = e[0][0]

        hessian = hessian * (-1) * (1/4) # this is (-i/2 <0|U)(-i/2 V|0>)
        return hessian.real


    def manual_layer2_preconditional(self, ham, parameters, k_list, P):
        ham = Hamiltonian(QubitOperator(''))
        sim = Simulator('projectq', self.n_qubits_per_subsystems+1)
        hessian = np.zeros((len(k_list), len(k_list)), dtype=np.complex128)
        for i, ki in enumerate(k_list):
            # for j, kj in enumerate(k_list):

            P.set_grad_key(ki)
            circ_left = self.qt.layer(P, n_qubits=self.n_qubits_per_subsystems+1, layer_type='second')
            circ_left.no_grad()

            P.set_grad_key('')
            circ_right = self.qt.layer(P, n_qubits=self.n_qubits_per_subsystems+1, layer_type='second')

            # circ_right.no_grad()

            grad_ops = sim.get_expectation_with_grad(ham, circ_right, circ_left)
            e, g = grad_ops(parameters)
            # print(ki, kj, e)
            hessian[i] = g.squeeze()

        # hessian = hessian * (-1) * (1/4) # this is (-i/2 <0|U)(-i/2 V|0>)
        hessian = hessian.imag * (1/2) # this is (-i/2 <0|U)(-i/2 V|0>)
        return hessian


    def layer_operator_component_optimization(self, subsystem_index):
        r'''
        gradient down to the second layer
        '''
        circ, pr = self.subsystem_dict[subsystem_index]
        boundary_condition = self.boundary_pairs[subsystem_index]

        # n_qubits = self.n_qubits_per_subsystems
        H = Hamiltonian_1D(n_qubits=self.n_qubits_per_subsystems)
        sim = Simulator('projectq', self.n_qubits_per_subsystems+1)      
        P = self.subsystem_P[subsystem_index]


        f_first, f_last = boundary_condition

        # get parameters from pr
        parameters = []
        k_list = []
        for k in pr.keys():
            k_list.append(k)
            parameters.append(pr[k])

        parameters = np.array(parameters)

        grads = 0

        for pauli_key in ['I', 'X', 'Y', 'Z']:
            # root_gradient for the backpropogation
            root_gradient = self.layer1_gradient[(subsystem_index, pauli_key)]

            ham = H.local_Hamiltonian_with_pauli(f_first, f_last, pauli_key)

            # sim.apply_circuit(circ, pr=pr)
            grad_ops = sim.get_expectation_with_grad(ham, circ)
            f, g = grad_ops(parameters)
            # g = g.squeeze(0).squeeze(0)
            
            preconditional = self.manual_layer2_preconditional(ham, parameters, k_list, P)
            g = g.squeeze(0).dot(preconditional).squeeze(0) * (-1)

            g = g*root_gradient

            grads += g


        parameters = self.layer2_circuit_optimizer[subsystem_index].step(parameters, g).real

        # push parameters back to pr
        _pr = {}
        for k, p in zip(k_list, parameters.tolist()):
            _pr[k] = p
        pr = PR(_pr)

        self.subsystem_dict[subsystem_index] = (circ, pr)
                   
    
    
    def step4layer2circuit(self):
        self.clear_layer1_gradient()
        
        for qubit_index in range(self.n_subsystems):
            for pauli_key in ['I', 'X', 'Y', 'Z']:
                self.update_layer1_gradient(qubit_index, pauli_key)
        for subsystem_index in range(self.n_subsystems):
            self.layer_operator_component_optimization(subsystem_index)
    

    def get_pr(self):
        _dict = {}
        for key in self.subsystem_dict.keys():
            _, pr = self.subsystem_dict[key]
            _dict[key] = pr
        return self.layer1_pr, _dict


    def auto_layer1_gradient(self, ham, parameters):
        sim = Simulator('projectq', self.n_subsystems)
        # sim.apply_circuit(self.layer1_circ, pr=self.layer1_pr)
        grad_ops = sim.get_expectation_with_grad(ham, self.layer1_circ)
        _, g = grad_ops(parameters)
        return g

    def _manual_layer1_preconditional(self, ham, parameters, k_list):
        ham = Hamiltonian(QubitOperator(''))
        sim = Simulator('projectq', self.n_subsystems)
        hessian = np.zeros((len(k_list), len(k_list)), dtype=np.complex128)
        for i, ki in enumerate(k_list):
            for j, kj in enumerate(k_list):

                self.layer1_P.set_grad_key(ki)
                layer1_circ_left = self.qt.layer(self.layer1_P, n_qubits=self.n_subsystems, layer_type='first')

                self.layer1_P.set_grad_key(kj)
                layer1_circ_right = self.qt.layer(self.layer1_P, n_qubits=self.n_subsystems, layer_type='first')

                grad_ops = sim.get_expectation_with_grad(ham, layer1_circ_right, layer1_circ_left)
                e, _ = grad_ops(parameters)
                # print(ki, kj, e)
                hessian[i,j] = e[0][0]

        hessian = hessian * (-1) * (1/4) # this is (-i/2 <0|U)(-i/2 V|0>)
        return hessian.real


    def manual_layer1_preconditional(self, ham, parameters, k_list):
        ham = Hamiltonian(QubitOperator(''))
        sim = Simulator('projectq', self.n_subsystems)
        hessian = np.zeros((len(k_list), len(k_list)), dtype=np.complex128)
        for i, ki in enumerate(k_list):
            # for j, kj in enumerate(k_list):

            self.layer1_P.set_grad_key(ki)
            layer1_circ_left = self.qt.layer(self.layer1_P, n_qubits=self.n_subsystems, layer_type='first')
            layer1_circ_left.no_grad()

            self.layer1_P.set_grad_key('')
            layer1_circ_right = self.qt.layer(self.layer1_P, n_qubits=self.n_subsystems, layer_type='first')

            # layer1_circ_left.no_grad()
            # layer1_circ_right.no_grad()

            grad_ops = sim.get_expectation_with_grad(ham, layer1_circ_right, layer1_circ_left)
            e, g = grad_ops(parameters)
            # print(ki, kj, e)
            hessian[i] = g.squeeze()

        # hessian = hessian * (-1) * (1/4) # this is (-i/2 <0|U)(-i/2 V|0>)
        hessian = hessian.imag * (1/2) # this is (-i/2 <0|U)(-i/2 V|0>)
        return hessian


    def step4layer1circuit(self):

        qo = self.layer1_operator().to_QubitOperator()
        qo = qo.real #convert QubitOperator to its real part, since it is constructed using previous measure, there exist small imaginary part, due to numerical issue.
        ham = Hamiltonian(qo)

        # get parameters from pr
        parameters = []
        k_list = []
        for k in self.layer1_pr.keys():
            k_list.append(k)
            parameters.append(self.layer1_pr[k])

        parameters = np.array(parameters)

        # sim = Simulator('projectq', n_subsystems)
        # grad_ops = sim.get_expectation_with_grad(ham, self.layer1_circ)
        # f, g = grad_ops(parameters)
        g = self.auto_layer1_gradient(ham, parameters)
        # grads = self.manual_layer1_gradient(ham)

        # g = g.squeeze(0).squeeze(0)

        preconditional = self.manual_layer1_preconditional(ham, parameters, k_list)
        g = g.squeeze(0).dot(preconditional).squeeze(0) * (-1)

        # print(g.shape, hess.shape)
        # raise ValueError()
        
        parameters = self.layer1_circuit_optimizer.step(parameters, g).real
        
        # push parameters back to pr
        pr = {}
        for k, p in zip(k_list, parameters.tolist()):
            pr[k] = p
        
        self.layer1_pr = PR(pr)


if __name__ == '__main__':
    # from hqt_qtn import VQE_1D as qtn_VQE_1D

    n_subsystems=2
    n_qubits_per_subsystems=3
    m = OPT_VQE1D(n_subsystems=n_subsystems, n_qubits_per_subsystems=n_qubits_per_subsystems, lr=0.05)

    for i in range(10000):
        # for _ in range(10):
        m.step4layer1circuit()
        m.step4layer2circuit()

        
        if i%2==0:
            E = m.layer1_expectation()
            norm = m.norm()
            print(E.real, norm.real, (E/norm).real, (E/norm).real/(n_subsystems*n_qubits_per_subsystems))

        # if i%500==499:
        #     parameters = m.get_pr()
        #     v = qtn_VQE_1D(n_subsystems=n_subsystems, n_qubits_per_subsystems=n_qubits_per_subsystems, backend='auto', parameters=parameters, interaction_coeff=m.interaction_coeff)
        #     E = m.layer1_expectation()
        #     print(E)
        #     E = v.layer1_expectation()
        #     print(E)

        #     raise ValueError()