import mindquantum.core.gates as G
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator
from mindquantum import Hamiltonian
import copy
import numpy as np


from hybrid_tn.faithful_gradients.helper import Gradient_test
from hybrid_tn.faithful_gradients.utils import pr2array, array2pr



def get_gradient_preconditional(gate):
    if isinstance(gate, G.RX):
        return G.X.on(gate.obj_qubits)
    elif isinstance(gate, G.RY):
        return G.Y.on(gate.obj_qubits)
    elif isinstance(gate, G.RZ):
        return G.Z.on(gate.obj_qubits)
    else:
        raise NotImplementedError()

J = np.complex(0,1)
def grad_circuit_symbolic_forward(circ):
    circ_list, circ_coeff_list = [], []
    for i, gate in enumerate(circ):
        if isinstance(gate, (G.RX, G.RY, G.RZ)):
            n_circ = copy.deepcopy(circ)
            n_circ.insert(i, get_gradient_preconditional(gate))
            circ_list.append(n_circ)
            circ_coeff_list.append(-1./2 * J) #for example, grad(RX) = -j X RX
    return circ_list, circ_coeff_list

import itertools
class Grad:
    def __init__(self, circ, pr, ham, n_qubits):
        self.circ, self.pr, self.ham = circ, pr, ham
        self.n_qubits = n_qubits
        self.parameters, self.k_list = pr2array(self.pr)
        self.circ_list, self.circ_coeff_list = grad_circuit_symbolic_forward(self.circ)

        assert len(self.circ_list)==len(self.k_list), '{} vs {}'.format(len(self.circ_list), len(self.k_list))

    def grad(self):
        r'''
        calculate gradient using forwardMode, while also calculate Hessian with hybridMode
        '''
        jac = np.zeros(len(self.parameters)).astype(np.complex)
        hess = np.zeros((len(self.parameters), len(self.parameters))).astype(np.complex)

        for i, (circ_right, coeff) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
            sim = Simulator('projectq', self.n_qubits)
            circ_right.no_grad()
            grad_ops = sim.get_expectation_with_grad(self.ham, circ_right, self.circ)
            e, g = grad_ops(self.parameters)
            jac[i] = e[0][0] * coeff #this is \partial E/ \partial circ_right
            hess[i] = g.squeeze() * coeff * J

        jac = jac * 2 #+ jac * J # add h.c.

        return jac, hess

    def Hess_forwardMode(self):
        r'''
        calculate Hessian using forward mode
        '''
        hess = np.zeros((len(self.parameters), len(self.parameters))).astype(np.complex)
        for i, (circ_left, coeff_left) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
            for j, (circ_right, coeff_right) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
                sim = Simulator('projectq', self.n_qubits)
                circ_right.no_grad()
                circ_left.no_grad()

                grad_ops = sim.get_expectation_with_grad(self.ham, circ_right, circ_left)
                e, g = grad_ops(self.parameters)
                hess[i][j] = e[0][0] * coeff_left * coeff_right * J

        return hess

    def grad_reserveMode(self):
        r'''
        test method that generate gradient using backpropogation(reverse mode differentiation)
        '''
        sim = Simulator('projectq', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(self.ham, self.circ, self.circ)
        e, g = grad_ops(self.parameters)
        return g.squeeze()


class FisherInformation:
    def __init__(self, circ, pr, n_qubits):
        ham = QubitOperator('')
        ham = Hamiltonian(ham)
        self.G = Grad(circ, pr, ham, n_qubits)

        self.circ, self.pr, self.ham = circ, pr, ham
        self.n_qubits = n_qubits

    def fisherInformation_hybridMode(self):
        jac, hess = self.G.grad()
        jac_left = np.expand_dims(jac, axis=1)
        jac_right = np.expand_dims(jac, axis=0) * J

        matrix = hess - (jac_left * jac_right)
        matrix = 4 * matrix.real
        return matrix

    # def fisherInformation_builtin(self):
    #     sim = Simulator('projectq', self.n_qubits)
    #     # matrix = sim.get_fisher_information_matrix(circ, self.G.parameters, diagonal=False)
    #     matrix = sim.fisher_information_matrix(circ.get_cpp_obj(),
    #                                               circ.get_cpp_obj(hermitian=True),
    #                                               self.G.parameters,
    #                                               circ.params_name,
    #                                               False)
    #     return matrix

if __name__ == '__main__':
    gt = Gradient_test(5)
    ham = gt.Ham.local_Hamiltonian()
    # circ = G.circ
    # grad_circuit_symbolic_forward(circ)

    gsfm = Grad(gt.circ, gt.pr, ham, gt.n_qubits)

    jac, hess = gsfm.grad()
    # hess_forward = gsfm.Hess_forwardMode()
    # g = gsfm.grad_reserveMode()

    # jac_helper, hess_helper = gt.gradient(ham)

    # print(jac - jac_helper)
    # print(jac - g)
    # print(jac)

    # print(hess - hess_forward)
    # print(hess_forward)

    F = FisherInformation(gt.circ, gt.pr, gt.n_qubits)
    fisher = F.fisherInformation_hybridMode()
    # fisher_builtin = F.fisherInformation_builtin()
    print(fisher)
    # print(fisher_builtin)