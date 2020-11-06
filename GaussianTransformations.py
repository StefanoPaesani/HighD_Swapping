"""
Useful functions to efficiently simulate Gaussian states evolving through Gaussian transformations using the symplectic
formalism, and to import the evolved Gaussian states into Xanadu's StrawberryFields package.

All transformations used the same notation as in the StrawberryFields package, with a basis for the Gaussian
transformations and states given by the quadrature operators in the order (x_1, ..., x_N, p_1, ..., p_N).

Stefano Paesani, Nov. 2020
"""

# We are going to use Xanadu's StrawberryFields module to get photon-counting statistics from Gaussian states.
# The evolution of Gaussian states is pretty slow in StrawberryFields as they decompose the unitary into BSs
# ans phases with Clements decomposition which is pretty slow, so I am going to use our own code to simulate
# the evolution in the symplectic formalism, and then import it into StrawberryFields.
from strawberryfields.ops import S2gate
import strawberryfields as sf
# from strawberryfields.backends.gaussianbackend import *

import numpy as np

#####################################################################
### Functions for Gaussian operations in the symplectic formalism ###
#####################################################################

U_bs = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
U_phase = np.array([[1., 0.], [0., 1.j]])


def get_unitary_sym(U):
    ### Puts a linear-optical transformation in its symplectic form.
    return np.block([[np.real(U), -np.imag(U)], [np.imag(U), np.real(U)]])


def get_bs_sym(Mode1, Mode2, NumModes):
    ### Symplectic transformation associated to a balanced beam-splitter.
    tempMat = np.identity(NumModes)
    tempMat[np.ix_([Mode1, Mode2], [Mode1, Mode2])] = U_bs
    return get_unitary_sym(tempMat), tempMat


def get_tunable_bs_sym(refl, Mode1, Mode2, NumModes):
    ### Symplectic transformation associated to a beam-splitter with tunable reflectivity.
    tempMat = np.identity(NumModes)
    tempMat[Mode1, Mode1] = np.sqrt(1-refl)
    tempMat[Mode1, Mode2] = np.sqrt(refl)
    tempMat[Mode2, Mode1] = - np.sqrt(refl)
    tempMat[Mode2, Mode2] = np.sqrt(1-refl)
    return get_unitary_sym(tempMat), tempMat


def get_phase_sym(phi, Mode, NumModes):
    ### Symplectic transformation associated to a tunable phase-shifter.
    tempMat = np.identity(NumModes, dtype=np.csingle)
    tempMat[Mode, Mode] = np.exp(1.j * phi)
    return get_unitary_sym(tempMat), tempMat



def get_sms_sym(r, phi, Mode, NumModes):
    ### Symplectic transformation associated to single-mode squeezing.
    tempMat = np.identity(2 * NumModes)
    tempMat[Mode, Mode] = np.exp(-r)
    tempMat[NumModes + Mode, NumModes + Mode] = np.exp(r)
    tempMat = get_phase_sym(phi, Mode, NumModes)[0] @ tempMat
    return tempMat


def get_tms_sym(r, phi, Mode1, Mode2, NumModes):
    ### Symplectic transformation associated to two-mode squeezing. Uses the fact that two SMS states passing through a
    ### pi/2 phase and a BS gives rise to TMS.
    tempMat = get_sms_sym(r, 0, Mode1, NumModes)
    tempMat = get_sms_sym(r, 0, Mode2, NumModes) @ tempMat
    tempMat = (get_phase_sym(np.pi / 2., Mode2, NumModes)[0]) @ tempMat
    tempMat = (get_bs_sym(Mode1, Mode2, NumModes)[0]) @ tempMat
    tempMat = (get_phase_sym(phi, Mode2, NumModes)[0]) @ tempMat
    return tempMat


def symplectic_evolution(Cov0, SymplMatrix):
    ### evolves the covariance matrix through a Gaussian transformation
    return SymplMatrix @ Cov0 @ np.transpose(np.conjugate(SymplMatrix))


def loss_evolution(Cov0, transmission):
    ### adds losses. TODO: check if this is actually correct
    NumModes = int(len(Cov0) / 2.)
    return transmission * Cov0 + (1 - transmission) * np.identity(2 * NumModes, dtype=np.complex)


######################################################
### Mapping a Gaussian state into StrawberryFields ###
######################################################


def map_into_StrawberryFields(Displ, Cov, NumModes):
    ### Maps a Gaussian state into StrawberryFields
    sfBackend = sf.backends.gaussianbackend.GaussianBackend()
    sfBackend.begin_circuit(NumModes)
    Means = [2 * np.abs(alpha) * np.cos(np.angle(alpha)) for alpha in Displ] + \
            [2 * np.abs(alpha) * np.sin(np.angle(alpha)) for alpha in Displ]
    sfBackend.prepare_gaussian_state(Means, Cov, sfBackend.get_modes())
    return sfBackend.state()


##############################
### Other useful functions ###
##############################


def conf_to_Fock(Conf, NumModes):
    ### Converts a configuration into a Fock vector
    tempFock = np.zeros(NumModes)
    for x in Conf:
        tempFock[x] = tempFock[x] + 1
    return tempFock


def dB_to_Lin(dB):
    ### dB to linear scale converter
    return 10 ** (dB / 10)


def get_DFT(m):
    ### Obtains the m-mode Discrete Fourier Transform matrix
    return np.array([[np.exp(1.j*j*k*2*np.pi/m)/np.sqrt(m) for j in range(m)] for k in range(m)])


##########################################################################################
if __name__ == "__main__":
    s_par = 0.2
    num_modes = 2
    cov0 = np.identity(2 * num_modes)
    cov = cov0

    ### Tests with Stefano's evolution method
    cov_new = cov0
    cov_new = symplectic_evolution(cov_new, get_tms_sym(s_par, 0, 0, 1, num_modes))

    ##### tests with strawberryfield

    eng = sf.Engine('gaussian')

    ##### test TMS
    prog2 = sf.Program(num_modes)
    with prog2.context as q:
        S2 = S2gate(s_par)
        S2 | (q[0], q[1])
    state2 = eng.run(prog2).state
    # print(state2.means())
    print(state2.cov())

    mapped_state = map_into_StrawberryFields(np.zeros(num_modes), cov_new, num_modes)
    # print(mapped_state.means())
    print(mapped_state.cov())

    print(mapped_state == state2)
