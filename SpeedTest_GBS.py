from GaussianTransformations import *

from threshold_detection import threshold_detection_prob, threshold_detection_prob_parallel

import numpy as np
from time import time
from scipy.stats import unitary_group


def CoherentGBS_threshold_speedtest(nmodes, det_modes, parallelized=False, num_reps=10, hbar=2):

    if num_reps > 1:
        start_time = time()
    for rep_ix in range(num_reps):
        ##############################
        ### DEFINE COHERENT STATES ###
        ##############################

        ## Defines the coherent state amplitudes in the input modes.
        ## Because no coherent state is present here, they are all zeros.
        ampls = np.array([np.random.rand() for i in range(nmodes)])

        ##############################
        ### DEFINE SQUEEZED STATES ###
        ##############################

        ## Initialises the initial covariance matrix of the Gaussian state to be the identity,
        ## meaning we start with vacuum in all modes.
        cov_mat = np.identity(2 * nmodes)

        ## Defines squeezers
        sms_sources_sym = np.identity(2 * nmodes)

        for i in range(nmodes):
            squeeze_par = np.random.rand()
            this_source = get_sms_sym(squeeze_par, phi=0, Mode=i, NumModes=nmodes)
            sms_sources_sym = sms_sources_sym @ this_source

        ############################
        ### DEFINE LINEAR OPTICS ###
        ############################

        ## Defines the multiport interferometer
        LO_unitary = unitary_group.rvs(nmodes)

        ## Gets the linear-optical transformation in the symplectic form.
        LO_unitary_sym = get_unitary_sym(LO_unitary)

        ###############################################
        ### CALCULATE TOTAL GAUSSIAN TRANSFORMATION ###
        ###############################################

        ## Obtains the total Gaussian transformation matrix in the symplectic formalism
        sym_transf = LO_unitary_sym @ sms_sources_sym

        #############################
        ### EVOLVE GAUSSIAN STATE ###
        #############################

        ## Obtains the covariance matrix of the output Gaussian state
        cov_mat = symplectic_evolution(cov_mat, sym_transf)

        ## Obtains the amplitudes of the output Gaussian state
        ampls = LO_unitary @ ampls
        means = np.concatenate((np.real(ampls), np.imag(ampls)), axis=0) * np.sqrt(2 * hbar)


        ###############################
        ### SINGLE PHOTON DETECTION ###
        ###############################

        ## Map state into the StrawberryField package, which has fast functions for
        ## photon counting (i.e. non-Gaussian measurements) from Gaussian states.
        gauss_state = map_into_StrawberryFields(ampls, cov_mat, nmodes)

        output_Fock = conf_to_Fock(det_modes, nmodes)

        if num_reps == 1:
            start_time = time()

        if parallelized:
            det_prob = threshold_detection_prob_parallel(cov_mat, means, output_Fock)
        else:
            det_prob = threshold_detection_prob(cov_mat, means, output_Fock)

    end_time = time()
    used_time = end_time - start_time

    return used_time/num_reps



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ###################################################################################
    ## Test 1: simple test with few modes - used to warm-up (compile) the numpa function
    ####################################################################################

    print("\nTest 1: simple test with few modes - warming-up numpa (needs to compile).")
    nphots = 4
    nmodes = 2*nphots #nphots**2

    det_modes = np.arange(nphots)

    # warm it up
    _ = CoherentGBS_threshold_speedtest(nmodes, det_modes, parallelized=False)
    _ = CoherentGBS_threshold_speedtest(nmodes, det_modes, parallelized=True)

    t_stand = CoherentGBS_threshold_speedtest(nmodes, det_modes, parallelized=False)
    print("Time used (standard):", t_stand)
    t_parallel = CoherentGBS_threshold_speedtest(nmodes, det_modes, parallelized=True)
    print("Time used (parallelized):", t_parallel)

    ###################################################################################
    ## Test 2: Scanning diffent numbers of photons
    ####################################################################################

    print("\nTest 2: Scanning diffent numbers of photons.")

    nphots_list = range(4, 25)

    t_stand_list = []
    t_parallel_list = []
    for nphots in nphots_list:
        print("nphots:", nphots)

        nmodes = 2*nphots #nphots**2
        det_modes = np.arange(nphots)

        t_stand = CoherentGBS_threshold_speedtest(nmodes, det_modes, parallelized=False)
        t_stand_list.append(t_stand)

        t_parallel = CoherentGBS_threshold_speedtest(nmodes, det_modes, parallelized=True)
        t_parallel_list.append(t_parallel)
        print('t_stand:', t_stand, 't_parallel:', t_parallel)

    plt.plot(nphots_list, t_stand_list, 'b', label="NewCode")
    plt.plot(nphots_list, t_parallel_list, 'k--', label="NewCode-Parall.")
    plt.xlabel(r'Photon number')
    plt.ylabel('Computation time (s)')
    plt.yscale('log')
    plt.legend()
    plt.show()
