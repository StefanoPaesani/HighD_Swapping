from GaussianTransformations import *

from threshold_detection import threshold_detection_prob, threshold_detection_prob_parallel

import numpy as np
from itertools import product


def dft(modes, num_all_modes):
    """
    generates the unitary discrete fourier transform matrix for a subset ('modes) of modes over 'num_all_modes' overall
    number of modes.
    """
    n = len(modes)
    dft_U = (1 / np.sqrt(n)) * np.fft.fft(np.eye(n, dtype=complex))
    U_tot = np.identity(num_all_modes, dtype=np.cdouble)
    U_tot[np.ix_(modes, modes)] = dft_U
    return U_tot


def state_preparation_unitary(modes, state, num_all_modes, chosen_inmode=0):
    """
    Unitary that, given an input photon in the mode modes[chosen_inmode], prepares it in the state 'state'
    over the modes 'modes'.
    """
    n = len(modes)
    U_state = np.identity(n, dtype=np.cdouble)
    state_vec = np.array(state, dtype=np.cdouble)
    if max(state_vec) == 0:
        raise ValueError("State must have at least one non-zero element.")
    state_vec = state_vec / np.linalg.norm(state_vec)

    new_col = U_state[:, chosen_inmode]
    inner_prod = new_col.conj() @ state_vec
    if np.abs(inner_prod) == 0:
        non_zero_ix = np.where(state_vec != 0)[0][0]
        target_col = copy(U_state[:, non_zero_ix])
        U_state[:, non_zero_ix] = new_col
        U_state[:, chosen_inmode] = target_col

    for i in range(n):
        if i != chosen_inmode:
            new_col = U_state[:, i]
            inner_prod = new_col.conj() @ state_vec

            if np.abs(inner_prod) == 1.:
                U_state[:, i] = U_state[:, chosen_inmode]
                break
            new_col = new_col - inner_prod.conj() * state_vec
            for j in range(i):
                if j != chosen_inmode:
                    this_col = U_state[:, j]
                    inner_prod = new_col.conj() @ this_col
                    new_col = new_col - inner_prod.conj() * this_col
            new_col = new_col / np.linalg.norm(new_col)
            U_state[:, i] = new_col

    U_state[:, chosen_inmode] = state_vec

    U_tot = np.identity(num_all_modes, dtype=np.cdouble)
    U_tot[np.ix_(modes, modes)] = U_state
    return U_tot


def state_projection_unitary(modes, state, num_all_modes, chosen_outmode=0):
    """
    Unitary that, given a photon encoding a qudit in modes 'modes', performs a prejective measurement on the state
    'state' mapped into the output mode modes[chosen_outmode].
    """
    temp_U = state_preparation_unitary(modes, state, num_all_modes, chosen_inmode=chosen_outmode)
    return temp_U.T.conj()


######################################################################################################
ancilla_types_list = ['WeakCoherent', 'TMS', 'HeraldedPhoton']

def HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern, U_out, U_tilde,
                                        ancillas_param=0.1, s_par_photons=0.01, ancilla_type='WC',
                                        number_resolving_det=False, parallelized=False):
    """
    Function that simulates the high-dimensional teleportation protocol using Gaussian states for ancillas, and
    threshold detectors.

    :return: The probability of the projective measurement given the teleported state and the heralding pattern.
    """

    #####################
    ### INITALISATION ###
    #####################

    if ancilla_type not in ancilla_types_list:
        raise ValueError('Ancilla_type must be one of:', ancilla_types_list)

    nmodes = dim ** 2 + 3 * dim - 1

    signal_alice_modes = list(range(dim))
    # print('signal_alice_modes', signal_alice_modes)
    idler_alice_modes = list(range(dim, 2 * dim))
    # print('idler_alice_modes', idler_alice_modes)
    heralding_ancilla_modes = list(range(2 * dim, 3 * dim - 2))
    # print('heralding_ancilla_modes', heralding_ancilla_modes)
    ancilla_modes = [list(range((i + 2) * dim + 1, (i + 3) * dim + 1)) for i in range(dim - 2)]
    # print('ancilla_modes', ancilla_modes)
    all_ancilla_modes = [mode for submodes in ancilla_modes for mode in submodes]
    # print('all_ancilla_modes', all_ancilla_modes)
    idler_bob_modes = list(range(dim ** 2 + dim - 2, dim ** 2 + 2 * dim - 2))
    # print('idler_bob_modes', idler_bob_modes)
    signal_bob_modes = list(range(dim ** 2 + 2 * dim - 2, dim ** 2 + 3 * dim - 2))
    # print('signal_bob_modes', signal_bob_modes)
    extra_Utilde_modes = [dim ** 2 + 3 * dim - 2]
    # print('extra_Utilde_modes', extra_Utilde_modes)

    multiport_modes = idler_alice_modes + all_ancilla_modes + idler_bob_modes

    init_mode_ix_ancillas = 0
    init_anc_modes = [these_modes[init_mode_ix_ancillas] for these_modes in ancilla_modes]

    ## Initialises the initial covariance matrix of the Gaussian state to be the identity,
    ## meaning we start with vacuum in all modes.
    cov_mat = np.identity(2 * nmodes)

    ##############################
    ### DEFINE COHERENT STATES ###
    ##############################

    ## Defines the coherent state amplitudes in the input modes.
    ## Because no coherent state is present here, they are all zeros.
    ampls = np.zeros(nmodes)
    if ancilla_type == 'WeakCoherent':
        ampls[init_anc_modes] = np.ones(len(init_anc_modes)) * ancillas_param
        herald_ancilla_pattern = []
    # print('ampls0:', ampls)

    ###########################
    ### DEFINE SPDC SOURCES ###
    ###########################

    ## Defines the second SPDC source that generate the high-dimensional Bell state for Alice.
    ent_sources_alice = np.identity(2 * nmodes)
    for i in range(dim):
        this_source = get_tms_sym(s_par_photons, phi=0, Mode1=signal_alice_modes[i], Mode2=idler_alice_modes[i],
                                  NumModes=nmodes)
        ent_sources_alice = ent_sources_alice @ this_source

    ## Defines the second SPDC source that generate the high-dimensional Bell state for Bob.
    ent_sources_bob = np.identity(2 * nmodes)
    for i in range(dim):
        this_source = get_tms_sym(s_par_photons, phi=0, Mode1=signal_bob_modes[i], Mode2=idler_bob_modes[i],
                                  NumModes=nmodes)
        ent_sources_bob = ent_sources_bob @ this_source

    ## Defines SPDC sources for the ancillas.
    sources_ancillas = np.identity(2 * nmodes)

    if ancilla_type == 'HeraldedPhoton':
        herald_ancilla_pattern = heralding_ancilla_modes
        for i in range(dim - 2):
            this_source = get_tms_sym(ancillas_param, phi=0, Mode1=heralding_ancilla_modes[i], Mode2=init_anc_modes[i],
                                      NumModes=nmodes)
            sources_ancillas = sources_ancillas @ this_source

    elif ancilla_type == 'TMS':
        herald_ancilla_pattern = []
        if not (dim % 2) == 0:
            raise ValueError('Dimension needs to be even when using TMS for ancillas')
        for i in range(int((dim - 2) / 2)):
            # print('i:', i, init_anc_modes, int(dim/2), list(range(int(dim/2))))
            this_source = get_tms_sym(ancillas_param, phi=0, Mode1=init_anc_modes[2 * i],
                                      Mode2=init_anc_modes[2 * i + 1],
                                      NumModes=nmodes)
            sources_ancillas = sources_ancillas @ this_source

    ############################
    ### DEFINE LINEAR OPTICS ###
    ############################

    ## Defines the multiport interferometer
    multiport_U = np.identity(nmodes, dtype=np.cdouble)
    for i in range(dim):
        these_modes = [idler_alice_modes[i]] + [this_anc_modes[i] for this_anc_modes in ancilla_modes] + \
                      [idler_bob_modes[i]]
        this_dft = dft(these_modes, nmodes)
        multiport_U = this_dft @ multiport_U
    # print('\nmultiport_U:')
    # print(multiport_U)

    ## Define unitaries to prepare the input states for the ancillas
    prep_U_ancillas = np.identity(nmodes, dtype=np.cdouble)
    prep_vec = np.ones(dim) / np.sqrt(dim)
    for anc_modes in ancilla_modes:
        this_U = state_preparation_unitary(anc_modes, prep_vec, nmodes, chosen_inmode=init_mode_ix_ancillas)
        prep_U_ancillas = this_U @ prep_U_ancillas
    # print('\nprep_U_ancillas:')
    # print(prep_U_ancillas)

    ## U tilde unitary
    dim_utilde1, dim_utilde2 = np.shape(U_tilde)
    if dim_utilde1 != (dim + 1) or dim_utilde1 != dim_utilde2:
        raise ValueError('U_tilde needs to be a (dim+1)x(dim+1) square matrix')
    U_tilde_mat = np.identity(nmodes, dtype=np.cdouble)
    modes_Utilde = idler_bob_modes + extra_Utilde_modes
    U_tilde_mat[np.ix_(modes_Utilde, modes_Utilde)] = U_tilde
    # print('\nU_tilde:')
    # print(U_tilde_mat)

    ## Unitary U at the output
    dim_uout1, dim_uout2 = np.shape(U_out)
    if dim_uout1 != dim or dim_uout1 != dim_uout2:
        raise ValueError('U_out needs to be a (dim)x(dim) square matrix')
    U_out_mat = np.identity(nmodes, dtype=np.cdouble)
    U_out_mat[np.ix_(signal_bob_modes, signal_bob_modes)] = U_out
    # print('\nU_out:')
    # print(U_out_mat)

    ## Unitary for projective Alice measurement
    dim_ua1, dim_ua2 = np.shape(U_a)
    if dim_ua1 != dim or dim_ua1 != dim_ua2:
        raise ValueError('U_a needs to be a (dim)x(dim) square matrix')
    U_meas_alice = np.identity(nmodes, dtype=np.cdouble)
    modes_Ua = signal_alice_modes
    U_meas_alice[np.ix_(modes_Ua, modes_Ua)] = U_a
    # print('\nU_a:')
    # print(U_meas_alice)

    ## Unitary for projective Bob measurement
    dim_ub1, dim_ub2 = np.shape(U_b)
    if dim_ub1 != dim or dim_ub1 != dim_ub2:
        raise ValueError('U_b needs to be a (dim)x(dim) square matrix')
    U_meas_bob = np.identity(nmodes, dtype=np.cdouble)
    modes_Ua = signal_bob_modes
    U_meas_bob[np.ix_(modes_Ua, modes_Ua)] = U_b
    # print('\nU_b:')
    # print(U_meas_bob)

    ## Calculate the total linear-optical unitary.
    LO_unitary = U_meas_alice @ U_meas_bob @ U_out_mat @ multiport_U @ U_tilde_mat @ prep_U_ancillas

    ## Gets the linear-optical transformation in the symplectic form.
    LO_unitary_sym = get_unitary_sym(LO_unitary)

    ###############################################
    ### CALCULATE TOTAL GAUSSIAN TRANSFORMATION ###
    ###############################################

    ## Obtains the total Gaussian transformation matrix in the symplectic formalism
    sym_transf = LO_unitary_sym @ sources_ancillas @ ent_sources_bob @ ent_sources_alice

    #############################
    ### EVOLVE GAUSSIAN STATE ###
    #############################

    ## Obtains the covariance matrix of the output Gaussian state
    cov_mat = symplectic_evolution(cov_mat, sym_transf)

    ## Obtains the amplitudes of the output Gaussian state
    ampls = LO_unitary @ ampls
    # print('ampls1:', ampls)

    ###############################
    ### SINGLE PHOTON DETECTION ###
    ###############################

    ## Map state into the StrawberryField package, which has fast functions for
    ## photon counting (i.e. non-Gaussian measurements) from Gaussian states.
    gauss_state = map_into_StrawberryFields(ampls, cov_mat, nmodes)

    ## Define in which modes we want t observe a coincidence detection.
    ## Repeated elements would represent multiple photons in the same mode. I.e. [0, 0, 2]
    ## would indicate two photons in mode 0 and one in mode 2.
    herald_multiport_modes = [multiport_modes[ix] for ix in herald_pattern]

    detected_modes_list = [herald_ancilla_pattern + herald_multiport_modes + [out_alice, out_bob]
                           for (out_alice, out_bob) in product(signal_alice_modes, signal_bob_modes)]

    prob_list = []
    # print('\nSTART Detecting')
    for detected_modes in detected_modes_list:
        ## convert the detection configuration into a Fock state
        output_Fock = conf_to_Fock(detected_modes, nmodes)
        # print('Det:', detected_modes, 'Fock:', output_Fock)
        if number_resolving_det:
            ## Calculates the detection probability considering number-resolving detectors.
            prob = gauss_state.fock_prob(output_Fock)
        else:
            ## Calculates the detection probability considering threshold detectors.
            if parallelized:
                prob = threshold_detection_prob_parallel(gauss_state.cov(), gauss_state.means(), output_Fock)
            else:
                prob = threshold_detection_prob(gauss_state.cov(), gauss_state.means(), output_Fock)

        prob_list.append(prob)

    meas_probs = np.array(prob_list)
    tot_prob = np.sum(meas_probs)
    meas_probs_norm = meas_probs / tot_prob

    return meas_probs_norm, tot_prob
