from GaussianTransformations import *
from threshold_detection import threshold_detection_prob

import numpy as np


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
def HighD_Teleportation_CoherentAncillas_simulator(dim, psi, projection_state, herald_pattern, U_out, U_tilde,
                                                   alpha_ancillas=0.1, s_par_photons=0.01, normalize_output=True,
                                                   number_resolving_det=False):
    """
    Function that simulates the high-dimensional teleportation protocol using weak-coherent states with ancillas, and
    threshold detectors.

    :return: The probability of the projective measurement given the teleported state and the heralding pattern.
    """

    #####################
    ### INITALISATION ###
    #####################

    nmodes = dim * (dim + 1) + 2

    heralding_input_modes = [0]
    input_state_modes = list(range(1, dim + 1))
    ancilla_modes = [list(range((i + 1) * dim + 1, (i + 2) * dim + 1)) for i in range(dim - 2)]
    all_ancilla_modes = [mode for submodes in ancilla_modes for mode in submodes]
    bellpair_todft_modes = list(range(dim * (dim - 1) + 1, dim ** 2 + 1))
    extra_Utilde_modes = [dim ** 2 + 1]
    teleportation_modes = list(range(dim ** 2 + 2, nmodes))

    multiport_modes = input_state_modes + all_ancilla_modes + bellpair_todft_modes

    ## Initialises the initial covariance matrix of the Gaussian state to be the identity,
    ## meaning we start with vacuum in all modes.
    cov_mat = np.identity(2 * nmodes)

    ##############################
    ### DEFINE COHERENT STATES ###
    ##############################

    ## Defines the coherent state amplitudes in the input modes.
    ## Because no coherent state is present here, they are all zeros.
    ampls = np.zeros(nmodes)
    ampls[all_ancilla_modes] = np.ones(len(all_ancilla_modes)) * alpha_ancillas / np.sqrt(dim)

    ###########################
    ### DEFINE SPDC SOURCES ###
    ###########################

    ## Defines the SPDC source generating the heralded single photon with the input state.
    chosen_in_ix = 0
    chosen_in_mode = input_state_modes[chosen_in_ix]
    input_source = get_tms_sym(s_par_photons, phi=0, Mode1=heralding_input_modes[0], Mode2=chosen_in_mode,
                               NumModes=nmodes)

    ## Defines the second SPDC source that generate the high-dimensional Bell state.
    entanglement_sources = np.identity(2 * nmodes)

    for i in range(dim):
        this_source = get_tms_sym(s_par_photons, phi=0, Mode1=bellpair_todft_modes[i], Mode2=teleportation_modes[i],
                                  NumModes=nmodes)
        entanglement_sources = entanglement_sources @ this_source

    ############################
    ### DEFINE LINEAR OPTICS ###
    ############################

    ## Defines the multiport interferometer
    multiport_U = np.identity(nmodes, dtype=np.cdouble)
    for i in range(dim):
        these_modes = [input_state_modes[i]] + [this_anc_modes[i] for this_anc_modes in ancilla_modes] + \
                      [bellpair_todft_modes[i]]
        this_dft = dft(these_modes, nmodes)
        multiport_U = this_dft @ multiport_U

    ## Define unitary to prepare the input state
    if len(psi) != dim:
        raise ValueError('The length of state vector needs to be same as dimensionality')
    in_state_U = state_preparation_unitary(input_state_modes, psi, nmodes, chosen_inmode=chosen_in_ix)

    ## U tilde unitary
    dim_utilde1, dim_utilde2 = np.shape(U_tilde)
    if dim_utilde1 != (dim + 1) or dim_utilde1 != dim_utilde2:
        raise ValueError('U_tilde needs to be a (dim+1)x(dim+1) square matrix')
    U_tilde_mat = np.identity(nmodes, dtype=np.cdouble)
    modes_Utilde = bellpair_todft_modes + extra_Utilde_modes
    U_tilde_mat[np.ix_(modes_Utilde, modes_Utilde)] = U_tilde

    ## Unitary U at the output
    dim_uout1, dim_uout2 = np.shape(U_out)
    if dim_uout1 != dim or dim_uout1 != dim_uout2:
        raise ValueError('U_out needs to be a (dim)x(dim) square matrix')
    U_out_mat = np.identity(nmodes, dtype=np.cdouble)
    U_out_mat[np.ix_(teleportation_modes, teleportation_modes)] = U_out

    ## Unitary for projective measurement
    chosen_out_ix = 0
    chosen_out_mode = teleportation_modes[chosen_out_ix]
    U_meas = state_projection_unitary(teleportation_modes, projection_state, nmodes, chosen_outmode=chosen_out_ix)

    ## Calculate the total linear-optical unitary.
    LO_unitary = U_meas @ U_out_mat @ multiport_U @ U_tilde_mat @ in_state_U

    ## Gets the linear-optical transformation in the symplectic form.
    LO_unitary_sym = get_unitary_sym(LO_unitary)

    ###############################################
    ### CALCULATE TOTAL GAUSSIAN TRANSFORMATION ###
    ###############################################

    ## Obtains the total Gaussian transformation matrix in the symplectic formalism
    sym_transf = LO_unitary_sym @ entanglement_sources @ input_source

    #############################
    ### EVOLVE GAUSSIAN STATE ###
    #############################

    ## Obtains the covariance matrix of the output Gaussian state
    cov_mat = symplectic_evolution(cov_mat, sym_transf)

    ## Obtains the amplitudes of the output Gaussian state
    ampls = LO_unitary @ ampls

    ###############################
    ### SINGLE PHOTON DETECTION ###
    ###############################

    ## Map state into the StrawberryField package, which has fast functions for
    ## photon counting (i.e. non-Gaussian measurements) from Gaussian states.
    gauss_state = map_into_StrawberryFields(ampls, cov_mat, nmodes)

    ## Define in which modes we want t observe a coincidence detection.
    ## In this case we are looking for a 4-fold coincidence detection with a photon in all four modes.
    ## Repeated elements would represent multiple photons in the same mode. I.e. [0, 0, 2]
    ## would indicate two photons in mode 0 and one in mode 2.
    herald_modes = [multiport_modes[ix] for ix in herald_pattern]

    if normalize_output:
        detected_modes_list = [heralding_input_modes + herald_modes + [out_mode] for out_mode in teleportation_modes]
    else:
        detected_modes_list = [heralding_input_modes + herald_modes + [chosen_out_mode]]

    prob_list = []
    for detected_modes in detected_modes_list:
        ## convert the detection configuration into a Fock state
        output_Fock = conf_to_Fock(detected_modes, nmodes)
        if number_resolving_det:
            ## Calculates the detection probability considering number-resolving detectors.
            prob = gauss_state.fock_prob(output_Fock)
        else:
            ## Calculates the detection probability considering threshold detectors.
            prob = threshold_detection_prob(gauss_state, output_Fock)
        prob_list.append(prob)

    if normalize_output:
        det_prob = prob_list[chosen_out_ix] / sum(prob_list)
        succ_prob = prob_list[chosen_out_ix]
        return det_prob, succ_prob
    else:
        det_prob = prob_list[0]
        return det_prob,
