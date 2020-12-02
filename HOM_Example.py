"""
This is an example of how to use the GaussianTransformations functions to simulate an HOM experiment using two
SPDC sources of two-mode squeezing (TMS) (i.e. heralded single photons in the low-squeezing regime).

The experiment simulates is the following:


              O        O
               \      /
               _\____/_
              |___BS__|
     O         /     \
      \       /       \        O
       \     /         \      /
   mode0\   /mode1 mode2\   /mode3
     ____\/__        ____\/__
    | TMS_1 |       | TMS_2 |
    --------        --------


Stefano Paesani, Nov. 2020
"""

from GaussianTransformations import *
import numpy as np


######################################################################################################
def Heralded_HOM_exp_simulator(squeezing_parameter, bs_reflectivity, number_resolving_det=True, coher_ampl=0., old_PNR_func=False,
                               cutoff=6):
    """
    Function that simulates the HOM experiment as a function of the squeezing, the BS reflectivity,
    and of the type of single photon detectors used.

    :param float squeezing_parameter: The squeezing parameter of the SPDC sources.
                                      (probability of emitting a photon pair approximately tanh(squeezing_parameter)^2,
                                      in the low-squeezing regime)
    :param float bs_reflectivity: reflectivity of the beam-splitter
    :param bool number_resolving_det: If True, photon-number resolving detectors are used,
                                      if False threshold detectors are used.
    :param int cutoff: Cutoff for maximum number of photons to be present in all detected modes,
                       used when simulating threshold detection.
    :return: The 4-fold coincidence detection probability, i.e. probability that all 4 output detectors click.
    """

    #####################
    ### INITALISATION ###
    #####################

    nmodes = 4

    ## Initialises the initial covariance matrix of the Gaussian state to be the identity,
    ## meaning we start with vacuum in all modes.
    cov_mat = np.identity(2 * nmodes)

    ##############################
    ### DEFINE COHERENT STATES ###
    ##############################

    ## Defines the coherent state amplitudes in the input modes.
    ## Because no coherent state is present here, they are all zeros.
    ampls = np.ones(nmodes) * coher_ampl

    ###########################
    ### DEFINE SPDC SOURCES ###
    ###########################

    ## Defines the first SPDC source as a TMS between modes 0 and 1, as a symplectic transformation
    TMS1 = get_tms_sym(squeezing_parameter, phi=0, Mode1=0, Mode2=1, NumModes=nmodes)

    ## Defines the second SPDC source as a TMS between modes 2 and 3, as a symplectic transformation
    TMS2 = get_tms_sym(squeezing_parameter, phi=0, Mode1=2, Mode2=3, NumModes=nmodes)

    ############################
    ### DEFINE LINEAR OPTICS ###
    ############################

    ## Defines the linear-optical unitary (in this case a simple tunable beam-splitter),
    tunable_bs_matrix = np.identity(nmodes)
    tunable_bs_matrix[1, 1] = np.sqrt(1 - bs_reflectivity)
    tunable_bs_matrix[1, 2] = np.sqrt(bs_reflectivity)
    tunable_bs_matrix[2, 1] = - np.sqrt(bs_reflectivity)
    tunable_bs_matrix[2, 2] = np.sqrt(1 - bs_reflectivity)
    LO_unitary = tunable_bs_matrix

    ## Gets the linear-optical transformation in the symplectic form.
    LO_unitary_sym = get_unitary_sym(LO_unitary)

    ###############################################
    ### CALCULATE TOTAL GAUSSIAN TRANSFORMATION ###
    ###############################################

    ## Obtains the total Gaussian transformation matrix in the symplectic formalism
    sym_transf = LO_unitary_sym @ TMS2 @ TMS1

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
    output_configuration = [0, 1, 2, 3]

    ## convert the detection configuration into a Fock state
    output_Fock = conf_to_Fock(output_configuration, nmodes)

    if number_resolving_det:
        ## Calculates the detection probability considering number-resolving detectors.
        det_prob = gauss_state.fock_prob(output_Fock)
    else:
        if old_PNR_func:
            ## Calculates the detection probability considering threshold detectors.
            det_prob = threshold_detection_prob_old(gauss_state, output_Fock, cutoff=cutoff)
        else:
            ## Calculates the detection probability considering threshold detectors.
            det_prob = threshold_detection_prob(gauss_state, output_Fock)

    return det_prob


#########################################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    resolv_det = False
    cutoff = 8  # indicates maximum number of photons assumed in the experiment

    ################################################################################
    ## Test 1: calculate visibility with fixed bs reflectivity and squeezing value
    ################################################################################

    print("\nTest 1: calculate visibility with fixed bs reflectivity and squeezing value.")
    s_par = 0.2
    ampl = 0.2
    bs_refl = 0.5

    coinc_prob = Heralded_HOM_exp_simulator(s_par, bs_refl, resolv_det, coher_ampl=ampl, cutoff=cutoff)
    print("Squeezing:", s_par, "; Reflectivity:", bs_refl, "; 4-fold probability:", coinc_prob)

    coinc_prob_old = Heralded_HOM_exp_simulator(s_par, bs_refl, resolv_det, coher_ampl=ampl, old_PNR_func=True, cutoff=cutoff)
    print("Squeezing:", s_par, "; Reflectivity:", bs_refl, "; 4-fold probability:", coinc_prob_old)

    #########################################################
    ## Test 2 (HOM fringe): photon counts vs. bs reflectivity
    #########################################################

    print("\nTest 2: (HOM fringe): photon counts vs. BS reflectivity.")
    s_par = 0.2
    ampl = 0.2
    refl_list = np.linspace(0., 1, 101)
    det_counts_list = [Heralded_HOM_exp_simulator(s_par, bs_refl, resolv_det, coher_ampl=ampl, cutoff=cutoff) for bs_refl in refl_list]

    det_counts_list_old = [Heralded_HOM_exp_simulator(s_par, bs_refl, resolv_det, coher_ampl=ampl, old_PNR_func=True, cutoff=cutoff) for bs_refl in refl_list]

    plt.plot(refl_list, det_counts_list, label="NewCode")
    plt.plot(refl_list, det_counts_list_old, label="OldCode")
    plt.xlabel('Beam-splitter reflectivity')
    plt.ylabel('4-Fold counts')
    plt.legend()
    plt.show()

    #########################################################
    ## Test 3: HOM visibility vs squeezing parameter, 50/50 BS
    #########################################################

    def get_HOM_visibility(squeezing_parameter, number_resolving_det=resolv_det, coher_ampl=0., old_PNR_func=False, phot_cutoff=cutoff):
        max_prob = Heralded_HOM_exp_simulator(squeezing_parameter, 0, number_resolving_det, coher_ampl=coher_ampl, old_PNR_func=old_PNR_func, cutoff=phot_cutoff)
        min_prob = Heralded_HOM_exp_simulator(squeezing_parameter, 0.5, number_resolving_det, coher_ampl=coher_ampl, old_PNR_func=old_PNR_func, cutoff=phot_cutoff)
        return (max_prob - 2 * min_prob)/max_prob # see Adcock et al. Nature Comm. (2019)

    print("\nTest 3: HOM visibility vs squeezing parameter, 50/50 BS.")
    ampl = 0.2
    squeeze_list = np.linspace(0.01, 0.5, 101)

    vis_list = [get_HOM_visibility(s_par, resolv_det, coher_ampl=ampl) for s_par in squeeze_list]

    vis_list_old = [get_HOM_visibility(s_par, resolv_det, coher_ampl=ampl, old_PNR_func=True) for s_par in squeeze_list]

    plt.plot(np.tanh(squeeze_list), vis_list, label="NewCode")
    plt.plot(np.tanh(squeeze_list), vis_list_old, label="OldCode")
    plt.xlabel(r'Squeezing parameter  $|tanh(s)|$')
    plt.ylabel('HOM visibility')
    plt.legend()
    plt.show()



    #########################################################
    ## Test 3: HOM visibility vs amplitude, 50/50 BS
    #########################################################


    print("\nTest 4: HOM visibility vs amplitude, 50/50 BS.")
    s_par = 0.1
    ampl_list = np.linspace(0.001, 0.5, 101)

    vis_list = [get_HOM_visibility(s_par, resolv_det, coher_ampl=ampl) for ampl in ampl_list]

    vis_list_old = [get_HOM_visibility(s_par, resolv_det, coher_ampl=ampl, old_PNR_func=True) for ampl in ampl_list]

    plt.plot(ampl_list, vis_list, label="NewCode")
    plt.plot(ampl_list, vis_list_old, label="OldCode")
    plt.xlabel(r'Amplitude parameter  $\alpha$')
    plt.ylabel('HOM visibility')
    plt.legend()
    plt.show()






