"""
Test statistics of a two-mode cooherent squeezed state
"""

from GaussianTransformations import *
import numpy as np

from threshold_detection import threshold_detection_prob, threshold_detection_prob_parallel

######################################################################################################
def CoherSqueeze_Stats_simulator(squeezing_parameter, coher_ampl, outputs, number_resolving_det=False,
                                 old_PNR_func=False, cutoff=6, parallelized = True):
    """
    Function that simulates the statistics of a two-mode coherent squeezed state
    """

    #####################
    ### INITALISATION ###
    #####################

    nmodes = 2

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

    ############################
    ### DEFINE LINEAR OPTICS ###
    ############################

    ## Defines the linear-optical unitary (in this case a simple tunable beam-splitter),
    tunable_bs_matrix = np.identity(nmodes)

    LO_unitary = tunable_bs_matrix

    ## Gets the linear-optical transformation in the symplectic form.
    LO_unitary_sym = get_unitary_sym(LO_unitary)

    ###############################################
    ### CALCULATE TOTAL GAUSSIAN TRANSFORMATION ###
    ###############################################

    ## Obtains the total Gaussian transformation matrix in the symplectic formalism
    sym_transf = LO_unitary_sym @ TMS1

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
    ## Repeated elements would represent multiple photons in the same mode. I.e. [0, 0, 2]
    ## would indicate two photons in mode 0 and one in mode 2.
    output_configuration = outputs

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
            if parallelized:
                det_prob = threshold_detection_prob_parallel(np.array(gauss_state.cov()), np.array(gauss_state.means()), np.array(output_Fock))
            else:
                det_prob = threshold_detection_prob(gauss_state.cov(), gauss_state.means(), output_Fock)

    return det_prob


#########################################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    resolv_det = False
    cutoff = 8  # indicates maximum number of photons assumed in the experiment


    #########################################################
    ## Test 1 photon counts in []
    #########################################################

    print("\nTest 1: photon counts in [].")

    def prob00_Milb(r, ampl):
        return np.exp(-2*(np.abs(ampl)**2 - (ampl**2).conj() * np.tanh(r)))/(np.cosh(r)**2)

    s_par = 0.5
    ampl_list = np.linspace(0., 1, 50)
    outputs = np.array([])

    prob_new_list = [CoherSqueeze_Stats_simulator(s_par, ampl, outputs) for ampl in ampl_list]
    # prob_old_list = [CoherSqueeze_Stats_simulator(s_par, ampl, outputs, old_PNR_func=True, cutoff=cutoff) for ampl in ampl_list]

    prob_TheoMilb_list = [prob00_Milb(s_par, ampl) for ampl in ampl_list]


    plt.plot(ampl_list, prob_new_list, label="NewCode", lw=3)
    # plt.plot(ampl_list, prob_old_list, label="OldCode")
    plt.plot(ampl_list, prob_TheoMilb_list, 'k--', label="Analytical")
    plt.xlabel(r'Amplitude $\alpha$')
    plt.ylabel('2-Fold counts')
    plt.legend()
    plt.title('p(00), s_par='+str(s_par))
    plt.show()

    #########################################################
    ## Test 2 photon counts in [0, 1]
    #########################################################

    print("\nTest 2: photon counts in [0, 1].")


    def prob11_Milb(r, ampl):
        fact_0 = np.cosh(r)**2
        fact_1 = -2*np.exp(-(np.abs(ampl)**2)/(np.cosh(r)**2))
        fact_2 = np.exp(-2*(np.abs(ampl)**2 - (ampl**2).conj() * np.tanh(r)))
        return (fact_0 + fact_1 + fact_2)/(np.cosh(r)**2)


    s_par = 0.5
    ampl_list = np.linspace(0., 1, 50)
    outputs = [0, 1]

    prob_new_list = [CoherSqueeze_Stats_simulator(s_par, ampl, outputs) for ampl in ampl_list]
    # prob_old_list = [CoherSqueeze_Stats_simulator(s_par, ampl, outputs, old_PNR_func=True, cutoff=cutoff) for ampl in ampl_list]

    prob_TheoMilb_list = [prob11_Milb(s_par, ampl) for ampl in ampl_list]

    plt.plot(ampl_list, prob_new_list, label="NewCode", lw=3)
    # plt.plot(ampl_list, prob_old_list, label="OldCode")
    plt.plot(ampl_list, prob_TheoMilb_list, 'k--', label="Analytical")
    plt.xlabel(r'Amplitude $\alpha$')
    plt.ylabel('2-Fold counts')
    plt.legend()
    plt.title('p(11), s_par='+str(s_par))
    plt.show()



    #########################################################
    ## Test 3 photon counts in [0]
    #########################################################

    print("\nTest 3: photon counts in [0].")


    def prob01_Milb(r, ampl):
        fact_0 = np.exp(-(np.abs(ampl)**2)/(np.cosh(r)**2))
        return fact_0/(np.cosh(r)**2) - prob00_Milb(r, ampl)


    s_par = 0.5
    ampl_list = np.linspace(0., 1, 50)
    outputs = [0]

    prob_new_list = [CoherSqueeze_Stats_simulator(s_par, ampl, outputs) for ampl in ampl_list]
    # prob_old_list = [CoherSqueeze_Stats_simulator(s_par, ampl, outputs, old_PNR_func=True, cutoff=cutoff) for ampl in ampl_list]

    prob_TheoMilb_list = [prob01_Milb(s_par, ampl) for ampl in ampl_list]

    plt.plot(ampl_list, prob_new_list, label="NewCode", lw=3)
    # plt.plot(ampl_list, prob_old_list, label="OldCode")
    plt.plot(ampl_list, prob_TheoMilb_list, 'k--', label="Analytical")
    plt.xlabel(r'Amplitude $\alpha$')
    plt.ylabel('1-Fold counts')
    plt.legend()
    plt.title('p(01), s_par='+str(s_par))
    plt.show()