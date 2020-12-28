from HighD_Swapping_SchemeDefinition import HighD_Swapping_simulator
from SATWAP_functions import get_I_tilde, tsirelson_bound, classical_bound, get_SATWAP_meas_vects, theo_max_prob, \
    get_SATWAP_AB_matrices_list, SATWAP_rel_quantum_violation

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

#########################################################################################################
if __name__ == "__main__":

    #########################################################
    ## Test 1: SATWAP - dim=2
    #########################################################

    print('\nTest1: SATWAP - dim=2')

    dim = 2
    U_tilde = np.identity(dim + 1)
    U_out = np.identity(dim)

    # anc_type = 'WeakCoherent'
    anc_type = 'HeraldedPhoton'
    alpha = 0.1
    normalized_prob = True

    herald_pattern = np.array([0, 1]) + 0 * dim

    SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    simul_probs = np.array([
        HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern, U_out, U_tilde,
                                 ancillas_param=alpha,
                                 s_par_photons=0.1,
                                 ancilla_type=anc_type,
                                 number_resolving_det=False,  # TODO: change this False,
                                 parallelized=False)[0]
        for (U_a, U_b) in SATWAP_Umeas_list])
    simul_probs[[0, 1]] = simul_probs[[1, 0]]

    probs_theo = np.array([[theo_max_prob(x, y, a, b, dim)
                            for (a, b) in product(range(dim), repeat=2)]
                           for (x, y) in product(range(1, 3), repeat=2)])

    # print('\nsimul_probs:')
    # print(simul_probs)
    # print('\nTheo probs:')
    # print(probs_theo)

    meas_I_tilde = get_I_tilde(simul_probs, dim)
    print('\nGot:', meas_I_tilde)
    print('Tsirelson:', tsirelson_bound(dim))
    print('Classical:', classical_bound(dim))
    print('Relative Quantum Violation:', SATWAP_rel_quantum_violation(meas_I_tilde, dim))

    #########################################################
    ## Test 2: SATWAP - dim=3
    #########################################################

    print('\nTest2: SATWAP - dim=3')

    dim = 3
    U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    U_out = np.identity(dim)

    anc_type = 'WeakCoherent'
    # anc_type = 'HeraldedPhoton'
    alpha = 0.2
    normalized_prob = True

    herald_pattern = np.array([0, 1, 2]) + 0 * dim

    SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    simul_probs = np.array([
        HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern, U_out, U_tilde,
                                 ancillas_param=alpha,
                                 s_par_photons=0.01,
                                 ancilla_type=anc_type,
                                 number_resolving_det=False,  # TODO: change this False,
                                 parallelized=False)[0]
        for (U_a, U_b) in SATWAP_Umeas_list])

    probs_theo = np.array([[theo_max_prob(x, y, a, b, dim)
                            for (a, b) in product(range(dim), repeat=2)]
                           for (x, y) in product(range(1, 3), repeat=2)])

    # print('\nsimul_probs:')
    # print(simul_probs)
    # print('\nTheo probs:')
    # print(probs_theo)

    meas_I_tilde = get_I_tilde(simul_probs, dim)
    print('\nGot:', meas_I_tilde)
    print('Tsirelson:', tsirelson_bound(dim))
    print('Classical:', classical_bound(dim))
    print('Relative Quantum Violation:', SATWAP_rel_quantum_violation(meas_I_tilde, dim))

    #########################################################
    ## Test 3: SATWAP - dim=3 - Test performance against amplitude and squeezing parameters
    #########################################################

    print('\nTest2: SATWAP - dim=3 - Test performance against amplitude and squeezing parameters')

    dim = 3
    U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    U_out = np.identity(dim)

    anc_type = 'WeakCoherent'
    # anc_type = 'HeraldedPhoton'

    normalized_prob = True

    herald_pattern = np.array([0, 1, 2]) + 0 * dim

    SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)

    num_alphas_scan = 100
    num_s_par_scan = 100
    alpha_list = np.linspace(0.01, 1, num_alphas_scan)
    spar_list = np.linspace(0.01, 0.2, num_s_par_scan)

    SATWAP_scan_results = []
    for alpha in alpha_list:
        print('Doing alpha:', alpha)
        for s_par in spar_list:
            simul_probs_list = []
            succ_prob = 0
            for (U_a, U_b) in SATWAP_Umeas_list:
                temp_simul_probs, temp_succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
                                                                            U_out, U_tilde,
                                                                            ancillas_param=alpha,
                                                                            s_par_photons=s_par,
                                                                            ancilla_type=anc_type,
                                                                            number_resolving_det=False,
                                                                            # TODO: change this False,
                                                                            parallelized=False)
                simul_probs_list.append(temp_simul_probs)
                succ_prob += temp_succ_prob

            simul_probs = np.array(simul_probs_list)
            meas_I_tilde = get_I_tilde(simul_probs, dim)
            relative_quantum_violation = SATWAP_rel_quantum_violation(meas_I_tilde, dim)
            SATWAP_scan_results.append(np.array([alpha, s_par, relative_quantum_violation, succ_prob]))

    ###################### DATA ANALYSIS

    SATWAP_scan_results_array = np.array(SATWAP_scan_results)

    RQV_list = SATWAP_scan_results_array[:, 2]
    RQV_list_zeroed = np.array(list(map(lambda x: max(0, x), RQV_list)))
    RQV_list_zeroed_mat = RQV_list_zeroed.reshape((num_alphas_scan, num_s_par_scan)).T

    ### Obtain best success probability and associated squeezing and amplitude parameters

    RQV_num_steps = 40
    min_RQV = 0.0001#min(RQV_list_zeroed)
    max_RQV = max(RQV_list_zeroed)
    RQV_step = (max_RQV-min_RQV)/(RQV_num_steps-1)

    binned_RQV_results = [[] for i in range(RQV_num_steps)]

    for this_result in SATWAP_scan_results:
        this_RVQ = this_result[2]
        if this_RVQ > 0:
            bin_num = int((this_RVQ-min_RQV)/RQV_step)
            binned_RQV_results[bin_num].append(this_result)

    best_RQV_results = []
    for this_binned_res in binned_RQV_results:
        if len(this_binned_res) > 0:
            best_prob = max(np.array(this_binned_res)[:, 3])
            this_best_res = [t for t in this_binned_res if t[3] == best_prob][0]
            best_RQV_results.append(this_best_res)
    best_RQV_res_array = np.array(best_RQV_results)

    [best_alpha_list, best_spar_list, best_RVQ_list, best_succ_prob] = best_RQV_res_array.T

    #################### PLOTS ###########
    ### RQV vs alpha and s_par & alpha vs. s_par associated to Max succ_prob for given RQV
    fig, ax = plt.subplots()
    extent = [min(spar_list), max(spar_list), min(alpha_list), max(alpha_list)]
    aspect = (max(spar_list)-min(spar_list))/(max(alpha_list)-min(alpha_list))

    CS = ax.contour((alpha_list*aspect)+min(alpha_list), (spar_list-min(spar_list))/aspect, RQV_list_zeroed_mat, colors='k', extent=extent)
    ax.clabel(CS, fontsize=9, inline=1)

    ax.plot((best_alpha_list*aspect)+min(alpha_list), (best_spar_list-min(spar_list))/aspect, 'r:')

    this_plot = ax.imshow(RQV_list_zeroed_mat, interpolation='bilinear', cmap='seismic',
                          origin='lower',
                          aspect=aspect,
                          extent=extent,
                          vmax=1.0, vmin=0.0)

    ax.set_ylabel(r'Amplitude $\alpha$')
    ax.set_xlabel(r'Source Squeezing $s$')
    ax.set_title(r'SATWAP violation ($d=3$)')
    cbar = fig.colorbar(this_plot, ax=ax)
    cbar.set_label(r'Relative Quantum Violation $\frac{\tilde{I}_3 - C_3}{Q_3 - C_3}$', rotation=270, labelpad=25)
    cbar.minorticks_on()
    plt.show()

    ### Max succ_prob for given RQV
    fig, ax = plt.subplots()

    ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    left, bottom, width, height = [0.5, 0.45, 0.45, 0.45]
    ax2 = fig.add_axes([left, bottom, width, height])

    ax.plot(best_RVQ_list, best_succ_prob, 'k')
    ax.set_ylabel('Optimal success probability')
    ax.set_xlabel(r'Relative Quantum Violation $\frac{\tilde{I}_3 - C_3}{Q_3 - C_3}$')
    ax.set_title(r'SATWAP violation ($d=3$) - Optimal success prob.')
    ax.set_xticks(ticks)

    ax2.plot(best_RVQ_list, best_succ_prob, 'k')
    ax2.set_yscale('log')
    ax2.set_xticks(ticks)
    # ax.set_xlabel(r'Relative Quantum Violation $\frac{\tilde{I}_3 - C_3}{Q_3 - C_3}$')
    # ax.set_title(r'SATWAP violation ($d=3$) - Optimal success prob.')
    fig.tight_layout()
    plt.show()



    ### Max succ_prob for given RQV


    #########################################################
    ## Test 4: SATWAP - dim=4
    #########################################################

    # dim = 4
    # U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    # U_out = np.identity(dim)
    #
    # # anc_type = 'WeakCoherent'
    # # anc_type = 'TMS'
    # anc_type = 'HeraldedPhoton'
    # alpha = 0.01
    # normalized_prob = True
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # get_SATWAP_meas = get_SATWAP_meas_vects(dim)
    #
    # simul_probs = np.array([[
    #     HighD_Swapping_simulator(dim, state_a, state_b, herald_pattern, U_out, U_tilde,
    #                              ancillas_param=alpha,
    #                              s_par_photons=0.01,
    #                              ancilla_type=anc_type,
    #                              normalize_output=True,
    #                              number_resolving_det=False,
    #                              parallelized=False)[0]
    #     for (state_a, state_b) in get_SATWAP_meas[2 * (x-1) + (y-1)]]
    #     for (x, y) in product(range(1, 3), range(1, 3))])
    #
    # probs_theo = np.array([[theo_max_prob(x, y, a, b, dim)
    #                         for (a, b) in product(range(dim), repeat=2)]
    #                        for (x, y) in product(range(1, 3), repeat=2)])
    #
    # print('\nsimul_probs:')
    # print(simul_probs)
    # print('\nTheo probs:')
    # print(probs_theo)
    #
    # print('\nGot:', get_I_tilde(simul_probs, dim))
    # print('Tsirelson:', tsirelson_bound(dim))
    # print('Classical:', classical_bound(dim))
