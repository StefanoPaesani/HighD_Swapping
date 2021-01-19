from HighD_Swapping_SchemeDefinition import HighD_Swapping_simulator
from SATWAP_functions import get_I_tilde, tsirelson_bound, classical_bound, get_SATWAP_meas_vects, theo_max_prob, \
    get_SATWAP_AB_matrices_list, SATWAP_rel_quantum_violation

import numpy as np
from itertools import product
import matplotlib.pyplot as plt

import os
import pickle

DataSavingFolder = os.path.join(os.getcwd(), 'Results')

#########################################################################################################
if __name__ == "__main__":

    #########################################################
    ## Test 1: SATWAP - dim=2
    #########################################################

    # print('\nTest1: SATWAP - dim=2')
    #
    # dim = 2
    # U_tilde = np.identity(dim + 1)
    # U_out = np.identity(dim)
    #
    # # anc_type = 'WeakCoherent'
    # anc_type = 'HeraldedPhoton'
    # alpha = 0.1
    # normalized_prob = True
    #
    # herald_pattern = np.array([0, 1]) + 0 * dim
    #
    # SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    # simul_probs = np.array([
    #     HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern, U_out, U_tilde,
    #                              ancillas_param=alpha,
    #                              s_par_photons=0.1,
    #                              ancilla_type=anc_type,
    #                              number_resolving_det=False,  # TODO: change this False,
    #                              parallelized=False)[0]
    #     for (U_a, U_b) in SATWAP_Umeas_list])
    # simul_probs[[0, 1]] = simul_probs[[1, 0]]
    #
    # probs_theo = np.array([[theo_max_prob(x, y, a, b, dim)
    #                         for (a, b) in product(range(dim), repeat=2)]
    #                        for (x, y) in product(range(1, 3), repeat=2)])
    #
    # # print('\nsimul_probs:')
    # # print(simul_probs)
    # # print('\nTheo probs:')
    # # print(probs_theo)
    #
    # meas_I_tilde = get_I_tilde(simul_probs, dim)
    # print('\nGot:', meas_I_tilde)
    # print('Tsirelson:', tsirelson_bound(dim))
    # print('Classical:', classical_bound(dim))
    # print('Relative Quantum Violation:', SATWAP_rel_quantum_violation(meas_I_tilde, dim))

    #########################################################
    ## Test 2: SATWAP - dim=3
    #########################################################

    print('\nTest2: SATWAP - dim=3')

    dim = 3
    U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    U_out = np.identity(dim)

    # anc_type = 'WeakCoherent'
    anc_type = 'HeraldedPhoton'
    alpha = 0.1
    normalized_prob = True

    herald_pattern = np.array([0, 1, 2]) + 0 * dim

    SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)

    simul_probs = []
    succ_prob = 0
    for (U_a, U_b) in SATWAP_Umeas_list:
        this_prob, this_succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern, U_out, U_tilde,
                                 ancillas_param=alpha,
                                 s_par_photons=0.01,
                                 ancilla_type=anc_type,
                                 number_resolving_det=False,  # TODO: change this False,
                                 parallelized=False)

        simul_probs.append(this_prob)
        succ_prob += this_succ_prob

    print('Success probability:', succ_prob)

    simul_probs = np.array(simul_probs)

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

    # print('\nTest2: SATWAP - dim=3 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 3
    # U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    # U_out = np.identity(dim)
    #
    # anc_type = 'WeakCoherent'
    # # anc_type = 'HeraldedPhoton'
    #
    # normalized_prob = True
    #
    # herald_pattern = np.array([0, 1, 2]) + 0 * dim
    #
    # SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    #
    # num_alphas_scan = 200
    # num_s_par_scan = 200
    # alpha_list = np.linspace(0.01, 1, num_alphas_scan)
    # spar_list = np.linspace(0.01, 0.2, num_s_par_scan)
    #
    # SATWAP_scan_results = []
    # for alpha in alpha_list:
    #     print('Doing alpha:', alpha)
    #     for s_par in spar_list:
    #         simul_probs_list = []
    #         succ_prob = 0
    #         for (U_a, U_b) in SATWAP_Umeas_list:
    #             temp_simul_probs, temp_succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                         U_out, U_tilde,
    #                                                                         ancillas_param=alpha,
    #                                                                         s_par_photons=s_par,
    #                                                                         ancilla_type=anc_type,
    #                                                                         number_resolving_det=False,
    #                                                                         # TODO: change this False,
    #                                                                         parallelized=False)
    #             simul_probs_list.append(temp_simul_probs)
    #             succ_prob += temp_succ_prob
    #
    #         simul_probs = np.array(simul_probs_list)
    #         meas_I_tilde = get_I_tilde(simul_probs, dim)
    #         relative_quantum_violation = SATWAP_rel_quantum_violation(meas_I_tilde, dim)
    #         SATWAP_scan_results.append(np.array([alpha, s_par, relative_quantum_violation, succ_prob]))
    #
    #
    #
    # SATWAP_scan_results_array = np.array(SATWAP_scan_results)
    #
    # ###################### SAVE DATA
    # filename = 'SATWAPscan_dim' + str(dim) + '_numpoints' + str(num_s_par_scan * num_alphas_scan)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, SATWAP_scan_results_array, delimiter=",")
    # print('\nResults saved in: '+full_saving_name)


    #########################################################
    ## Test 4: SATWAP - dim=4
    #########################################################

    # dim = 4
    #
    # U_tilde = np.ones((dim + 1, dim + 1))
    # for i in range(dim):
    #     U_tilde[i, i] = -2.
    #     U_tilde[dim, i] = np.sqrt(2)
    #     U_tilde[i, dim] = np.sqrt(2)
    # U_tilde[dim, dim] = -1.
    # U_tilde = U_tilde / 3.
    #
    # U_out = np.identity(dim)
    #
    # # anc_type = 'WeakCoherent'
    # anc_type = 'TMS'
    # # anc_type = 'HeraldedPhoton'
    #
    # alpha = 0.1
    # normalized_prob = True
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    # simul_probs = np.array([
    #     HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern, U_out, U_tilde,
    #                              ancillas_param=alpha,
    #                              s_par_photons=0.1,
    #                              ancilla_type=anc_type,
    #                              number_resolving_det=False,  # TODO: change this False,
    #                              parallelized=False)[0]
    #     for (U_a, U_b) in SATWAP_Umeas_list])
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
    # meas_I_tilde = get_I_tilde(simul_probs, dim)
    # print('\nMeasured I_tilde:', meas_I_tilde)
    # print('Tsirelson:', tsirelson_bound(dim))
    # print('Classical:', classical_bound(dim))
    # print('Relative Quantum Violation:', SATWAP_rel_quantum_violation(meas_I_tilde, dim))

    #########################################################
    ## Test 5: SATWAP - dim=4 - Test performance against amplitude and squeezing parameters
    #########################################################

    # print('\nTest5: SATWAP - dim=4 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 4
    #
    # U_tilde = np.ones((dim + 1, dim + 1))
    # for i in range(dim):
    #     U_tilde[i, i] = -2.
    #     U_tilde[dim, i] = np.sqrt(2)
    #     U_tilde[i, dim] = np.sqrt(2)
    # U_tilde[dim, dim] = -1.
    # U_tilde = U_tilde / 3.
    #
    # U_out = np.identity(dim)
    #
    # # anc_type = 'WeakCoherent'
    # anc_type = 'TMS'
    # # anc_type = 'HeraldedPhoton'
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    #
    # num_alphas_scan = 200
    # num_s_par_scan = 200
    # # alpha_list = np.linspace(0.1, 0.5, num_alphas_scan)
    # # spar_list = np.linspace(0.015, 0.25, num_s_par_scan)
    # alpha_list = np.linspace(0.06, 0.5, num_alphas_scan)
    # spar_list = np.linspace(0.02, 0.25, num_s_par_scan)
    #
    # SATWAP_scan_results = []
    # for alpha in alpha_list:
    #     print('Doing alpha:', alpha)
    #     for s_par in spar_list:
    #         simul_probs_list = []
    #         succ_prob = 0
    #         for (U_a, U_b) in SATWAP_Umeas_list:
    #             temp_simul_probs, temp_succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                         U_out, U_tilde,
    #                                                                         ancillas_param=alpha,
    #                                                                         s_par_photons=s_par,
    #                                                                         ancilla_type=anc_type,
    #                                                                         number_resolving_det=False,
    #                                                                         # TODO: change this False,
    #                                                                         parallelized=False)
    #             simul_probs_list.append(temp_simul_probs)
    #             succ_prob += temp_succ_prob
    #
    #         simul_probs = np.array(simul_probs_list)
    #         meas_I_tilde = get_I_tilde(simul_probs, dim)
    #         relative_quantum_violation = SATWAP_rel_quantum_violation(meas_I_tilde, dim)
    #         SATWAP_scan_results.append(np.array([alpha, s_par, relative_quantum_violation, succ_prob]))
    #
    # SATWAP_scan_results_array = np.array(SATWAP_scan_results)
    #
    # ###################### SAVE DATA
    # filename = 'SATWAPscan_dim' + str(dim) + '_numpoints' + str(num_s_par_scan * num_alphas_scan)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, SATWAP_scan_results_array, delimiter=",")
    # print('\nResults saved in: '+full_saving_name)


    #########################################################
    ## Test 6: SATWAP - dim=2 - Test performance against amplitude and squeezing parameters
    #########################################################

    # print('\nTest5: SATWAP - dim=2 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 2
    #
    # U_tilde = np.identity(dim+1)
    # U_out = np.identity(dim)
    #
    # anc_type = 'WeakCoherent'
    # # anc_type = 'TMS'
    # # anc_type = 'HeraldedPhoton'
    #
    # herald_pattern = np.array([0, 1]) + 0 * dim
    #
    # SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    #
    # num_s_par_scan = 400
    # alpha = 0
    # spar_list = np.linspace(0.001, 1, num_s_par_scan)
    #
    # SATWAP_scan_results = []
    #
    # for s_par in spar_list:
    #     print('Doing s_par:', s_par)
    #     simul_probs_list = []
    #     succ_prob = 0
    #     for (U_a, U_b) in SATWAP_Umeas_list:
    #         temp_simul_probs, temp_succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                     U_out, U_tilde,
    #                                                                     ancillas_param=alpha,
    #                                                                     s_par_photons=s_par,
    #                                                                     ancilla_type=anc_type,
    #                                                                     number_resolving_det=False,
    #                                                                     # TODO: change this False,
    #                                                                     parallelized=False)
    #         simul_probs_list.append(temp_simul_probs)
    #         succ_prob += temp_succ_prob
    #
    #     simul_probs = np.array(simul_probs_list)
    #     simul_probs[[0, 1]] = simul_probs[[1, 0]]
    #
    #     meas_I_tilde = get_I_tilde(simul_probs, dim)
    #     relative_quantum_violation = SATWAP_rel_quantum_violation(meas_I_tilde, dim)
    #     SATWAP_scan_results.append(np.array([s_par, relative_quantum_violation, succ_prob]))
    #
    # SATWAP_scan_results_array = np.array(SATWAP_scan_results)
    #
    # ###################### SAVE DATA
    # filename = 'SATWAPscan_dim' + str(dim) + '_numpoints' + str(num_s_par_scan)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, SATWAP_scan_results_array, delimiter=",")
    # print('\nResults saved in: '+full_saving_name)



    #########################################################
    ## Test 7: SATWAP - dim=3 - Test QBER performance against amplitude and squeezing parameters
    #########################################################

    # print('\nTest7: SATWAP - dim=3 - Test performance against amplitude and squeezing parameters')
    #
    #
    # dim = 3
    # U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    # U_out = np.identity(dim)
    #
    # anc_type = 'WeakCoherent'
    # # anc_type = 'HeraldedPhoton'
    #
    # herald_pattern = np.array([0, 1, 2]) + 0 * dim
    #
    # SATWAP_Umeas_list = get_SATWAP_AB_matrices_list(dim)
    #
    # num_alphas_scan = 200
    # num_s_par_scan = 200
    # alpha_list = np.linspace(0.01, 2.25, num_alphas_scan)
    # spar_list = np.linspace(0.01, 4, num_s_par_scan)
    #
    # U_a = np.identity(dim)
    # U_b = np.identity(dim)
    #
    # QBER_scan_results = []
    # for alpha in alpha_list:
    #     print('Doing alpha:', alpha)
    #     for s_par in spar_list:
    #         simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                     U_out, U_tilde,
    #                                                                     ancillas_param=alpha,
    #                                                                     s_par_photons=s_par,
    #                                                                     ancilla_type=anc_type,
    #                                                                     number_resolving_det=False,
    #                                                                     # TODO: change this False,
    #                                                                     parallelized=False)
    #
    #         simul_probs = np.array(simul_probs_list)
    #         simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    #         this_fid = sum(np.sqrt(np.diag(simul_probs_mat)/dim))**2
    #         QBER_scan_results.append(np.array([alpha, s_par, this_fid, succ_prob]))
    #
    # QBER_scan_results_array = np.array(QBER_scan_results)
    #
    # ###################### SAVE DATA
    # filename = 'QBERscan_dim' + str(dim) + '_numpoints' + str(num_s_par_scan * num_alphas_scan)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, QBER_scan_results_array, delimiter=",")
    # print('\nResults saved in: '+full_saving_name)



    #########################################################
    ## Test 8: SATWAP - dim=4 - Test QBER performance against amplitude and squeezing parameters
    #########################################################

    # print('\nTest8: SATWAP - dim=4 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 4
    #
    # U_tilde = np.ones((dim + 1, dim + 1))
    # for i in range(dim):
    #     U_tilde[i, i] = -2.
    #     U_tilde[dim, i] = np.sqrt(2)
    #     U_tilde[i, dim] = np.sqrt(2)
    # U_tilde[dim, dim] = -1.
    # U_tilde = U_tilde / 3.
    #
    # U_out = np.identity(dim)
    #
    # # anc_type = 'WeakCoherent'
    # anc_type = 'TMS'
    # # anc_type = 'HeraldedPhoton'
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # num_alphas_scan = 200
    # num_s_par_scan = 200
    #
    # alpha_list = np.linspace(0.06, 2, num_alphas_scan)
    # spar_list = np.linspace(0.02, 2, num_s_par_scan)
    #
    # U_a = np.identity(dim)
    # U_b = np.identity(dim)
    #
    # QBER_scan_results = []
    # for alpha in alpha_list:
    #     print('Doing alpha:', alpha)
    #     for s_par in spar_list:
    #         simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                     U_out, U_tilde,
    #                                                                     ancillas_param=alpha,
    #                                                                     s_par_photons=s_par,
    #                                                                     ancilla_type=anc_type,
    #                                                                     number_resolving_det=False,
    #                                                                     # TODO: change this False,
    #                                                                     parallelized=False)
    #
    #         simul_probs = np.array(simul_probs_list)
    #         simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    #         this_fid = sum(np.sqrt(np.diag(simul_probs_mat)/dim))**2
    #         QBER_scan_results.append(np.array([alpha, s_par, this_fid, succ_prob]))
    #
    # SATWAP_scan_results_array = np.array(QBER_scan_results)
    #
    # ###################### SAVE DATA
    # filename = 'QBERscan_dim' + str(dim) + '_numpoints' + str(num_s_par_scan * num_alphas_scan)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, SATWAP_scan_results_array, delimiter=",")
    # print('\nResults saved in: '+full_saving_name)
