from HighD_Swapping_SchemeDefinition import HighD_Swapping_simulator
from MARCUS_functions import get_U_MUB, get_KeyRate_Marcus

import numpy as np
from itertools import product

import os
import pickle

DataSavingFolder = os.path.join(os.getcwd(), 'Results')




#########################################################################################################
if __name__ == "__main__":

    #########################################################
    ## Test 1: MARCUS - dim=3, k=3
    #########################################################

    # print('\nTest1: MARCUS - dim=3, k=3')
    #
    # dim = 3
    # k_sub = 3
    #
    # U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    # U_out = np.identity(dim)
    #
    # anc_type = 'WeakCoherent'
    # # anc_type = 'HeraldedPhoton'
    #
    # alpha = 0.2
    # s_par = 0.01
    #
    # normalized_prob = True
    #
    # transm = 1.0
    # dark_count_prob = 10**(-4)
    # xtalk_par = 0.
    #
    # herald_pattern = np.array([0, 1, 2]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                        U_out, U_tilde,
    #                                                        ancillas_param=alpha,
    #                                                        s_par_photons=s_par,
    #                                                        transm=transm,
    #                                                        dark_counts_prob=dark_count_prob,
    #                                                        xtalk_par=xtalk_par,
    #                                                        ancilla_type=anc_type,
    #                                                        number_resolving_det=False,
    #                                                        parallelized=False)
    #
    # simul_probs = abs(np.array(simul_probs_list))
    # simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    # this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    # print(this_K)
    # print(succ_prob)
    #
    # print(simul_probs_mat)

    #########################################################
    ## Test 3: SATWAP - dim=3 k=3- Test performance against Xtalk
    #########################################################

    # print('\nTest3: SATWAP - dim=3 k=3 - Test performance against Xtalk')
    #
    # dim = 3
    # k_sub = 3
    #
    # U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    # U_out = np.identity(dim)
    #
    # anc_type = 'WeakCoherent'
    # # anc_type = 'HeraldedPhoton'
    #
    # herald_pattern = np.array([0, 1, 2]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # num_alphas_scan = 20
    # num_s_par_scan = num_alphas_scan
    # alpha_list = np.linspace(0.012, 1, num_alphas_scan)
    # spar_list = np.linspace(0.01, 0.12, num_s_par_scan)
    #
    # dark_count_prob = 0.
    # transm = 1.
    #
    # xtalk_par_list = np.linspace(0, 0.5, 12)
    #
    # MARCUS_scan_results = []
    # for this_xtalk_par in xtalk_par_list:
    #     print('   Doing xtalk_par:', this_xtalk_par)
    #     for alpha in alpha_list:
    #         print('Doing alpha:', alpha)
    #         for s_par in spar_list:
    #
    #             simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                    U_out, U_tilde,
    #                                                                    ancillas_param=alpha,
    #                                                                    s_par_photons=s_par,
    #                                                                    transm=transm,
    #                                                                    dark_counts_prob=dark_count_prob,
    #                                                                    xtalk_par=this_xtalk_par,
    #                                                                    ancilla_type=anc_type,
    #                                                                    number_resolving_det=False,
    #                                                                    parallelized=False)
    #
    #             simul_probs = abs(np.array(simul_probs_list))
    #             simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    #             this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    #             MARCUS_scan_results.append(np.array([alpha, s_par, this_xtalk_par, this_K, succ_prob]))
    #
    # MARCUS_scan_results_array = np.array(MARCUS_scan_results)
    #
    # # ###################### SAVE DATA
    # filename = 'MARCUSscan_withNoise_scanXtalk_dim' + str(dim) + '_k' + str(k_sub) + \
    #            '_numpoints' + str(num_alphas_scan) + '_t' + str(transm) + '_DCprob' + str(dark_count_prob)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    # print('\nResults saved in: ' + full_saving_name)


    #########################################################
    ## Test 3: SATWAP - dim=3 k=3- Test performance against DarkCounts
    #########################################################

    # print('\nTest3: SATWAP - dim=3 k=3 - Test performance against dark counts')
    #
    # dim = 3
    # k_sub = 3
    #
    # U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    # U_out = np.identity(dim)
    #
    # anc_type = 'WeakCoherent'
    # # anc_type = 'HeraldedPhoton'
    #
    # herald_pattern = np.array([0, 1, 2]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # num_alphas_scan = 20
    # num_s_par_scan = num_alphas_scan
    # alpha_list = np.linspace(0.012, 1, num_alphas_scan)
    # spar_list = np.linspace(0.01, 0.12, num_s_par_scan)
    #
    # transm = 1.
    # xtalk_par = 0.
    #
    # dark_count_prob_list = np.logspace(-7, -3, num=12)
    #
    # MARCUS_scan_results = []
    # for this_DC_par in dark_count_prob_list:
    #     print('   Doing dark_count_prob:', this_DC_par)
    #     for alpha in alpha_list:
    #         print('Doing alpha:', alpha)
    #         for s_par in spar_list:
    #
    #             simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                    U_out, U_tilde,
    #                                                                    ancillas_param=alpha,
    #                                                                    s_par_photons=s_par,
    #                                                                    transm=transm,
    #                                                                    dark_counts_prob=this_DC_par,
    #                                                                    xtalk_par=xtalk_par,
    #                                                                    ancilla_type=anc_type,
    #                                                                    number_resolving_det=False,
    #                                                                    parallelized=False)
    #
    #             simul_probs = abs(np.array(simul_probs_list))
    #             simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    #             this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    #             MARCUS_scan_results.append(np.array([alpha, s_par, this_DC_par, this_K, succ_prob]))
    #
    # MARCUS_scan_results_array = np.array(MARCUS_scan_results)
    #
    # # ###################### SAVE DATA
    # filename = 'MARCUSscan_withNoise_scanDarkCounts_dim' + str(dim) + '_k' + str(k_sub) + \
    #            '_numpoints' + str(num_alphas_scan) + '_t' + str(transm)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    # print('\nResults saved in: ' + full_saving_name)

    #########################################################
    ## Test 4: SATWAP - dim=4, k=4
    #########################################################

    # print('\nTest4: SATWAP - dim=4 k=4 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 4
    # k_sub = 4
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
    # alpha = 0.05
    # s_par = 0.03
    #
    # transm = 1.
    # dark_count_prob = 0
    # xtalk_par = 0.2
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                        U_out, U_tilde,
    #                                                        ancillas_param=alpha,
    #                                                        s_par_photons=s_par,
    #                                                        transm=transm,
    #                                                        dark_counts_prob=dark_count_prob,
    #                                                        xtalk_par=xtalk_par,
    #                                                        ancilla_type=anc_type,
    #                                                        number_resolving_det=False,
    #                                                        parallelized=False)
    #
    # simul_probs = abs(np.array(simul_probs_list))
    # simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    # this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    # print('This K:', this_K)
    # print('Succ prob:', succ_prob)
    #
    # print(simul_probs_mat)



    #########################################################
    ## Test 4: SATWAP - dim=4, k=4
    #########################################################

    # print('\nTest4: SATWAP - dim=4 k=4 - Test performance against Xtalk')
    #
    # dim = 4
    # k_sub = 4
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
    # num_alphas_scan = 20
    # num_s_par_scan = num_alphas_scan
    # alpha_list = np.linspace(0.06, 0.5, num_alphas_scan)
    # spar_list = np.linspace(0.02, 0.25, num_s_par_scan)
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # dark_count_prob = 0.
    # transm = 1.
    #
    # xtalk_par_list = np.linspace(0, 0.25, 12)
    #
    # MARCUS_scan_results = []
    # for this_xtalk_par in xtalk_par_list:
    #     print('   Doing xtalk_par:', this_xtalk_par)
    #     for alpha in alpha_list:
    #         print('Doing alpha:', alpha)
    #         for s_par in spar_list:
    #
    #             simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                    U_out, U_tilde,
    #                                                                    ancillas_param=alpha,
    #                                                                    s_par_photons=s_par,
    #                                                                    transm=transm,
    #                                                                    dark_counts_prob=dark_count_prob,
    #                                                                    xtalk_par=this_xtalk_par,
    #                                                                    ancilla_type=anc_type,
    #                                                                    number_resolving_det=False,
    #                                                                    parallelized=False)
    #
    #             simul_probs = abs(np.array(simul_probs_list))
    #             simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    #             this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    #             MARCUS_scan_results.append(np.array([alpha, s_par, this_xtalk_par, this_K, succ_prob]))
    #
    # MARCUS_scan_results_array = np.array(MARCUS_scan_results)

    # # ###################### SAVE DATA
    # filename = 'MARCUSscan_withNoise_scanXtalk_dim' + str(dim) + '_k' + str(k_sub) + \
    #            '_numpoints' + str(num_alphas_scan) + '_t' + str(transm) + '_DCprob' + str(dark_count_prob)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    # print('\nResults saved in: ' + full_saving_name)



    #########################################################
    ## Test 4: SATWAP - dim=4, k=4
    #########################################################

    print('\nTest4: SATWAP - dim=4 k=4 - Test performance against Xtalk')

    dim = 4
    k_sub = 4

    U_tilde = np.ones((dim + 1, dim + 1))
    for i in range(dim):
        U_tilde[i, i] = -2.
        U_tilde[dim, i] = np.sqrt(2)
        U_tilde[i, dim] = np.sqrt(2)
    U_tilde[dim, dim] = -1.
    U_tilde = U_tilde / 3.

    U_out = np.identity(dim)

    # anc_type = 'WeakCoherent'
    anc_type = 'TMS'
    # anc_type = 'HeraldedPhoton'

    num_alphas_scan = 20
    num_s_par_scan = num_alphas_scan
    alpha_list = np.linspace(0.06, 0.5, num_alphas_scan)
    spar_list = np.linspace(0.02, 0.25, num_s_par_scan)

    herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim

    MUB_U = get_U_MUB(dim, k_sub)
    U_a = MUB_U
    U_b = np.conj(MUB_U)

    transm = 1.
    xtalk_par = 0.

    dark_count_prob_list = np.logspace(-5, -3, num=8)

    MARCUS_scan_results = []
    for this_DC_par in dark_count_prob_list:
        print('   Doing dark_count_prob:', this_DC_par)
        for alpha in alpha_list:
            print('Doing alpha:', alpha)
            for s_par in spar_list:

                simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
                                                                       U_out, U_tilde,
                                                                       ancillas_param=alpha,
                                                                       s_par_photons=s_par,
                                                                       transm=transm,
                                                                       dark_counts_prob=this_DC_par,
                                                                       xtalk_par=xtalk_par,
                                                                       ancilla_type=anc_type,
                                                                       number_resolving_det=False,
                                                                       parallelized=False)

                simul_probs = abs(np.array(simul_probs_list))
                simul_probs_mat = np.reshape(simul_probs, (dim, dim))
                this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
                MARCUS_scan_results.append(np.array([alpha, s_par, this_DC_par, this_K, succ_prob]))

    MARCUS_scan_results_array = np.array(MARCUS_scan_results)

    # ###################### SAVE DATA
    filename = 'MARCUSscan_withNoise_scanDarkCounts_dim' + str(dim) + '_k' + str(k_sub) + \
               '_numpoints' + str(num_alphas_scan) + '_t' + str(transm)
    full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    print('\nResults saved in: ' + full_saving_name)


    #########################################################
    ## Test 5: SATWAP - dim=4, k=2
    #########################################################

    # print('\nTest5: SATWAP - dim=4 k=2 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 4
    # k_sub = 2
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
    # alpha = 0.05
    # s_par = 0.03
    #
    # transm = 1
    # dark_counts_prob = 0
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                        U_out, U_tilde,
    #                                                        ancillas_param=alpha,
    #                                                        s_par_photons=s_par,
    #                                                        transm=transm,
    #                                                        dark_counts_prob=dark_count_prob,
    #                                                        ancilla_type=anc_type,
    #                                                        number_resolving_det=False,
    #                                                        parallelized=False)
    #
    # simul_probs = abs(np.array(simul_probs_list))
    # simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    # this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    # print(this_K)
    #
    # print(simul_probs_mat)


    #########################################################
    ## Test 4: SATWAP - dim=4, k=2
    #########################################################

    # print('\nTest4: SATWAP - dim=4 k=2 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 4
    # k_sub = 2
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
    # num_alphas_scan = 20
    # num_s_par_scan = num_alphas_scan
    # alpha_list = np.linspace(0.06, 0.5, num_alphas_scan)
    # spar_list = np.linspace(0.02, 0.25, num_s_par_scan)
    #
    # herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # dark_count_prob = 0.
    # transm = 1.
    #
    # xtalk_par_list = np.linspace(0, 0.1, 12)
    #
    # MARCUS_scan_results = []
    # for this_xtalk_par in xtalk_par_list:
    #     print('   Doing xtalk_par:', this_xtalk_par)
    #     for alpha in alpha_list:
    #         print('Doing alpha:', alpha)
    #         for s_par in spar_list:
    #
    #             simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                    U_out, U_tilde,
    #                                                                    ancillas_param=alpha,
    #                                                                    s_par_photons=s_par,
    #                                                                    transm=transm,
    #                                                                    dark_counts_prob=dark_count_prob,
    #                                                                    xtalk_par=this_xtalk_par,
    #                                                                    ancilla_type=anc_type,
    #                                                                    number_resolving_det=False,
    #                                                                    parallelized=False)
    #
    #             simul_probs = abs(np.array(simul_probs_list))
    #             simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    #             this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    #             MARCUS_scan_results.append(np.array([alpha, s_par, this_xtalk_par, this_K, succ_prob]))
    #
    # MARCUS_scan_results_array = np.array(MARCUS_scan_results)
    #
    # # ###################### SAVE DATA
    # filename = 'MARCUSscan_withNoise_scanXtalk_dim' + str(dim) + '_k' + str(k_sub) + \
    #            '_numpoints' + str(num_alphas_scan) + '_t' + str(transm) + '_DCprob' + str(dark_count_prob)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    # print('\nResults saved in: ' + full_saving_name)


    #########################################################
    ## Test 4: SATWAP - dim=4, k=2
    #########################################################

    print('\nTest4: SATWAP - dim=4 k=2 - Test performance against Dark Counts')

    dim = 4
    k_sub = 2

    U_tilde = np.ones((dim + 1, dim + 1))
    for i in range(dim):
        U_tilde[i, i] = -2.
        U_tilde[dim, i] = np.sqrt(2)
        U_tilde[i, dim] = np.sqrt(2)
    U_tilde[dim, dim] = -1.
    U_tilde = U_tilde / 3.

    U_out = np.identity(dim)

    # anc_type = 'WeakCoherent'
    anc_type = 'TMS'
    # anc_type = 'HeraldedPhoton'

    num_alphas_scan = 20
    num_s_par_scan = num_alphas_scan
    alpha_list = np.linspace(0.06, 0.5, num_alphas_scan)
    spar_list = np.linspace(0.02, 0.25, num_s_par_scan)

    herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim

    MUB_U = get_U_MUB(dim, k_sub)
    U_a = MUB_U
    U_b = np.conj(MUB_U)

    transm = 1.
    xtalk_par = 0.

    dark_count_prob_list = np.logspace(-5, -2, num=12)

    MARCUS_scan_results = []
    for this_DC_par in dark_count_prob_list:
        print('   Doing dark_count_prob:', this_DC_par)
        for alpha in alpha_list:
            print('Doing alpha:', alpha)
            for s_par in spar_list:

                simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
                                                                       U_out, U_tilde,
                                                                       ancillas_param=alpha,
                                                                       s_par_photons=s_par,
                                                                       transm=transm,
                                                                       dark_counts_prob=this_DC_par,
                                                                       xtalk_par=xtalk_par,
                                                                       ancilla_type=anc_type,
                                                                       number_resolving_det=False,
                                                                       parallelized=False)

                simul_probs = abs(np.array(simul_probs_list))
                simul_probs_mat = np.reshape(simul_probs, (dim, dim))
                this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
                MARCUS_scan_results.append(np.array([alpha, s_par, this_DC_par, this_K, succ_prob]))

    MARCUS_scan_results_array = np.array(MARCUS_scan_results)

    # ###################### SAVE DATA
    filename = 'MARCUSscan_withNoise_scanDarkCounts_dim' + str(dim) + '_k' + str(k_sub) + \
               '_numpoints' + str(num_alphas_scan) + '_t' + str(transm)
    full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    print('\nResults saved in: ' + full_saving_name)

    #########################################################
    ## Test 5: SATWAP - dim=2, k=2
    #########################################################

    # print('\nTest5: SATWAP - dim=2 k=2 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 2
    # k_sub = 2
    #
    # U_tilde = np.identity(dim+1)
    #
    # U_out = np.array([[0, 1], [1, 0]])
    #
    # # anc_type = 'WeakCoherent'
    # anc_type = 'TMS'
    # # anc_type = 'HeraldedPhoton'
    #
    # alpha = 0.05
    # s_par = 0.03
    #
    # transm = 1
    # dark_counts_prob = 0
    #
    # herald_pattern = np.array([0, 1]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                        U_out, U_tilde,
    #                                                        ancillas_param=alpha,
    #                                                        s_par_photons=s_par,
    #                                                        transm=transm,
    #                                                        dark_counts_prob=dark_count_prob,
    #                                                        ancilla_type=anc_type,
    #                                                        number_resolving_det=False,
    #                                                        parallelized=False)
    #
    # simul_probs = abs(np.array(simul_probs_list))
    # simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    # this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    # print(this_K)
    #
    # print(simul_probs_mat)


    #########################################################
    ## Test 5: SATWAP - dim=2, k=2
    #########################################################

    # print('\nTest5: SATWAP - dim=2 k=2 - Test performance against amplitude and squeezing parameters')
    #
    # dim = 2
    # k_sub = 2
    #
    # U_tilde = np.identity(dim+1)
    #
    # U_out = np.array([[0, 1], [1, 0]])
    #
    # # anc_type = 'WeakCoherent'
    # anc_type = 'TMS'
    # # anc_type = 'HeraldedPhoton'
    #
    # alpha = 0.05
    #
    # num_s_par_scan = 400
    # spar_list = np.linspace(0.01, 0.4, num_s_par_scan)
    #
    # dark_count_prob = 0
    # transm = 1
    #
    # herald_pattern = np.array([0, 1]) + 0 * dim
    #
    # MUB_U = get_U_MUB(dim, k_sub)
    # U_a = MUB_U
    # U_b = np.conj(MUB_U)
    #
    # xtalk_par_list = np.linspace(0, 0.1, 40)
    #
    # MARCUS_scan_results = []
    # for this_xtalk_par in xtalk_par_list:
    #     print('   Doing xtalk_par:', this_xtalk_par)
    #     for s_par in spar_list:
    #         print('Doing s_par:', s_par)
    #         simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
    #                                                                U_out, U_tilde,
    #                                                                ancillas_param=alpha,
    #                                                                s_par_photons=s_par,
    #                                                                transm=transm,
    #                                                                dark_counts_prob=dark_count_prob,
    #                                                                xtalk_par=this_xtalk_par,
    #                                                                ancilla_type=anc_type,
    #                                                                number_resolving_det=False,
    #                                                                parallelized=False)
    #
    #         simul_probs = abs(np.array(simul_probs_list))
    #         simul_probs_mat = np.reshape(simul_probs, (dim, dim))
    #         this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
    #         MARCUS_scan_results.append(np.array([alpha, s_par, this_xtalk_par, this_K, succ_prob]))
    #
    # MARCUS_scan_results_array = np.array(MARCUS_scan_results)
    #
    # # ###################### SAVE DATA
    # filename = 'MARCUSscan_withNoise_scanXtalk_dim' + str(dim) + '_k' + str(k_sub) + \
    #            '_numpoints' + str(num_s_par_scan) + '_t' + str(transm) + '_DCprob' + str(dark_count_prob)
    # full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    # np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    # print('\nResults saved in: ' + full_saving_name)

    #########################################################
    ## Test 5: SATWAP - dim=2, k=2
    #########################################################

    print('\nTest5: SATWAP - dim=2 k=2 - Test performance against Dark Counts')

    dim = 2
    k_sub = 2

    U_tilde = np.identity(dim+1)

    U_out = np.array([[0, 1], [1, 0]])

    # anc_type = 'WeakCoherent'
    anc_type = 'TMS'
    # anc_type = 'HeraldedPhoton'

    alpha = 0.05

    num_s_par_scan = 400
    spar_list = np.linspace(0.01, 0.4, num_s_par_scan)

    transm = 1
    xtalk_par = 0

    herald_pattern = np.array([0, 1]) + 0 * dim

    MUB_U = get_U_MUB(dim, k_sub)
    U_a = MUB_U
    U_b = np.conj(MUB_U)

    dark_count_prob_list = np.logspace(-6, -3, num=30)

    MARCUS_scan_results = []
    for this_DC_par in dark_count_prob_list:
        print('   Doing dark_count_prob:', this_DC_par)
        for s_par in spar_list:
            print('Doing s_par:', s_par)
            simul_probs_list, succ_prob = HighD_Swapping_simulator(dim, U_a, U_b, herald_pattern,
                                                                   U_out, U_tilde,
                                                                   ancillas_param=alpha,
                                                                   s_par_photons=s_par,
                                                                   transm=transm,
                                                                   dark_counts_prob=this_DC_par,
                                                                   xtalk_par=xtalk_par,
                                                                   ancilla_type=anc_type,
                                                                   number_resolving_det=False,
                                                                   parallelized=False)

            simul_probs = abs(np.array(simul_probs_list))
            simul_probs_mat = np.reshape(simul_probs, (dim, dim))
            this_K = get_KeyRate_Marcus(simul_probs_mat, dim, k_sub)
            MARCUS_scan_results.append(np.array([alpha, s_par, this_DC_par, this_K, succ_prob]))

    MARCUS_scan_results_array = np.array(MARCUS_scan_results)

    # ###################### SAVE DATA
    filename = 'MARCUSscan_withNoise_scanDarkCounts_dim' + str(dim) + '_k' + str(k_sub) + \
               '_numpoints' + str(num_s_par_scan) + '_t' + str(transm)
    full_saving_name = os.path.join(DataSavingFolder, filename) + ".csv"
    np.savetxt(full_saving_name, MARCUS_scan_results_array, delimiter=",")
    print('\nResults saved in: ' + full_saving_name)

