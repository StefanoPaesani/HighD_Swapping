from HighD_Teleportation_SchemeDefinition import HighD_Teleportation_CoherentAncillas_simulator

import numpy as np
import matplotlib.pyplot as plt

#########################################################################################################
if __name__ == "__main__":
    from time import time

    #########################################################
    ## Test 1: Single state teleportation - dim=3
    #########################################################

    print("\nTest 1: Teleportation of a single state in d=3.")

    dim = 3
    U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    U_out = np.identity(dim)

    teleported_state = [1, 0, 0]
    state_projection = teleported_state

    alpha = 0.1
    normalized_prob = True

    herald_pattern = np.array([0, 1, 2]) + 0 * dim

    print()
    print("Teleported state:", teleported_state)
    print("Heralding pattern:", herald_pattern)
    print("Amplitude of ancilla weak-coherent light:", alpha)

    teleport_fid, success_prob = HighD_Teleportation_CoherentAncillas_simulator(dim, teleported_state,
                                                                                state_projection,
                                                                                herald_pattern,
                                                                                U_out, U_tilde,
                                                                                alpha_ancillas=alpha,
                                                                                s_par_photons=0.01,
                                                                                normalize_output=normalized_prob,
                                                                                number_resolving_det=False)

    print('Teleported state fidelity:', teleport_fid)
    print('Success probability:', success_prob)

    #########################################################
    ## Test 2: Teleportation in d=3 performance vs weak-coherent amplitude of ancillas - single state
    #########################################################

    print("\nTest 2: Teleportation in d=3 performance vs weak-coherent amplitude of ancillas - single state")

    dim = 3
    U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    U_out = np.identity(dim)

    teleported_state = [1, 1.j, -1]
    state_projection = teleported_state

    alpha_list = np.linspace(0.05, 1, 101)

    norm_prob = True

    herald_pattern = np.array([0, 1, 2]) + 0 * dim

    print()
    print("Teleported state:", teleported_state)
    print("Heralding pattern:", herald_pattern)

    teleport_fid_list = []
    success_prob_list = []
    for alpha in alpha_list:
        teleport_fid, success_prob = HighD_Teleportation_CoherentAncillas_simulator(dim, teleported_state,
                                                                                    state_projection,
                                                                                    herald_pattern,
                                                                                    U_out, U_tilde,
                                                                                    alpha_ancillas=alpha,
                                                                                    s_par_photons=0.01,
                                                                                    normalize_output=norm_prob,
                                                                                    number_resolving_det=False)
        teleport_fid_list.append(teleport_fid)
        success_prob_list.append(success_prob)

    # Plot them together
    fig, ax = plt.subplots()
    ax.plot(alpha_list, teleport_fid_list, color='red', label='Fid.')
    ax.set_xlabel(r'Weak-coherent ancilla amplitude $\alpha$')
    ax.set_ylabel('Teleported state fidelity', color='red')
    ax.set_title('d:' + str(dim) + ', state:' + str(teleported_state) + ', heralding pattern:' + str(herald_pattern))
    ax2 = ax.twinx()
    ax2.plot(alpha_list, success_prob_list, color='blue', label='Succ.Prob.')
    ax2.set_ylabel('Success probability', color='blue')
    plt.show()

    #########################################################
    ## Test 3: Teleportation in d=3 performance vs weak-coherent amplitude of ancillas - many random states
    #########################################################

    print("\nTest 3: Teleportation in d=3 performance vs weak-coherent amplitude of ancillas - many random states")

    dim = 3
    U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    U_out = np.identity(dim)

    alpha_list = np.linspace(0.05, 1, 101)

    normalized_prob = True

    herald_pattern = np.array([0, 1, 2]) + 0 * dim

    print()
    print("Teleported state:", teleported_state)
    print("Heralding pattern:", herald_pattern)

    num_random_states = 100

    teleport_fid_list_list = []
    success_prob_list_list = []

    start_time = time()
    for state_ix in range(num_random_states):

        teleported_state = np.random.rand(dim) + 1.j * np.random.rand(dim)
        # print('Testing random state ' + str(state_ix + 1) + ' of ' + str(num_random_states) + ': ', teleported_state)
        state_projection = teleported_state

        teleport_fid_list = []
        success_prob_list = []
        for alpha in alpha_list:
            teleport_fid, success_prob = HighD_Teleportation_CoherentAncillas_simulator(dim, teleported_state,
                                                                                        state_projection,
                                                                                        herald_pattern,
                                                                                        U_out, U_tilde,
                                                                                        alpha_ancillas=alpha,
                                                                                        s_par_photons=0.01,
                                                                                        normalize_output=norm_prob,
                                                                                        number_resolving_det=False)
            teleport_fid_list.append(teleport_fid)
            success_prob_list.append(success_prob)

        teleport_fid_list_list.append(teleport_fid_list)
        success_prob_list_list.append(success_prob_list)

    end_time = time()
    print('Time used:', end_time - start_time, 's')

    # Plot them together
    fig, ax = plt.subplots()
    for teleport_fid_list in teleport_fid_list_list:
        ax.plot(alpha_list, teleport_fid_list, color='red', alpha=0.1, label='Fid.')
    ax.set_xlabel(r'Weak-coherent ancilla amplitude $\alpha$')
    ax.set_ylabel('Teleported state fidelity', color='red')
    ax.set_title('d:' + str(dim) + ', heralding pattern:' + str(herald_pattern))
    ax2 = ax.twinx()
    for success_prob_list in success_prob_list_list:
        ax2.plot(alpha_list, success_prob_list, color='blue', alpha=0.1, label='Succ.Prob.')
    ax2.set_ylabel('Success probability', color='blue')
    plt.show()

    #########################################################
    ## Test 4: Single state teleportation - dim=4
    #########################################################

    print("\nTest 4: Teleportation of a single state in d=4.")

    dim = 4
    U_tilde = np.array([[-1 if j == i else +1 for j in range(dim + 1)] for i in range(dim + 1)]) / np.sqrt(dim + 1)
    U_out = np.identity(dim)

    teleported_state = [1, 0, 0, 0]
    state_projection = teleported_state

    alpha = 0.1
    normalized_prob = True

    herald_pattern = np.array([0, 1, 2, 3]) + 0 * dim

    print()
    print("Teleported state:", teleported_state)
    print("Heralding pattern:", herald_pattern)
    print("Amplitude of ancilla weak-coherent light:", alpha)

    teleport_fid, success_prob = HighD_Teleportation_CoherentAncillas_simulator(dim, teleported_state,
                                                                                state_projection,
                                                                                herald_pattern,
                                                                                U_out, U_tilde,
                                                                                alpha_ancillas=alpha,
                                                                                s_par_photons=0.01,
                                                                                normalize_output=norm_prob,
                                                                                number_resolving_det=False)

    print('Teleported state fidelity:', teleport_fid)
    print('Success probability:', success_prob)
