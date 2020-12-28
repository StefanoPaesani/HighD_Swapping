import numpy as np
from itertools import product


################################
#####   SATWAP FUNCTIONS   #####
################################


def get_g(x, d, m=2):
    return 1. / np.tan(np.pi * (x + (0.5 / m)) / d)


def get_alpha(k, d, m=2):
    return np.tan(np.pi / (2 * m)) * (get_g(k, d, m) - get_g(int(d / 2.), d, m)) / (2 * d)


def get_beta(k, d, m=2):
    return np.tan(np.pi / (2 * m)) * (get_g(k + 1 - (1 / m), d, m) + get_g(int(d / 2.), d, m)) / (2 * d)


def get_a(ell, d, m=2):
    w = np.exp(2. * np.pi * 1.j / d)
    return sum([get_alpha(k, d, m) * (w ** (-(k * ell))) - get_beta(k, d, m) * (w ** ((k + 1) * ell))
                for k in range(int(d / 2.))])


def get_AB(k, ell, x, y, prob_matr, d, m=2):
    w = np.exp(2. * np.pi * 1.j / d)
    return sum([(w ** ((a * k) + (b * ell))) * prob_matr[(y - 1) + m * (x - 1), b + d * a] for (a, b) in
                product(range(d), repeat=2)])


def get_AB_tilde(k, ell, x, y, prob_matr, d, m=2):
    term1 = get_a(ell, d, m) * get_AB(k, d - ell, x, y, prob_matr, d)
    if y == 1:
        w = np.exp(2. * np.pi * 1.j / d)
        term2 = (w ** ell) * np.conj(get_a(ell, d, m)) * get_AB(k, d - ell, x, m, prob_matr, d)
    else:
        term2 = np.conj(get_a(ell, d, m)) * get_AB(k, d - ell, x, y - 1, prob_matr, d)
    return term1 + term2


def get_I_tilde(prob_matr, d, m=2):
    return abs(sum([get_AB_tilde(ell, ell, i, i, prob_matr, d, m)
                    for (i, ell) in product(range(1, m + 1), range(1, d))]))


def tsirelson_bound(d, m=2):
    return m * (d - 1)


def classical_bound(d, m=2):
    return (1/2) * np.tan(np.pi/(2*m)) * (get_g(0, d, m)*(2*m - 1) - get_g(1 - (1/m), d, m)) - m


def SATWAP_rel_quantum_violation(Itilde_meas, d, m=2):
    SATWAP_Q = tsirelson_bound(d, m)
    SATWAP_C = classical_bound(d, m)
    return (Itilde_meas-SATWAP_C)/(SATWAP_Q-SATWAP_C)


####################################
#####   SIMULATION FUNCTIONS   #####
####################################
def get_bell_state(d):
    state = np.zeros(d ** 2)
    for i in range(d):
        state[i + d * i] = 1
    return state / np.sqrt(d)


def get_A_vec(x, a, d):
    if x == 2:
        alpha = 0
    if x == 1:
        alpha = 1 / 2
    return np.array([np.exp(i * (a + alpha) * 2 * np.pi * 1.j / d) for i in range(d)]) / np.sqrt(d)


def get_B_vec(y, b, d):
    if y == 2:
        beta = 1 / 4
    if y == 1:
        beta = -1 / 4
    return np.array([np.exp(i * (-b + beta) * 2 * np.pi * 1.j / d) for i in range(d)]) / np.sqrt(d)


def get_prob(state, x, y, a, b, d):
    state_norm = np.array(state)
    state_norm = state_norm / np.linalg.norm(state_norm)
    vec_a = get_A_vec(x, a, d)
    vec_b = get_B_vec(y, b, d)
    join_meas_vec = np.kron(vec_a, vec_b)
    return abs(np.conj(join_meas_vec) @ state_norm) ** 2


def theo_max_prob(x, y, a, b, d, m=2):
    theta = (x - 1 / 2) / m
    zeta = y / m
    gamma = 1 / np.sqrt(d)
    return abs(sum([gamma * np.exp((a - b - theta + zeta) * q * (2 * np.pi * 1.j) / d) for q in range(d)]) / d) ** 2


#####################################################
#####   MEASUREMENTS FOR EXPERIMENT/SIMULATOR   #####
#####################################################
# Assumes the state |00>+|11>+...|d-1,d-1> is used, didn't check if they still work for different phases.
# Uses m=2 (like CGLMP)


def get_SATWAP_meas_vects(d, m=2):
    return [[(get_A_vec(x, a, d), get_B_vec(y, b, d))
             for (a, b) in product(range(d), repeat=2)]
            for (x, y) in product(range(1, m + 1), repeat=2)]


def get_SATWAP_A_matrix(x, d):
    return np.conj(np.array([get_A_vec(x, a, d) for a in range(d)]))

def get_SATWAP_B_matrix(y, d):
    return np.conj(np.array([get_B_vec(y, b, d) for b in range(d)]))

def get_SATWAP_AB_matrices_list(d, m=2):
    return [(get_SATWAP_A_matrix(x, d), get_SATWAP_B_matrix(y, d)) for (x, y) in product(range(1, m + 1), repeat=2)]



#####################################################################
if __name__ == "__main__":
    num_meas = 2
    dim = 6

    state = get_bell_state(dim)

    probs = np.array([[get_prob(state, x, y, a, b, dim)
                       for (a, b) in product(range(dim), repeat=2)]
                      for (x, y) in product(range(1, num_meas + 1), repeat=2)])

    # print(probs)

    probs_theo = np.array([[theo_max_prob(x, y, a, b, dim)
                            for (a, b) in product(range(dim), repeat=2)]
                           for (x, y) in product(range(1, num_meas + 1), repeat=2)])

    # print()
    # print('theo probs')
    # print(probs_theo)
    #
    # print()
    # print('difference')
    # print(probs_theo-probs)

    print('Got:', get_I_tilde(probs, dim, m=num_meas))
    print('Tsirelson:', tsirelson_bound(dim, m=num_meas))
    print('Classical:', classical_bound(dim, m=num_meas))
