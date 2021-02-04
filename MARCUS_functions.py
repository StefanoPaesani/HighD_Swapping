import numpy as np
from itertools import product


################################
#####   MARCUS FUNCTIONS   #####
################################

def get_U_MUB(d, k):
    omega = np.exp(2.j * np.pi / k)
    temp_U = np.zeros((d, d), dtype=np.cdouble)
    for m in range(int(d/k)):
        for i in range(k):
            for j in range(k):
                temp_U[m*k + i, m*k + j] = omega**(i*j)
    return temp_U/np.sqrt(k)


def get_W(prob_matr, d, k, m):
    prob_m = np.sum(prob_matr[np.ix_(range(m*k, (m+1)*k), range(m*k, (m+1)*k))])
    return sum([prob_matr[m*k + i, m*k + i]/prob_m for i in range(k)])


def get_Hmin(prob_matr, d, k, m):
    W = get_W(prob_matr, d, k, m)
    return -np.log2(((np.sqrt(W) + np.sqrt((k-1)*(1 - W)))**2)/k)

def get_Hshannon(prob_matr, d):
    my_sum = 0
    for j in range(d):
        p_yj = sum(prob_matr[j])
        for i in range(d):
            if prob_matr[j, i] > 0:
                my_sum += prob_matr[j, i] * np.log2(prob_matr[j, i] / p_yj)
    return - my_sum


def get_KeyRate_Marcus(prob_matr, d, k):
    all_Ks = [get_Hmin(prob_matr, d, k, m) - get_Hshannon(prob_matr, d) for m in range(int(d / k))]
    return max(all_Ks)

#####################################################################
if __name__ == "__main__":
    dim = 4
    k_sub = 4

    asd = get_U_MUB(dim, k_sub)
    print(asd)
    print(asd @ np.conj(asd.T))

