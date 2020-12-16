"""
Function to simulate photon-counting in GBS with threshold detectors and non-zero amplitude.

Based on the new formula derived by Jake via conditional probabilities of multigaussian distributions.
It also uses formula 5.144 from Serafini's 'Quantum Continuous Variables', suggested to us by Juan Miguel Arrazola
and Nicolas Quesada.

Stefano Paesani & Jake F. Bulmer, Dec. 2020
Update 15/12: Jake added numba interface to speed up the code.
"""

import numpy as np
# from itertools import chain, combinations
# from thewalrus.quantum import Qmat

import numba


@numba.jit(nopython=True)
def combinations(pool, r):
    n = len(pool)
    indices = list(range(r))
    empty = not (n and (0 < r <= n))

    if not empty:
        result = [pool[i] for i in indices]
        yield result

    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1

            result = [pool[i] for i in indices]
            yield result


@numba.jit(nopython=True)
def powerset(S):
    n = len(S)
    for i in range(n + 1):
        for s in combinations(S, i):
            yield s


@numba.jit(nopython=True)
def nb_block(X):
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)


@numba.jit(nopython=True)
def numba_ix(arr, rows, cols):
    return arr[rows][:, cols]


@numba.jit(nopython=True)
def Qmat_numba(cov, hbar=2):
    # numba compatible version of thewalrus.quantum Qmat
    r"""Returns the :math:`Q` Husimi matrix of the Gaussian state.

    Args:
        cov (array): :math:`2N\times 2N xp-` Wigner covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the :math:`Q` matrix.
    """
    # number of modes
    N = len(cov) // 2
    I = np.identity(N)

    x = cov[:N, :N] * (2. / hbar)
    xp = cov[:N, N:] * (2. / hbar)
    p = cov[N:, N:] * (2. / hbar)
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = nb_block(((aidaj, aiaj.conj()), (aiaj, aidaj.conj()))) + np.identity(2 * N)
    return Q


@numba.jit(nopython=True)
def threshold_detection_prob(covM, means, det_pattern, hbar=2):
    det_pattern = np.asarray(det_pattern)

    m = len(covM)
    assert (covM.shape == (m, m))
    assert (m % 2 == 0)
    n = m // 2

    means_x = means[:n]
    means_p = means[n:]
    avec = np.concatenate((means_x + 1j * means_p, means_x - 1j * means_p), axis=0) / np.sqrt(2 * hbar)

    Q = Qmat_numba(covM, hbar=hbar)

    if max(det_pattern) > 1:
        raise ValueError("When using threshold detectors, the detection pattern can contain only 1s or 0s.")

    nonzero_idxs = np.where(det_pattern == 1)[0]
    zero_idxs = np.where(det_pattern == 0)[0]

    ii1 = np.concatenate((nonzero_idxs, nonzero_idxs + n), axis=0)
    ii0 = np.concatenate((zero_idxs, zero_idxs + n), axis=0)

    Qaa = numba_ix(Q, ii0, ii0)
    Qab = numba_ix(Q, ii0, ii1)
    Qba = numba_ix(Q, ii1, ii0)
    Qbb = numba_ix(Q, ii1, ii1)

    Qaa_inv = np.linalg.inv(Qaa)
    Qcond = Qbb - Qba @ Qaa_inv @ Qab

    avec_a = avec[ii0]
    avec_b = avec[ii1]
    avec_cond = avec_b - Qba @ Qaa_inv @ avec_a

    p0a_fact_exp = np.exp(avec_a @ Qaa_inv @ avec_a.conj() * (-0.5)).real
    p0a_fact_det = np.sqrt(np.linalg.det(Qaa).real)
    p0a = p0a_fact_exp / p0a_fact_det

    n_det = len(nonzero_idxs)
    p_sum = 1.  # empty set is not included in the powerset function so we start at 1
    for z in powerset(np.arange(n_det)):
        Z = np.asarray(z)
        ZZ = np.concatenate((Z, Z + n_det), axis=0)

        avec0 = avec_cond[ZZ]
        Q0 = numba_ix(Qcond, ZZ, ZZ)
        Q0inv = np.linalg.inv(Q0)

        fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
        fact_det = np.sqrt(np.linalg.det(Q0).real)

        p_sum += ((-1) ** len(Z)) * fact_exp / fact_det

    return p0a * p_sum


@numba.jit(nopython=True, parallel=True)
def threshold_detection_prob_parallel(covM, means, det_pattern, hbar=2, chunk_size=1000):
    det_pattern = np.asarray(det_pattern)

    m = len(covM)
    assert (covM.shape == (m, m))
    assert (m % 2 == 0)
    n = m // 2

    means_x = means[:n]
    means_p = means[n:]
    avec = np.concatenate((means_x + 1j * means_p, means_x - 1j * means_p), axis=0) / np.sqrt(2 * hbar)

    Q = Qmat_numba(covM, hbar=hbar)

    if max(det_pattern) > 1:
        raise ValueError("When using threshold detectors, the detection pattern can contain only 1s or 0s.")

    nonzero_idxs = np.where(det_pattern == 1)[0]
    zero_idxs = np.where(det_pattern == 0)[0]

    ii1 = np.concatenate((nonzero_idxs, nonzero_idxs + n), axis=0)
    ii0 = np.concatenate((zero_idxs, zero_idxs + n), axis=0)

    Qaa = numba_ix(Q, ii0, ii0)
    Qab = numba_ix(Q, ii0, ii1)
    Qba = numba_ix(Q, ii1, ii0)
    Qbb = numba_ix(Q, ii1, ii1)

    Qaa_inv = np.linalg.inv(Qaa)
    Qcond = Qbb - Qba @ Qaa_inv @ Qab

    avec_a = avec[ii0]
    avec_b = avec[ii1]
    avec_cond = avec_b - Qba @ Qaa_inv @ avec_a

    p0a_fact_exp = np.exp(avec_a @ Qaa_inv @ avec_a.conj() * (-0.5)).real
    p0a_fact_det = np.sqrt(np.linalg.det(Qaa).real)
    p0a = p0a_fact_exp / p0a_fact_det

    n_det = len(nonzero_idxs)

    powset = powerset(np.arange(n_det))
    looping = True
    powset_size = 2 ** n_det - 1

    j = 0
    p_sum = 1  # empty set is not included in the powerset function so we start at 1

    # break loops into chunks where sets are stored in memory
    # and can be computed in parallel
    while looping:
        # generate some sets for the chunk
        i = 0
        chunk_sets = []
        while i < chunk_size and j < powset_size:
            chunk_sets.append(next(powset))
            i += 1
            j += 1

        if j == powset_size:
            chunk_size = len(chunk_sets)
            looping = False

        # loop over the sets in the chunk
        ps = np.zeros(chunk_size)
        for i in numba.prange(chunk_size):
            Z = np.asarray(chunk_sets[i])
            ZZ = np.concatenate((Z, Z + n_det), axis=0)

            avec0 = avec_cond[ZZ]
            Q0 = numba_ix(Qcond, ZZ, ZZ)
            Q0inv = np.linalg.inv(Q0)

            fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
            fact_det = np.sqrt(np.linalg.det(Q0).real)

            ps[i] = ((-1) ** len(Z)) * fact_exp / fact_det
        p_sum += ps.sum()

    return p0a * p_sum

