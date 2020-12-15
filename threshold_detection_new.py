"""
Functions to simulate photon-counting in GBS with threshold detectors and non-zero amplitude.

Based on formula 5.144 from Serafini's 'Quantum Continuous Variables', highlighted to us by Juan Miguel Arrazola
and Nicolas Quesada.

A further ~quadratic speed-up in the polynomial term is likely to be possible (lots of big determinants in the sum could
be avoided).

Stefano Paesani & Jake F. Bulmer, Dec. 2020
"""

import numpy as np
from itertools import chain, combinations

from thewalrus.quantum import Qmat


def powerset(iterable):
    """powerset([1,2,3]) -->(,) (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def threshold_detection_prob_new(gauss_state, det_pattern, hbar=2):
    """
    Calculates the single photon-detection probability for a Gaussian state considering threshold detectors.
    """

    covM = gauss_state.cov()
    means = gauss_state.means()

    m = len(covM)
    assert (covM.shape == (m, m))
    assert (m % 2 == 0)
    n = m // 2

    means_x = means[:n]
    means_p = means[n:]
    avec = np.concatenate((means_x + 1.j * means_p, means_x - 1.j * means_p), axis=0) / np.sqrt(2 * hbar)

    Q = Qmat(covM, hbar=hbar)

    if max(det_pattern) > 1:
        raise ValueError("When using threshold detectors, the detection pattern can contain only 1s or 0s.")

    nonzero_idxs = np.array([this_mode for this_mode, phot_num in enumerate(det_pattern) if phot_num > 0])
    zero_idxs = np.array([this_mode for this_mode, phot_num in enumerate(det_pattern) if phot_num == 0])

    ii1 = list(np.concatenate((nonzero_idxs, nonzero_idxs + n), axis=0))
    ii0 = list(np.concatenate((zero_idxs, zero_idxs + n), axis=0))

    Qaa = Q[np.ix_(ii0, ii0)]
    Qab = Q[np.ix_(ii0, ii1)]
    Qba = Q[np.ix_(ii1, ii0)]
    Qbb = Q[np.ix_(ii1, ii1)]

    Qaa_inv = np.linalg.inv(Qaa)
    Qcond = Qbb - Qba @ Qaa_inv @ Qab

    avec_a = avec[ii0]
    avec_b = avec[ii1]
    avec_cond = avec_b - Qba @ Qaa_inv @ avec_a

    p0a_fact_exp = np.exp(avec_a @ Qaa_inv @ avec_a.conj() * (-0.5)).real
    p0a_fact_det = np.sqrt(np.linalg.det(Qaa).real)
    p0a = p0a_fact_exp / p0a_fact_det

    m = len(nonzero_idxs)

    p_sum = 0
    for i in powerset(range(m)):

        ia = np.array(i)
        ii = list(np.concatenate((ia, ia + m), axis=0))

        avec0 = avec_cond[ii]
        Q0 = Qcond[np.ix_(ii, ii)]
        Q0inv = np.linalg.inv(Q0)

        fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
        fact_det = np.sqrt(np.linalg.det(Q0).real)

        p_sum += ((-1) ** len(ia)) * fact_exp / fact_det

    return p0a * p_sum
