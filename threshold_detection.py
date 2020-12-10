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


def threshold_detection_prob(gauss_state, det_pattern, hbar=2):
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
    nonzero_idxs = [this_mode for this_mode, phot_num in enumerate(det_pattern) if phot_num > 0]
    zero_idxs = [this_mode for this_mode, phot_num in enumerate(det_pattern) if phot_num == 0]

    p_sum = 0
    for i in powerset(nonzero_idxs):
        i0 = list(i)
        ia = np.array(zero_idxs + i0)
        ii = list(np.concatenate((ia, ia + n), axis=0))

        avec0 = avec[ii]
        Q0 = Q[np.ix_(ii, ii)]
        Q0inv = np.linalg.inv(Q0)

        fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
        fact_det = np.sqrt(np.linalg.det(Q0).real)

        p_sum += ((-1) ** len(i0)) * fact_exp / fact_det

    return p_sum
