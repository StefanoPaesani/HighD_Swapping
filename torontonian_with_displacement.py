import numpy as np
from itertools import chain, combinations

def powersetiter(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def tor_with_displ(M, r):
    """ Compute the Torontonian_with_displacement of a Gaussian state.
    Equivalent to the Torontonian for cases with non-zero displacement.
    See https://arxiv.org/abs/1807.01639 for Torontonian details.
    """
    m = len(M)
    assert(M.shape == (m, m))
    assert(m % 2 == 0)
    n = m//2
    ssum = 0.0
    for i in powersetiter(range(n)):
        ia = np.array(i)
        ii = list(np.concatenate((ia, ia+n), axis=0))

        # matrix using elements associated to the set
        MMs = M[np.ix_(ii, ii)]
        rrs = r[ii]

        # ee = np.exp(rrs @ MMs @ rrs.conj() * (-0.5)).real
        ee = np.exp(rrs.conj() @ MMs @ rrs * (-0.5)).real
        print(np.exp(rrs.conj() @ MMs @ rrs * (-0.5)).imag)

        # matrix using elements associated to the complementary set
        Ms = np.delete(M, ii, axis=0)
        Ms = np.delete(Ms, ii, axis=1)

        ll = len(Ms)
        if ll != 0: #Check it is not the "empty matrix"
            dd = np.linalg.det(Ms).real
            print(np.linalg.det(Ms).imag)
        else:
            dd = 1
        ssum += ((-1)**(len(i))) * ee / np.sqrt(dd)

    return ssum