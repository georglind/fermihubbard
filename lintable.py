# File: lintable.py
#
# Simple and quick implementation of Lin tables for indexing Sz spin states.
# Used in the fermihubbard.py implementation.
#
# Code released under a MIT license, by
# Kim G. L. Pedersen, 2015
# (unless otherwise noted in the function description)
#
# Waiver: No guarantees given. Please use **completely** at your own risk.

from __future__ import division, print_function
import numpy as np
from itertools import permutations  # necessary in LinTable


class Table(object):

    def __init__(self, n, nu, nd):

        self.nu = nu
        self.nd = nd
        self.n = n

        if nu > n or nd > n:
            self.Jdu, self.Nu, self.basisu = (['1'], 1, [])
            self.Jdv, self.Nd, self.basisd = (['1'], 1, [])
        else:
            self.Juv, self.Nu, self.basisu = states(self.n, self.nu)
            self.Jdv, self.Nd, self.basisd = states(self.n, self.nd)

    @property
    def Js(self):
        """get J indices"""
        return {'u': self.Juv, 'd': self.Jdv}

    @property
    def Ns(self):
        """Get the Ns"""
        return (self.Nu, self.Nd)

    @property
    def N(self):
        return np.prod(self.Ns)

    @property
    def ns(self):
        return (self.nu, self.nd)

    @property
    def ne(self):
        return self.nu + self.nd


def states(n, nu):
    """
    Create all many-body spin states

    Parameters
    ----------
    n : int
        number of sites
    nu : int
        number of on spin-specie
    """
    x = [0]*(n-nu) + [1]*nu

    states = np.array(unique_permutations(x), dtype=int)

    N = states.shape[0]
    Jv = bi2de(states)

    return (Jv, N, states)


def state2index(states, Juv, Jdv=None):
    """
    Parameters
    ----------
    states : ndarray
        states to index
    Juv : list
        indexes of the spin-up subspace
    Jdv : list
        index of the spin-down subspace
    """

    Nu = Juv.shape[0]
    Ju = {J: i for i, J in enumerate(Juv)}

    if Jdv is None:
        if len(states.shape) < 2:
            states = np.array([states])
        Js = np.array([Ju[i] for i in bi2de(states)])
    else:
        # Nd = Jdv.shape[0]
        Jd = {J: i for i, J in enumerate(Jdv)}

        n = states.shape[1]/2

        Ius = bi2de(states[:, 1:n])
        Ids = bi2de(states[:, n+1:])

        Js = np.array([Jd[i] for i in Ids])*Nu + np.array([Ju[i] for i in Ius])

    return Js


def index2state(Is, n, Juv, Jdv=None):
    """
    Returns state with a given index

    Parameters
    ----------
    Is : ndarray
        list of indices
    n : int
        number of sites
    Juv : ndarray
        Lin table of spin-up states
    Jdv : ndarray
        Lin table for spin-down states
    """
    Nu = Juv.shape[0]

    if Jdv is None:
        Ius = np.mod(Is, Nu)
        states_up = de2bi(Juv[Ius], n)
        return states_up
    else:
        # Nd = Jdv.shape[0]
        Ius = np.mod(Is, Nu)
        Ids = np.floor(Is/Nu).astype(int)

        states_up = de2bi(Juv[Ius], n)
        states_down = de2bi(Jdv[Ids], n)

        return (states_up, states_down)


def unique_permutations(elements):
    """
    Get all unique permutations of a list of elements

    Parameters
    ----------
    elements : list
        a list containing the elements
    """
    n = len(elements)
    uniques = list(set(elements))
    nu = len(uniques)

    if not elements:
        return []
    elif n == 1 or nu == 1:
        return [elements]
    elif n == nu:
        ps = permutations(elements)
        return [list(p) for p in ps]
    else:
        pu = []
        # collect the results
        for i in np.arange(nu):
            # copy elements into v
            v = list(elements)
            # first instance of unique element
            ind = elements.index(uniques[i])
            # remove this element
            del v[ind]
            # extend the result
            pu.extend([[uniques[i]] + perm for perm in unique_permutations(v)])

    return pu


def bi2de(binaries):
    """
    Parameters
    ----------
    binaries : ndarray
        Here one row is one binary number.
    """
    n = binaries.shape[0]
    if len(binaries.shape) > 1:
        n = binaries.shape[1]
    decimals = np.dot(binaries, np.power(2, np.arange(n-1, -1, -1)))

    # print('d: {0}'.format(decimals))

    # if (decimals.size == 1):
        # return [decimals]

    return decimals


def de2bi(decimals, n=None):
    """
    Parameters
    ----------
    decimals : ndarray
        vector of decimals
    n : int
        number of binary digits
    """
    decimals = np.array(decimals)
    try:
        nd = np.ceil(np.log2(np.max(decimals)))
    except RuntimeWarning:
        print('{0}:{1}'.format(decimals, n))
    if n is None or n < nd:
        n = nd
    return np.remainder(np.floor(np.outer(decimals, np.power(2., np.arange(1-n,1)))), 2).astype(int)
