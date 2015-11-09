# File: fermihubbard.py
#
# Efficient but simplistic approach to generating the
# many-electron Hamiltonian for Fermi-Hubbard models.
# Note that this package is limited to spin 1/2 fermions.
#
# Can be used for exact diagonalization.
#
# Code released under a MIT license, by
# Kim G. L. Pedersen, 2015
# (unless otherwise noted in the function description)
#
# Waiver: No guarantees given. Please use **completely** at your own risk.

from __future__ import division, print_function
import warnings
import numpy as np
import scipy.sparse as sparse

import lintable as lin


class Model:
    """
    Defines some Bose-Hubbard model by specifying the onsite energies (omegas),
    the links between sites on the network, and
    the onsite interaction strength (U)

    Parameters
    ----------
    H1E : ndarray
        Diagonal matrix storing the onsite energies.
    H1T : ndarray
        The single particle hopping matrix (no diagonal part).
    H1U : ndarray
        The single particle interaction strength between electrons on different
        sites.
    """
    def __init__(self, H1E, H1T, H1U):
        self.n = H1E.shape[0]
        self.H1T = H1T
        self.H1E = H1E
        self.H1U = H1U

    @property
    def HU(self):
        return Model.HU_factory(self.H1U)

    @staticmethod
    def HU_factory(H1U):
        HUii = np.diag(H1U)
        HUij = H1U - np.diag(HUii)

        fcn = lambda statesu, statesd: \
            np.sum(.5 * ((statesu + statesd - 1).dot(HUij))*(statesu + statesd - 1), axis=1) \
            + ((statesu - .5)*(statesd - .5)).dot(HUii)

        return fcn

    @property
    def real(self):
        return np.sum(np.abs(np.imag(self.H1T))) < 1e-10

    def numbersector(self, ne):
        """
        Returns a specific particle number sector object based on this model.
        """

        return NumberSector(self, ne)


class NumberSector:
    """
    Defines a specific particle number sector of a given Fermi-Hubbard model.
    """
    def __init__(self, model, ne):
        self.model = model  # model
        self.ne = ne  # number of particle
        # cache
        self._cache = dict()

    def szsector(self, ds):
        """
        Returns a specific spin state indexed by the excess of spin up
        particles compared to spin down.

        Caches the result for later use.

        The difference must be commensurate with the number of particles.
        E.g. a difference ds = 2 is not "commensurate" with ne=3 particles.

        Parameters
        ----------
        ds : int
            ds = n(spin up) - n(spin down).
        """
        if ds % 2 != self.ne % 2:
            warnings.warn('Particle difference "{0}" not commensurate with total particle number'.format(ds))
            return False

        nu = int((ds + self.ne)/2)
        nd = self.ne - nu
        return SzSector(self.model, nu, nd)


class SzSector:
    """
    SpinState corresponding to a specific number of spin-up electrons,
    and spin-down electrons.

    Conventionially the many-electrons states are always built in the basis:
          basis[0]    = basisu[0] basisd[0]
          basis[1]    = basisu[1] basisd[0]
          basis[2]    = basisu[2] basisd[0]
          ...
          basis[Nu-1] = basisu[Nu-1] basisd[0]
          basis[Nu]   = basisu[0] basisd[1]
          basis[Nu+1] = basisu[1] basisd[1]
          ...
    and basis states have create_up operators before create_down operators.
    """
    def __init__(self, model, nu, nd):
        self.n = model.n
        self.model = model
        self.nu = nu
        self.nd = nd
        self.sz = nu - nd

    @property
    def hamiltonian(self):
        """
        Returns the Hamiltonian and caches the result for later retrieval.

        Returns
        -------
        A 3-tuple (H1Tu, H1Td, HU)
        """
        if not hasattr(self, '_hamiltonian'):
            self._lintable, self._hamiltonian = SzSector.generate_hamiltonian(self.model, self.nu + self.nd, self.nu)

        return self._hamiltonian

    @staticmethod
    def generate_hamiltonian(model, ne, nu):
        """
        Generates all the parts of the Hamiltonian
        in the specific Sz-spin state.

        Parameters
        ----------
        model : Model object
            A FermiHubbard Model
        ne : int
            Total number of electrons
        nu : int
            Total number of spin up electrons
        """
        n = model.n
        nd = ne - nu

        lita = lin.Table(n, nu, nd)  # create the lin table for this szstate

        if nu > n or nd > n:
            H = np.zeros((1, 1))
            return lita, (H, H, np.zeros((1,)))

        # hopping for spin up
        HTu = SzSector.hopping_hamiltonian(lita.basisu, lita.Juv, model.H1T,
                                           real=model.real)

        # hopping for spin down
        HTd = SzSector.hopping_hamiltonian(lita.basisd, lita.Jdv, model.H1T,
                                           real=model.real)

        HU = SzSector.interaction_hamiltonian(model, ne, nu)

        return lita, (HTu, HTd, HU)

    @staticmethod
    def interaction_hamiltonian(model, ne, nu, real=True):
        """
        The Coulomb energy of each states in our specific
        Sz subspace.

        Parameters
        ----------
        model : Model object
            FermiHubbard model object.
        ne : int
            Total number of electrons.
        nu : int
            Number of spin up electrons.
        real : boolean
            Whether or not the system has only real parameters.
        """
        n = model.n
        nd = ne-nu

        if nu > n or nd > n:
            return np.zeros((1,), dtype=np.float64 if real else np.complex128)

        lita = lin.Table(n, nu, nd)
        Nu, Nd = lita.Ns

        HU = model.HU  # the interaction function

        # The coulomb energy
        HUv = np.zeros(
            (Nd*Nu,),
            dtype=np.float64 if real else np.complex128)

        for cN in xrange(Nd):

            index = cN*Nu + np.arange(Nu)

            statesu, statesd = lin.index2state(
                index,
                n=n,
                Juv=lita.Juv,
                Jdv=lita.Jdv)   # get all states

            HUv[index] = HU(statesu, statesd) + \
                np.sum(statesd.dot(model.H1E), axis=1) + \
                np.sum(statesu.dot(model.H1E), axis=1)

        return HUv

    @staticmethod
    def hopping_hamiltonian(states, Jv, H1T, real=True):
        """
        The kinetic part of the Hamiltonian for a all electrons
        of a specific spin specie.

        Parameters
        ----------
        states: list
            List of many-electron states in each Sz spin sector.
        Jv : list
            List of Lin indices
        H1T : ndarray
            The hopping Hamiltonian.
        real : boolean
            Whether or not the system has only real parameters.
        """
        if len(Jv) == 1:
            return np.array([1, 1, 0])

        n = H1T.shape[0]                        # number of site

        ne = np.sum(states[0, :])               # number of electrons

        # only the upper triangular part
        H1T = np.triu(H1T)
        # count number of hopping elements
        nH1T = np.sum(H1T != 0)
        ts = np.transpose(np.nonzero(H1T))      # ts

        # J = {J: i for i, J in enumerate(Jv)}   # create a reverse index
        NJ = len(Jv)
        HS = sparse.dok_matrix(
            (NJ, NJ),
            dtype=np.float64 if real else np.complex128)

        for c in xrange(nH1T):

            H1H = np.eye(n, dtype=int)
            i = ts[c, 0]
            j = ts[c, 1]

            H1H[j, j] = 0
            H1H[i, i] = 0
            H1H[i, j] = 1

            vouts = states.dot(H1H).astype(int)
            vouts[states[:, i]+states[:, j] == 0, :] = 0
            vouts[states[:, i]+states[:, j] == 2, :] = 0

            idx = np.flatnonzero(np.sum(vouts, axis=1) == ne)

            if len(idx) > 0:
                touts = (-1)**np.mod(
                    np.sum(states[idx, min(i, j):max(i, j)], axis=1) - 1,
                    2)*H1T[i, j]
                HS[idx, lin.state2index(vouts[idx, :], Jv)] = touts

        return HS.tocsr() + HS.tocsr().H


#     $$\   $$\   $$\     $$\ $$\
#     $$ |  $$ |  $$ |    \__|$$ |
#     $$ |  $$ |$$$$$$\   $$\ $$ |
#     $$ |  $$ |\_$$  _|  $$ |$$ |
#     $$ |  $$ |  $$ |    $$ |$$ |
#     $$ |  $$ |  $$ |$$\ $$ |$$ |
#     \$$$$$$  |  \$$$$  |$$ |$$ |
#      \______/    \____/ \__|\__|

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    n = int(n)
    k = int(k)

    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
