from __future__ import division, print_function
import numpy as np
import lintable as lin
import fermihubbard

import scipy.sparse as sparse

# The model parameters:

# The onsite energies (0)
H1E = np.zeros((4, 4))

# Hopping on a ring
H1T = np.zeros((4, 4))
for i in xrange(4):
    H1T[i, (i+1) % 4] = -1.
    H1T[(i+1) % 4, i] = -1.

# Only onsite interaction
H1U = 2*np.eye(4)

# Construc the model
m = fermihubbard.Model(H1E, H1T, H1U)

# Construct a specific chargestate with eight electrons:
ns4 = m.numbersector(4)

# Consider the sector with net zero spin in the z direction, meaning that
# 4 electrons = 2 spin-up electrons + 2 spin-down electrons
s0 = ns4.szsector(0)

# Let us then take a look at the basis. Generate the Lin-table:
lita = lin.Table(4, 2, 2)

# Print the basis
print('Spin-up basis:')
print(lita.basisu)
# Spin-up basis:
# [[0 0 1 1]
#  [0 1 0 1]
#  [0 1 1 0]
#  [1 0 0 1]
#  [1 0 1 0]
#  [1 1 0 0]]

# Compute the Hamiltonian
(HTu, HTd, HUd) = s0.hamiltonian

# Print the hopping Hamiltonian for the spin-up electronic systems
print('Spin-up hopping:')
print(HTu)

# The Total hamiltonian can be generated from these parts.
Nu, Nd = lita.Ns

# The interaction part in sparse format
HU = sparse.coo_matrix((HUd, (np.arange(Nu*Nd), np.arange(Nu*Nd))), shape=(Nu*Nd, Nu*Nd)).tocsr()

# The kronecker product of the two hopping sectors.
HT = sparse.kron(np.eye(Nd), HTu, 'dok') + sparse.kron(HTd, np.eye(Nu), 'dok')

print('The total Hamiltonian:')
print(HU + HT)
