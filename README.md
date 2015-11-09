# fermihubbard

This is a small python library to generate the basis and many-body Hamiltonian for the Fermi-Hubbard model.

The Fermi-Hubbard model we are concerned with here is defined on some lattice is described by the Hamiltonian,

![The Fermi-Hubbard Hamiltonian](https://raw.githubusercontent.com/georglind/fermihubbard/master/figs/fermihubbard.png "The Fermi-Hubbard Hamiltonian")

Here the sum runs over the lattice indices and spin components (up and down), and the operator `c(i)` annihilates a particle on site `i`. Here the `n(i)` operator--appearing in the definition of the interaction operator--counts the total number of particles on site `i`. 

This library **requires** `numpy` and `scipy` installed on your system.

## How to use it?

As an example consider a ring with four sites. 

Construct the model from a onsite enegies, hopping amplitudes and interaction energy.

```python
import numpy as np
import lintable as lin
import fermihubbard

# The model parameters:

# The onsite energies (all zero)
H1E = np.zeros((4, 4))

# Hopping on a ring (amplitude t = -1)
H1T = np.zeros((4, 4))
for i in xrange(4):
    H1T[i, (i+1) % 4] = -1.
    H1T[(i+1) % 4, i] = -1.

# Interaction (her onsite only at U = 2)
H1U = 2*np.eye(4)

# Construc the model
m = fermihubbard.Model(H1E, H1T, H1U)
```

Because the Fermi-Hubbard Hamiltonian commutes with the total number operator, we can investigate each particle number sector separately,

```python
# Construct a specific chargestate with eight electrons:
ns4 = m.numbersector(4)
```

Assuming that our Fermi-Hubbard model also commutes with the spin component along e.g. the z-direction, we can investigate each of the z-projections separately,

```python
# Consider the sector with net zero spin in the z direction, meaning that
# 4 electrons = 2 spin-up electrons + 2 spin-down electrons

s0 = ns4.szsector(0)
```

Our basis is managed in a clever way referred to informally as Lin tables. You do not need to consider this here, but let us look at the many-body basis used for spin-up electrons:

```python
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
```

In our simple case the spin-down many-body basis looks exactly the same. We are now ready to look at the many-body Hamiltonian:

```python
# Compute the Hamiltonian
(HTu, HTd, HUd) = s0.hamiltonian

# Print the hopping Hamiltonian for the spin-up electronic systems
print('Spin-up hopping:')
print(HTu)
# Spin-up hopping:
#   (0, 1)    -1.0
#   (0, 4)    1.0
#   (1, 0)    -1.0
#   (1, 2)    -1.0
#   (1, 3)    -1.0
#   (1, 5)    1.0
#   (2, 1)    -1.0
#   (2, 4)    -1.0
#   (3, 1)    -1.0
#   (3, 4)    -1.0
#   (4, 0)    1.0
#   (4, 2)    -1.0
#   (4, 3)    -1.0
#   (4, 5)    -1.0
#   (5, 1)    1.0
#   (5, 4)    -1.0
```

We can explicitly construct the many-body Hamiltonian from these different components,

```python
import scipy.sparse as sparse

# The Total hamiltonian can be generated from these parts.
Nu, Nd = lita.Ns

# The interaction part in sparse format
HU = sparse.coo_matrix((HUd, (np.arange(Nu*Nd), np.arange(Nu*Nd))), shape=(Nu*Nd, Nu*Nd)).tocsr()

# The kronecker product of the two hopping sectors.
HT = sparse.kron(np.eye(Nd), HTu, 'dok') + sparse.kron(HTd, np.eye(Nu), 'dok')

print('The total Hamiltonian:')
print(HU + HT)
# The total Hamiltonian:
#   (0, 0)    2.0
#   (0, 1)    -1.0
#   (0, 4)    1.0
#   (0, 6)    -1.0
#   (0, 24)   1.0
#   (1, 0)    -1.0
#   (1, 2)    -1.0
#   (1, 3)    -1.0
#   (1, 5)    1.0
#   (1, 7)    -1.0
#   : :
#   (33, 34)  -1.0
#   (34, 10)  1.0
#   (34, 28)  -1.0
#   (34, 30)  1.0
#   (34, 32)  -1.0
#   (34, 33)  -1.0
#   (34, 35)  -1.0
#   (35, 11)  1.0
#   (35, 29)  -1.0
#   (35, 31)  1.0
#   (35, 34)  -1.0
#   (35, 35)  2.0
```

Also check out `example.py`. Where you can run and modify the above example code. The library code can easily be extended for your own pleasure. Remember though that it is released under a MIT license!

