# fonction qui à n associe la distribution à l'équilibre équivalente.
# on se place en D2Q9


# 6   2   5
#  \  |  /
# 3---0---1  
#  /  |  \
# 7   4   8


import numpy as np

LATTICE_D = 2
LATTICE_Q = 9

# 9 directions of D2Q9 lattice
LATTICE_C = np.zeros([LATTICE_Q, LATTICE_D])
LATTICE_C[0,:] = np.array([0,0])
LATTICE_C[1,:] = np.array([1,0])
LATTICE_C[2,:] = np.array([0,1])
LATTICE_C[3,:] = np.array([-1,0])
LATTICE_C[4,:] = np.array([0,-1])
LATTICE_C[5,:] = np.array([1,1])
LATTICE_C[6,:] = np.array([-1, 1])
LATTICE_C[7,:] = np.array([-1, -1])
LATTICE_C[8,:] = np.array([1, -1])

# weights of the D2Q9 lattice
LATTICE_W = np.zeros(LATTICE_Q)
LATTICE_W[0] = 4./9.
LATTICE_W[1:5] = 1./9.
LATTICE_W[5:] = 1./36.

# x and y components of the lattice velocities
LATTICE_CX = LATTICE_C[:,0]
LATTICE_CY = LATTICE_C[:,1]

LATTICE_INVCS2 = 3.


def equilibrium_distribution(N: np.array) -> np.array:
    rho,u,v = flow_properties(N)
    return equilibirum_from_moments(rho,u,v)


def equilibirum_from_moments(rho: float, u: np.array[float], v: np.array[float]) -> np.array[float]:
    vc = LATTICE_INVCS2 * (u*LATTICE_CX 
                           + v*LATTICE_CY)
    Neq = rho*LATTICE_W*(1 
                         + vc 
                         + (vc*vc)/2 
                         - LATTICE_INVCS2*(u*u + v*v)/2)
    return Neq


def flow_properties(N: np.array) -> tuple[float, np.array[float], np.array[float]]:
    rho = np.sum(N)
    u = np.sum(N * LATTICE_CX) / rho
    v = np.sum(N * LATTICE_CY) / rho
    return rho, u, v


rho, u, v = [1., 0.1, 0.]
N = equilibirum_from_moments(rho, u, v)
Neq = equilibrium_distribution(N)
print(flow_properties(N))
print(N)
print(N-Neq)