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

# inverse of the square of the speed of sound in the lattice
LATTICE_INVCS2 = 3.

# dimensions of the simulation grid
SIZE_X = 3
SIZE_Y = 4

NU = 0.01
TAU = NU*LATTICE_INVCS2 + 1./2.
points = np.zeros([SIZE_X, SIZE_Y, LATTICE_Q, LATTICE_D])


def equilibrium_distribution(N: np.array) -> np.array:
    rho, u, v = flow_properties(N)
    return equilibrium_from_moments(rho, u, v)


def equilibrium_from_moments(rho: np.array, u: np.array, v: np.array) -> np.array:
    def p(a,b):
        return np.tensordot(a,b, axes=0)
    
    vc = LATTICE_INVCS2 * (  p(u, LATTICE_CX) 
                           + p(v, LATTICE_CY))
    Neq = p(rho, LATTICE_W) * (  vc 
                               + vc*vc/2
                               - p(u*u + v*v, np.ones(LATTICE_Q)) * LATTICE_INVCS2/2.
                               + 1)
    return Neq


def flow_properties(N: np.array) -> tuple[np.array, np.array, np.array]:
    rho = np.sum(N, axis=2)
    u = np.sum(N * LATTICE_CX, axis=2) / rho
    v = np.sum(N * LATTICE_CY, axis=2) / rho
    return rho, u, v

def collide(N):
    N = N - (N - equilibrium_distribution(N))/TAU

    return N


def main():
    N = np.ones((SIZE_X, SIZE_Y, LATTICE_Q))/LATTICE_Q
    N_eq = equilibrium_distribution(N)

    print(N_eq)
    print("equilibrium collision ==============", collide(N_eq))

if __name__ == "__main__":
    main()
