import numpy as np
import pyopencl as cl
import os
from evtk import hl as vtkhl
import imageio.v2 as imageio

#=========================================

# 6   2   5
#  \  |  /
# 3---0---1  
#  /  |  \
# 7   4   8

#=========================================

LATTICE_D = 2
LATTICE_Q = 9

# 9 directions of D2Q9 lattice
LATTICE_C = np.zeros([LATTICE_Q, LATTICE_D], dtype=np.int32)
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

# bounce-back
LATTICE_BB = np.ones(LATTICE_Q, dtype=np.int32)
LATTICE_BB[0] = 0 
LATTICE_BB[1] = 3
LATTICE_BB[2] = 4 
LATTICE_BB[3] = 1 
LATTICE_BB[4] = 2 
LATTICE_BB[5] = 7 
LATTICE_BB[6] = 8 
LATTICE_BB[7] = 5
LATTICE_BB[8] = 6 

# inverse of the square of the speed of sound in the lattice
LATTICE_INVCS2 = 3.


NU = 0.05




cpt = iter(range(1000000)) #image counter

def get_walls_from_image(path):
    img = imageio.imread(path)
    size_x, size_y = img.shape[:2]
    walls = np.argwhere(np.sum(img, axis=2)<20)
    # red pixels are the top boundary condition
    bc_top = np.argwhere((img[:,:,0]>200) & (img[:,:,1]<20) & (img[:,:,2]<20))
    # blue pixels are the bottom boundary condition
    bc_bottom = np.argwhere((img[:,:,0]<20) & (img[:,:,1]<20) & (img[:,:,2]>200))
    return walls, size_x, size_y, bc_top, bc_bottom

# dimensions of the simulation grid and simulated walls 
WALLS, SIZE_X, SIZE_Y, bc_top, bc_bottom = get_walls_from_image("assets/simu_r.png")
points = np.zeros([SIZE_X, SIZE_Y, LATTICE_Q, LATTICE_D])

print(f"number of high pressure points: {len(bc_top)}")
print(f"number of low pressure points: {len(bc_bottom)}")

def idx_noq(i,j):
    return (j + SIZE_Y * i)




def save_to_vtk(rho, u, v, name):
    if not os.path.exists("images"):
        os.makedirs("images")


    u = np.reshape(u , (SIZE_X, SIZE_Y, 1), order='C')
    v = np.reshape(v , (SIZE_X, SIZE_Y, 1), order='C')
    rho = np.reshape(rho, (SIZE_X, SIZE_Y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
    pointData={"p": rho - 1., "u": u, "v": v}) 


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


def collide(N, tau):
    Nm = N - (N - equilibrium_distribution(N))/tau
    return Nm


def idx(i,j,q):
    return q + LATTICE_Q * (j + SIZE_Y * i)

def calc_permutation():
    P = np.zeros(SIZE_X * SIZE_Y * LATTICE_Q, dtype=np.int32)
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_CX[q], SIZE_X)
                y = np.mod(j + LATTICE_CY[q], SIZE_Y)
                P[idx(x,y,q)] = idx(i,j,q)
    return P


def stream(Nm, P):
    N = np.reshape(
            np.reshape(Nm, (SIZE_X * SIZE_Y * LATTICE_Q))[P],
            (SIZE_X, SIZE_Y, LATTICE_Q)
        )
    return N


def wall(N, i, j):
    # [i,j] is a wall node
    for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_CX[q], SIZE_X)
                y = np.mod(j + LATTICE_CY[q], SIZE_Y)
                N[x, y, q] = N[i, j, LATTICE_BB[q]]
    return N


def wall_permutation(Pm, walls):
    w_p = np.copy(Pm)
    for (i,j) in walls:
         for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_CX[q], SIZE_X)
                y = np.mod(j + LATTICE_CY[q], SIZE_Y)
                w_p[idx(x, y, q)], w_p[idx(i, j, LATTICE_BB[q])] = Pm[idx(i, j, LATTICE_BB[q])], Pm[idx(x, y, q)] 
    return w_p

#=========================================

# 6   2   5
#  \  |  /
# 3---0---1  
#  /  |  \
# 7   4   8

#=========================================

def pressure_bc_top(N, idx, rho):
    N2D = np.reshape(N, (SIZE_X * SIZE_Y, LATTICE_Q))

    rho_ux = rho - (N2D[idx, 0] + N2D[idx, 2] + N2D[idx, 4]) - 2.*(N2D[idx, 3] + N2D[idx, 6] + N2D[idx, 7])

    N2D[idx, 1] = N2D[idx, 3] + (2./3.)*rho_ux
    N2D[idx, 5] = N2D[idx, 7] - (1./2.)*(N2D[idx, 2]-N2D[idx, 4]) + (1./6.)*rho_ux
    N2D[idx, 8] = N2D[idx, 6] + (1./2.)*(N2D[idx, 2]-N2D[idx, 4]) + (1./6.)*rho_ux
    
    return np.reshape(N2D, (SIZE_X, SIZE_Y, LATTICE_Q))


def pressure_bc_bottom(N, idx, rho):
    N2D = np.reshape(N, (SIZE_X * SIZE_Y, LATTICE_Q))
    rho_ux = rho - (N2D[idx, 0] + N2D[idx, 2] +  N2D[idx, 4]) - 2 * (N2D[idx, 1] + N2D[idx, 5] + N2D[idx, 8])

    N2D[idx, 3] = N2D[idx, 1] + (2./3.)*rho_ux
    N2D[idx, 6] = N2D[idx, 8] - (1./2.)*(N2D[idx, 2]-N2D[idx, 4]) + (1./6.)*rho_ux
    N2D[idx, 7] = N2D[idx, 5] + (1./2.)*(N2D[idx, 2]-N2D[idx, 4]) + (1./6.)*rho_ux
    
    return np.reshape(N2D, (SIZE_X, SIZE_Y, LATTICE_Q))

BC_VEL_TOP    = [0, 1, 3, 4, 7, 8, 2, 5, 6]
BC_VEL_BOTTOM = [0, 1, 3, 2, 6, 5, 4, 8, 7]

def velocity_bc(N, idx, bc_vel, un, ut):
    N2D = np.reshape(N, (SIZE_X * SIZE_Y, LATTICE_Q))
    rho = (N2D[idx, bc_vel[0]] + N2D[idx, bc_vel[1]] + N2D[idx, bc_vel[2]] + 2 * (N2D[idx, bc_vel[3]] + N2D[idx, bc_vel[4]] + N2D[idx, bc_vel[5]]))/(1. - un)
    N2D[idx, bc_vel[6]] = N2D[idx, bc_vel[3]] + 2./3. * rho * un
    N2D[idx, bc_vel[7]] = N2D[idx, bc_vel[4]] - 0.5 * (N2D[idx, bc_vel[1]] - N2D[idx, bc_vel[2]]) + 1./6. * rho * (un + ut)
    N2D[idx, bc_vel[8]] = N2D[idx, bc_vel[5]] + 0.5 * (N2D[idx, bc_vel[1]] - N2D[idx, bc_vel[2]]) + 1./6. * rho * (un - ut)
    return np.reshape(N2D, (SIZE_X, SIZE_Y, LATTICE_Q))

def bounce_back(Nm, w_p):
    N = np.reshape(
            np.reshape(Nm, (SIZE_X * SIZE_Y * LATTICE_Q))[w_p],
            (SIZE_X, SIZE_Y, LATTICE_Q)
        )
    return N

def build_cl_obj(source_file):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    try:
        with open(source_file) as f:
            prg = cl.Program(ctx, f.read()).build()
    except Exception as e:
        print("Error building OpenCL program:")
        print(e)
        raise

    return ctx, queue, prg

def build_cl_buf(ctx, N, P, idx_bc_top, idx_bc_bottom):
    mf = cl.mem_flags
    N_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=N)
    P_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
    idx_bc_top_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx_bc_top)
    idx_bc_bottom_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx_bc_bottom)
    M_g = cl.Buffer(ctx, mf.READ_WRITE, N.nbytes)

    return N_g, M_g, P_g, idx_bc_top_g, idx_bc_bottom_g

def get_velocity(t):
    vel = min(t / 5000., 1.) * 0.05
    velx = 0
    if (t > 5000 and t < 7000):
        velx = 0.05 * np.sin((t - 5000.)/2000. * np.pi)
    return np.float64(vel), np.float64(velx)

def get_indexes_from_image(path):
    img = imageio.imread(path)
    size_x, size_y = img.shape[:2]
    walls = np.argwhere(np.sum(img, axis=2)<20)
    # red pixels are the top boundary condition
    bc_top = np.argwhere((img[:,:,0]>200) & (img[:,:,1]<20) & (img[:,:,2]<20))
    # blue pixels are the bottom boundary condition
    bc_bottom = np.argwhere((img[:,:,0]<20) & (img[:,:,1]<20) & (img[:,:,2]>200))
    return walls, size_x, size_y, bc_top, bc_bottom

def initialize_simulation():
    walls, size_x, size_y, bc_top, bc_bottom = get_indexes_from_image("assets/simu_r.png")
    i_bc_top = bc_top[:, 0]
    j_bc_top = bc_top[:, 1]
    idx_bc_top = idx_noq(i_bc_top, j_bc_top)

    i_bc_bottom = bc_bottom[:, 0]
    j_bc_bottom = bc_bottom[:, 1]
    idx_bc_bottom = idx_noq(i_bc_bottom, j_bc_bottom)
    
    rho = np.ones((SIZE_X, SIZE_Y))
    u = 0. * np.ones((SIZE_X, SIZE_Y))
    v = 0. * np.ones((SIZE_X, SIZE_Y))
    N = equilibrium_from_moments(rho, u, v)
    P = calc_permutation()
    w_p = wall_permutation(P, WALLS)
    tau = (NU * LATTICE_INVCS2 + 0.5) * np.ones((SIZE_X, SIZE_Y, LATTICE_Q))
    tau[:, SIZE_Y - 20:, :] = (0.1 * LATTICE_INVCS2 + 0.5)
    tau[:, 0:5, :] = (0.1 * LATTICE_INVCS2 + 0.5)
    
    return N, w_p, idx_bc_top, idx_bc_bottom, tau    

def main():

    N, w_p, idx_bc_top, idx_bc_bottom, tau = initialize_simulation()

    # --------- CL initialization ----------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(script_dir, "flute.cl")
    ctx, queue, prg = build_cl_obj(source) 
    
    N_g, M_g, P_g, idx_red_g, idx_blue_g = build_cl_buf(ctx, N, w_p, idx_bc_top, idx_bc_bottom)
    k_stream = prg.stream
    k_velocity_bc_top = prg.velocity_bc_top
    k_velocity_bc_bottom = prg.velocity_bc_bottom
    k_collide = prg.collide
    M = np.zeros_like(N)
    

    # --------- Simulation loop ----------
    for t in range(40001):

        vel, velx = get_velocity(t)

  
        # Stream
        k_stream(queue, (SIZE_X * SIZE_Y * LATTICE_Q,), None, N_g, M_g, P_g)
        k_velocity_bc_top(queue, (len(idx_bc_top),), None, M_g, idx_red_g, np.float64(vel), np.float64(velx))
        k_velocity_bc_bottom(queue, (len(idx_bc_bottom),), None, M_g, idx_blue_g, np.float64(-vel), np.float64(0))
        
        # Swap buffers for next iteration
        

        # check = np.allclose(M, N_CPU)
        # print(check, N_CPU.shape == N.shape)

        # print(f"max difference: {np.max(np.abs(M - N_CPU))}/{np.mean(np.abs(N_CPU))}")

        # N_CPU = velocity_bc(M, idx_bc_bottom, BC_VEL_BOTTOM, -vel, 0.)
        # N_CPU = collide(N_CPU, tau)
        # N = N_CPU

        # N = velocity_bc(M, idx_bc_bottom, BC_VEL_BOTTOM, -vel, 0.)
        # N_CPU = np.copy(N)
        # N_CPU = collide(N_CPU, tau)
        k_collide(queue, (SIZE_X * SIZE_Y,), None, N_g, M_g, P_g)
        cl.enqueue_copy(queue, N, M_g)
        queue.finish()
        N_g, M_g = M_g, N_g

        # --------- Save results ----------
        if t % 200 == 0:
            rho, u, v = flow_properties(N)
            save_to_vtk(rho, u, v, "sim")
            print(f"step: {t}")
        
        # Check for numerical instability (NaN values in N)
        if np.isnan(N).any():
            print(f"Instability detected at step {t}!")
            break
    
    print(f"Simulation terminated at step {t}")

if __name__ == "__main__":
    main()
