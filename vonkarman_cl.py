import numpy as np
import imageio.v2 as imageio
from evtk import hl as vtkhl
import pyopencl as cl
import os
os.environ['PYOPENCL_CTX'] = '0'

lattice_D = 2
lattice_Q = 9

lattice_c = np.zeros([lattice_Q, lattice_D], dtype = np.int32)
lattice_c[0, :] = [ 0,  0]
lattice_c[1, :] = [ 1,  0]
lattice_c[2, :] = [ 0,  1]
lattice_c[3, :] = [-1,  0]
lattice_c[4, :] = [ 0, -1]
lattice_c[5, :] = [ 1,  1]
lattice_c[6, :] = [-1,  1]
lattice_c[7, :] = [-1, -1]
lattice_c[8, :] = [ 1, -1]

lattice_cx = lattice_c[:, 0]
lattice_cy = lattice_c[:, 1]

lattice_bb = np.ones(lattice_Q, dtype = np.int32)
lattice_bb[0] = 0
lattice_bb[1] = 3
lattice_bb[2] = 4
lattice_bb[3] = 1
lattice_bb[4] = 2
lattice_bb[5] = 7
lattice_bb[6] = 8
lattice_bb[7] = 5
lattice_bb[8] = 6

lattice_w = np.zeros(lattice_Q)
lattice_w[0] = 4./9.
lattice_w[1:5] = 1./9.
lattice_w[5:9] = 1./36.

lattice_invcs2 = 3.

nu = 0.0001

cpt = iter(range(1000000))
def save_to_vtk(name, rho, u, v):
    u   = np.reshape(u  , (size_x, size_y, 1), order='C')
    v   = np.reshape(v  , (size_x, size_y, 1), order='C')
    rho = np.reshape(rho, (size_x, size_y, 1), order='C')
    vtkhl.imageToVTK(f"images/{name}_{next(cpt)}",
            pointData={"p": rho - 1., "u": u, "v": v})


def open_image(filename):
    image = imageio.imread(filename)
    SIZE_X = image.shape[0]
    SIZE_Y = image.shape[1]
    print(image.shape)
    walls   = [(i, j) for i in range(SIZE_X) for j in range(SIZE_Y) if np.sum(image[i, j, 0:3]) < 20]
    walls   = np.array(walls)
    ij_red  = [(i, j) for i in range(SIZE_X) for j in range(SIZE_Y) if (np.sum(image[i, j, 0:3]) < 275 and image[i, j, 0] > 200)]
    ij_red  = np.array(ij_red)
    ij_blue = [(i, j) for i in range(SIZE_X) for j in range(SIZE_Y) if (np.sum(image[i, j, 0:3]) < 275 and image[i, j, 2] > 200)]
    ij_blue = np.array(ij_blue)
    return SIZE_X, SIZE_Y, walls, ij_red, ij_blue


def equilibrium_from_moments(rho, u, v):
    def p(a, b):
        return np.tensordot(a, b, axes = 0)

    vc = lattice_invcs2 * (p(u, lattice_cx) + p(v, lattice_cy))
    Neq = p(rho, lattice_w) * (vc + vc * vc / 2. - p(u * u + v * v, np.ones(lattice_Q)) * lattice_invcs2 / 2. + 1.)
    return Neq

def flow_properties(N):
    rho = np.sum(N, axis = 2)
    u = np.sum(N * lattice_cx, axis = 2) / rho
    v = np.sum(N * lattice_cy, axis = 2) / rho
    return rho, u, v

def equilibrium_distribution(N):
    rho, u, v = flow_properties(N)
    index = 4 * size_x + 5
    #print("index = ", index)
    #print(f"N= {N[4,5]} rho={rho[4,5]}, u = {u[4,5]}, v = {v[4,5]}")
    Neq = equilibrium_from_moments(rho, u, v)
    return Neq

def collide(N):
    Nm = N - (N - equilibrium_distribution(N)) / tau
    return Nm

def idxnoq(i, j):
    return j + size_y * i

def idx(i, j, q):
    return q + lattice_Q * idxnoq(i, j)

def stream_permutation(size_x, size_y, lattice_Q):
    P = np.zeros(size_x * size_y * lattice_Q, dtype = np.int32)
    for i in range(size_x):
        for j in range(size_y):
            for q in range(lattice_Q):
                x = np.mod(i + lattice_cx[q], size_x)
                y = np.mod(j + lattice_cy[q], size_y)
                #N[x, y, q] = Nm[i, j, q]
                P[idx(x, y, q)] = idx(i, j, q)
    return P

def stream(Nm, P):
    N = np.reshape(np.reshape(Nm, (size_x * size_y * lattice_Q))[P], (size_x, size_y, lattice_Q))
    return N

def wall(N, i, j):
    # to be called post-streaming
    # [i, j] is a wall
    for q in range(lattice_Q):
        x = np.mod(i + lattice_cx[q], size_x)
        y = np.mod(j + lattice_cy[q], size_y)

        N[x, y, q], N[i, j, lattice_bb[q]] = N[i, j, lattice_bb[q]], N[x, y, q]
    return N

def wall_permutation(Pm, i, j):
    P = np.copy(Pm)
    for q in range(lattice_Q):
        x = np.mod(i + lattice_cx[q], size_x)
        y = np.mod(j + lattice_cy[q], size_y)

        P[idx(x, y, q)], P[idx(i, j, lattice_bb[q])] = P[idx(i, j, lattice_bb[q])], P[idx(x, y, q)]
    return P

#             0  1  2  3  4  5  6  7  8
bc_p_left  = [0, 2, 4, 3, 7, 6, 1, 5, 8]
bc_p_right = [0, 2, 4, 1, 8, 5, 3, 6, 7]

def pressure_bc(N, idx, bc_p, rho):
    N2D = np.reshape(N, (size_x * size_y, lattice_Q))
    rho_un = rho - (N2D[idx, bc_p[0]] + N2D[idx, bc_p[1]] + N2D[idx, bc_p[2]] + 2 * (N2D[idx, bc_p[3]] + N2D[idx, bc_p[4]] + N2D[idx, bc_p[5]]))
    N2D[idx, bc_p[6]] = N2D[idx, bc_p[3]] + 2./3. * rho_un
    N2D[idx, bc_p[7]] = N2D[idx, bc_p[4]] - 0.5 * (N2D[idx, bc_p[1]] - N2D[idx, bc_p[2]]) + 1./6. * rho_un
    N2D[idx, bc_p[8]] = N2D[idx, bc_p[5]] + 0.5 * (N2D[idx, bc_p[1]] - N2D[idx, bc_p[2]]) + 1./6. * rho_un
    return np.reshape(N2D, (size_x, size_y, lattice_Q))

#                0  1  2  3  4  5  6  7  8
bc_vel_top    = [0, 1, 3, 4, 7, 8, 2, 5, 6]
bc_vel_bottom = [0, 1, 3, 2, 6, 5, 4, 8, 7]

def velocity_bc(N, idx, bc_vel, un, ut):
    
    N2D = np.reshape(N, (size_x * size_y, lattice_Q))
    #print( ">", N2D[idx[12], bc_vel[0]] )
    N1 = np.reshape(N, size_x * size_y * lattice_Q)
    #print( ">>", N1[ idx[12]*lattice_Q + 0])
    
    
    rho = (N2D[idx, bc_vel[0]] + N2D[idx, bc_vel[1]] + N2D[idx, bc_vel[2]] + 2 * (N2D[idx, bc_vel[3]] + N2D[idx, bc_vel[4]] + N2D[idx, bc_vel[5]]))/(1. - un)
    N2D[idx, bc_vel[6]] = N2D[idx, bc_vel[3]] + 2./3. * rho * un
    N2D[idx, bc_vel[7]] = N2D[idx, bc_vel[4]] - 0.5 * (N2D[idx, bc_vel[1]] - N2D[idx, bc_vel[2]]) + 1./6. * rho * (un + ut)
    N2D[idx, bc_vel[8]] = N2D[idx, bc_vel[5]] + 0.5 * (N2D[idx, bc_vel[1]] - N2D[idx, bc_vel[2]]) + 1./6. * rho * (un - ut)
    return np.reshape(N2D, (size_x, size_y, lattice_Q))

def build_cl_obj(source_file):
    ctx = cl.create_some_context(0)
    queue = cl.CommandQueue(ctx)
    with open(source_file) as f:
        source = f.read()
        prg = cl.Program(ctx, source)
        prg.build()
    return ctx, queue, prg

def build_cl_buf(ctx, N, P, idx_red, idx_blue, tau):
    mf = cl.mem_flags
    N_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=N)
    P_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
    idx_red_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx_red)
    idx_blue_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx_blue)
    M_g = cl.Buffer(ctx, mf.READ_WRITE, N.nbytes)
    tau_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau)
    return N_g, M_g, P_g, idx_red_g, idx_blue_g, tau_g

def get_velocity(t):
    vel = min(t / 2000., 1) * 0.0001
    velx = 0
    """if (t > 5000 and t < 7000):
        velx = 0.05 * np.sin((t - 5000.)/2000. * np.pi)"""
    return np.float64(vel), np.float64(velx)

if __name__ == "__main__":
    size_x, size_y, walls, ij_red, ij_blue = open_image("simu_r2.png")
    iwalls = walls[:, 0]
    jwalls = walls[:, 1]

    idx_red  = idxnoq(ij_red [:, 0], ij_red [:, 1]).astype(np.int32)
    idx_blue = idxnoq(ij_blue[:, 0], ij_blue[:, 1]).astype(np.int32)

    rho = np.ones((size_x, size_y))
    u = 0.0 * np.ones((size_x, size_y))
    v = 0.0 * np.ones((size_x, size_y))

    N = equilibrium_from_moments(rho, u, v)

    P = stream_permutation(size_x, size_y, lattice_Q)
    P = wall_permutation(P, iwalls, jwalls)

    tau = (nu * lattice_invcs2 + 0.5) * np.ones((size_x, size_y, lattice_Q))
    for i, j in ij_blue:
        for i1 in range(-10, 11):
            for j1 in range(-10, 11):
                tau[i + i1, j + j1, :] = (0.1 * lattice_invcs2 + 0.5)
    #tau[:, size_y - 20:, :] = (0.1 * lattice_invcs2 + 0.5)
    #tau[:, 0:5, :] = (0.1 * lattice_invcs2 + 0.5)

    #N = N + np.random.rand(*N.shape)*0.001 pour le test et pour éviter de diviser par 0

    source = "flute.cl"
    ctx, queue, prg = build_cl_obj(source)
    N_g, M_g, P_g, idx_red_g, idx_blue_g, tau_g = build_cl_buf(ctx, N, P, idx_red, idx_blue, tau)
    k_stream = prg.stream
    k_velocity_bc_top = prg.velocity_bc_top
    k_velocity_bc_bottom = prg.velocity_bc_bottom
    k_collide = prg.collide
    
    M = np.zeros_like(N)

    for t in range(50000001):
        vel, velx = get_velocity(t)

        k_stream(queue, (size_x*size_y*lattice_Q,), None, M_g, N_g, P_g)
        #N = stream(N, P)
        #print(np.allclose(N, M))

        k_velocity_bc_top(queue, (len(idx_blue),), None, M_g, idx_blue_g, vel, velx)
        #N = velocity_bc(N, idx_blue, bc_vel_top, vel, velx)
        #print(np.allclose(N, M))


        k_velocity_bc_bottom(queue, (len(idx_red),), None, M_g, idx_red_g, - vel, np.float64(0.))
        #N = velocity_bc(N, idx_red , bc_vel_bottom, - vel, np.float64(0.))
        #print(np.allclose(N, M))

        
        #N = collide(N)
        k_collide(queue, (size_x*size_y,), None, M_g, tau_g)
        #print(np.allclose(N, M))
        #print(np.max(np.abs(N - M)))
        N_g, M_g = M_g, N_g


        if (t % 400000 == 0):
            cl.enqueue_copy(queue, M, M_g)
            rho, u, v = flow_properties(M)
            save_to_vtk("test", rho, u, v)
            print(t)