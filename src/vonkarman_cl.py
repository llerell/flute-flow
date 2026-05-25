import numpy as np
import imageio.v2 as imageio
from evtk import hl as vtkhl
import pyopencl as cl
import os
import get_sound
from pathlib import Path

os.environ['PYOPENCL_CTX'] = '0'

LATTICE_D = 2
LATTICE_Q = 9

LATTICE_C= np.zeros([LATTICE_Q, LATTICE_D], dtype = np.int32)
LATTICE_C[0, :] = [ 0,  0]
LATTICE_C[1, :] = [ 1,  0]
LATTICE_C[2, :] = [ 0,  1]
LATTICE_C[3, :] = [-1,  0]
LATTICE_C[4, :] = [ 0, -1]
LATTICE_C[5, :] = [ 1,  1]
LATTICE_C[6, :] = [-1,  1]
LATTICE_C[7, :] = [-1, -1]
LATTICE_C[8, :] = [ 1, -1]

LATTICE_Cx = LATTICE_C[:, 0]
LATTICE_Cy = LATTICE_C[:, 1]

LATTICE_BB = np.ones(LATTICE_Q, dtype = np.int32)
LATTICE_BB[0] = 0
LATTICE_BB[1] = 3
LATTICE_BB[2] = 4
LATTICE_BB[3] = 1
LATTICE_BB[4] = 2
LATTICE_BB[5] = 7
LATTICE_BB[6] = 8
LATTICE_BB[7] = 5
LATTICE_BB[8] = 6

lattice_w = np.zeros(LATTICE_Q)
lattice_w[0] = 4./9.
lattice_w[1:5] = 1./9.
lattice_w[5:9] = 1./36.

LATTICE_INVCS2 = 3.

NU = 0.005
TAU = NU * LATTICE_INVCS2 + 0.5

cpt = iter(range(1000000))

def save_to_vtk(name, rho, u, v, size_x, size_y):
    if not Path("images").exists():
        Path("images").mkdir(parents=True, exist_ok=True)
        
    u   = np.reshape(u  , (size_x, size_y, 1), order='C')
    v   = np.reshape(v  , (size_x, size_y, 1), order='C')
    rho = np.reshape(rho, (size_x, size_y, 1), order='C')
    output_path = Path("images") / f"{name}_{next(cpt)}"
    vtkhl.imageToVTK(str(output_path),
            pointData={"p": rho - 1., "u": u, "v": v})


def open_image(filename):
    image = imageio.imread(filename)
    size_x = image.shape[0]
    size_y = image.shape[1]
    print(image.shape)
    walls   = [(i, j) for i in range(size_x) for j in range(size_y) if np.sum(image[i, j, 0:3]) < 20]
    walls   = np.array(walls)
    ij_red  = [(i, j) for i in range(size_x) for j in range(size_y) if (np.sum(image[i, j, 0:3]) < 275 and image[i, j, 0] > 200)]
    ij_red  = np.array(ij_red)
    ij_blue = [(i, j) for i in range(size_x) for j in range(size_y) if (np.sum(image[i, j, 0:3]) < 275 and image[i, j, 2] > 200)]
    ij_blue = np.array(ij_blue)
    ij_green = [(i, j) for i in range(size_x) for j in range(size_y) if (np.sum(image[i, j, 0:3]) < 275 and image[i, j, 1] > 200)]
    ij_green = np.array(ij_green)
    assert len(ij_green) == 1

    green = ij_green[0]
    return size_x, size_y, walls, ij_red, ij_blue, green


def equilibrium_from_moments(rho, u, v):
    def p(a, b):
        return np.tensordot(a, b, axes = 0)

    vc = LATTICE_INVCS2 * (p(u, LATTICE_Cx) + p(v, LATTICE_Cy))
    Neq = p(rho, lattice_w) * (vc + vc * vc / 2. - p(u * u + v * v, np.ones(LATTICE_Q)) * LATTICE_INVCS2 / 2. + 1.)
    return Neq

def flow_properties(N):
    rho = np.sum(N, axis = 2)
    rho = np.maximum(rho, 1e-8)
    u = np.sum(N * LATTICE_Cx, axis = 2) / rho
    v = np.sum(N * LATTICE_Cy, axis = 2) / rho
    return rho, u, v

def equilibrium_distribution(N):
    rho, u, v = flow_properties(N)
    Neq = equilibrium_from_moments(rho, u, v)
    return Neq

def collide(N):
    Nm = N - (N - equilibrium_distribution(N)) / TAU
    return Nm

def idxnoq(i, j, size_y):
    return j + size_y * i

def idx(i, j, q, size_y):
    return q + LATTICE_Q * idxnoq(i, j, size_y)

def stream_permutation(size_x, size_y, LATTICE_Q):
    P = np.zeros(size_x * size_y * LATTICE_Q, dtype = np.int32)
    for i in range(size_x):
        for j in range(size_y):
            for q in range(LATTICE_Q):
                x = np.mod(i + LATTICE_Cx[q], size_x)
                y = np.mod(j + LATTICE_Cy[q], size_y)
                P[idx(x, y, q, size_y)] = idx(i, j, q, size_y)
    return P

def stream(Nm, P, size_x, size_y):
    N = np.reshape(np.reshape(Nm, (size_x * size_y * LATTICE_Q))[P], (size_x, size_y, LATTICE_Q))
    return N

def wall_permutation(Pm, i, j, size_x, size_y):
    P = np.copy(Pm)
    for q in range(LATTICE_Q):
        x = np.mod(i + LATTICE_Cx[q], size_x)
        y = np.mod(j + LATTICE_Cy[q], size_y)

        P[idx(x, y, q, size_y)], P[idx(i, j, LATTICE_BB[q], size_y)] = P[idx(i, j, LATTICE_BB[q] ,size_y)], P[idx(x, y, q, size_y)]
    return P

"""
#             0  1  2  3  4  5  6  7  8
bc_p_left  = [0, 2, 4, 3, 7, 6, 1, 5, 8]
bc_p_right = [0, 2, 4, 1, 8, 5, 3, 6, 7]

def pressure_bc(N, idx, bc_p, rho, size_x, size_y):
    N2D = np.reshape(N, (size_x * size_y, LATTICE_Q))
    rho_un = rho - (N2D[idx, bc_p[0]] + N2D[idx, bc_p[1]] + N2D[idx, bc_p[2]] + 2 * (N2D[idx, bc_p[3]] + N2D[idx, bc_p[4]] + N2D[idx, bc_p[5]]))
    N2D[idx, bc_p[6]] = N2D[idx, bc_p[3]] + 2./3. * rho_un
    N2D[idx, bc_p[7]] = N2D[idx, bc_p[4]] - 0.5 * (N2D[idx, bc_p[1]] - N2D[idx, bc_p[2]]) + 1./6. * rho_un
    N2D[idx, bc_p[8]] = N2D[idx, bc_p[5]] + 0.5 * (N2D[idx, bc_p[1]] - N2D[idx, bc_p[2]]) + 1./6. * rho_un
    return np.reshape(N2D, (size_x, size_y, LATTICE_Q))

#                0  1  2  3  4  5  6  7  8
bc_vel_top    = [0, 1, 3, 4, 7, 8, 2, 5, 6]
bc_vel_bottom = [0, 1, 3, 2, 6, 5, 4, 8, 7]

def velocity_bc(N, idx, bc_vel, un, ut):
    
    N2D = np.reshape(N, (size_x * size_y, LATTICE_Q))

    
    rho = (N2D[idx, bc_vel[0]] + N2D[idx, bc_vel[1]] + N2D[idx, bc_vel[2]] + 2 * (N2D[idx, bc_vel[3]] + N2D[idx, bc_vel[4]] + N2D[idx, bc_vel[5]]))/(1. - un)
    N2D[idx, bc_vel[6]] = N2D[idx, bc_vel[3]] + 2./3. * rho * un
    N2D[idx, bc_vel[7]] = N2D[idx, bc_vel[4]] - 0.5 * (N2D[idx, bc_vel[1]] - N2D[idx, bc_vel[2]]) + 1./6. * rho * (un + ut)
    N2D[idx, bc_vel[8]] = N2D[idx, bc_vel[5]] + 0.5 * (N2D[idx, bc_vel[1]] - N2D[idx, bc_vel[2]]) + 1./6. * rho * (un - ut)
    return np.reshape(N2D, (size_x, size_y, LATTICE_Q))
"""

def build_cl_obj(source_file):
    ctx = cl.create_some_context(0)
    queue = cl.CommandQueue(ctx)
    with open(source_file) as f:
        source = f.read()
        prg = cl.Program(ctx, source)
        prg.build()
    return ctx, queue, prg

def build_cl_buf(ctx, N, P, idx_red, idx_blue, tau_arr, rho_out):
    mf = cl.mem_flags
    N_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=N)
    P_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
    idx_red_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx_red)
    idx_blue_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx_blue)
    M_g = cl.Buffer(ctx, mf.READ_WRITE, N.nbytes)
    tau_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau_arr)
    rho_out_g = cl.Buffer(ctx, mf.WRITE_ONLY, rho_out.nbytes)
    return N_g, M_g, P_g, idx_red_g, idx_blue_g, tau_g, rho_out_g

def get_velocity(t):
    vel = min(t / 2000., 1) * 0.005
    velx = 0
    return np.float64(vel), np.float64(velx)

def main():
    N_iter = 10_000_000
    sound_record_freq = 66

    size_x, size_y, walls, ij_red, ij_blue, green = open_image(Path("assets") / "simu_r2.png")
    iwalls = walls[:, 0]
    jwalls = walls[:, 1]
    rho_out = np.zeros(N_iter//sound_record_freq, dtype=np.float64)

    idx_red  = idxnoq(ij_red [:, 0], ij_red [:, 1], size_y).astype(np.int32)
    idx_blue = idxnoq(ij_blue[:, 0], ij_blue[:, 1], size_y).astype(np.int32)
    idx_green = idxnoq(green[0], green[1], size_y).astype(np.int32)
    rho = np.ones((size_x, size_y))
    u = 0.0 * np.ones((size_x, size_y))
    v = 0.0 * np.ones((size_x, size_y))

    N = equilibrium_from_moments(rho, u, v)

    P = stream_permutation(size_x, size_y, LATTICE_Q)
    P = wall_permutation(P, iwalls, jwalls, size_x, size_y)

    tau_arr = (NU * LATTICE_INVCS2 + 0.5) * np.ones((size_x, size_y, LATTICE_Q), dtype=np.float64)
    for i, j in ij_blue:
        for i1 in range(-10, 11):
            for j1 in range(-10, 11):
                tau_arr[i + i1, j + j1, :] = (0.01 * LATTICE_INVCS2 + 0.5)


    N = N + np.random.rand(*N.shape)*0.001 # pour briser la symétrie et éviter les artefacts

    source = Path("src") / "flute.cl"
    ctx, queue, prg = build_cl_obj(source)
    N_g, M_g, P_g, idx_red_g, idx_blue_g, tau_g, rho_out_g = build_cl_buf(ctx, N, P, idx_red, idx_blue, tau_arr, rho_out)
    k_stream = prg.stream
    k_velocity_bc_top = prg.velocity_bc_top
    k_velocity_bc_bottom = prg.velocity_bc_bottom
    k_collide = prg.collide
    k_save_rho = prg.save_rho

    for t in range(N_iter + 1):
        vel, velx = get_velocity(t)

        k_stream(queue, (size_x*size_y*LATTICE_Q,), None, M_g, N_g, P_g)
        k_velocity_bc_top(queue, (len(idx_blue),), None, M_g, idx_blue_g, vel, velx)
        k_velocity_bc_bottom(queue, (len(idx_red),), None, M_g, idx_red_g, - vel, np.float64(0.))
        k_collide(queue, (size_x*size_y,), None, M_g, tau_g)
        N_g, M_g = M_g, N_g

        
        if (t % 400 == 0):
            queue.finish()
            cl.enqueue_copy(queue, N, N_g)
            rho, u, v = flow_properties(N)
            if np.any(np.isnan(rho)):
                print("NaN detected, stopping simulation.")
                print("NaN detected at index (ij):", np.argwhere(np.isnan(rho)))
                break
            save_to_vtk("test", rho, u, v, size_x, size_y)
            print(f"step {t}")
        
        if (t % sound_record_freq == 0):
            k_save_rho(queue, (1,), None, N_g, idx_green, np.int32(t//sound_record_freq), rho_out_g)
        

    queue.finish()
    cl.enqueue_copy(queue, rho_out, rho_out_g)
    output_file = Path("sounds") / "output.wav"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    get_sound.density_to_wav_file(rho_out, output_file)

if __name__ == "__main__":
    main()