constant int bc_vel_left[]    = {0, 1, 3, 4, 7, 8, 2, 5, 6};
constant int bc_vel_right[] = {0, 1, 3, 2, 6, 5, 4, 8, 7};
constant int lattice_q = 9;
constant double lattice_w[] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
constant double lattice_cx[] = {0, 1, 0, -1,  0, 1, -1, -1, 1};
constant double lattice_cy[] = {0, 0, 1,  0, -1, 1,  1, -1,-1};
constant double invcs2 = 3;
constant double nu = 0.05;

int xyq(int xy, int q){
    return xy * lattice_q + q;
}
kernel void stream(global double* N_in, global double* N_out, global int *P){
    int i = get_global_id(0);
    N_out[i] = N_in[P[i]];
}

kernel void velocity_bc_left(global double* N_out, global int* idx, 
double vel_n, double vel_t){
    int i = get_global_id(0);
    int xy = idx[i];
    
    int indexes[9];

    for (int q = 0; q<9; q++){
        indexes[q] = xyq(xy, bc_vel_left[q]);
    }
    // Read from all required indices using 2D indexing
    double n0 = N_out[indexes[0]];
    double n1 = N_out[indexes[1]];
    double n2 = N_out[indexes[2]];
    double n3 = N_out[indexes[3]];
    double n4 = N_out[indexes[4]];
    double n5 = N_out[indexes[5]];
    
    // mem_fence(CLK_GLOBAL_MEM_FENCE);
    double divisor = (1.0 - vel_n);
    if (fabs(divisor) < 1e-10) divisor = 1e-10;
    double rho = (n0 + n1 + n2 + 2.0 * (n3 + n4 + n5)) / divisor;
    
    // Write to output indices
    N_out[indexes[6]] = n3 + 2.0 / 3.0 * rho * vel_n;
    N_out[indexes[7]] = n4 - 0.5 * (n1 - n2) + (1.0 / 6.0) * rho * (vel_n + vel_t);
    N_out[indexes[8]] = n5 + 0.5 * (n1 - n2) + (1.0 / 6.0) * rho * (vel_n - vel_t);
}

kernel void velocity_bc_right(global double* N_out, global int* idx, 
double vel_n, double vel_t){
    int i = get_global_id(0);
    int xy = idx[i];
    
    int indexes[9];

    for (int q = 0; q<9; q++){
        indexes[q] = xyq(xy, bc_vel_right[q]);
    }
    // Read from all required indices using 2D indexing
    double n0 = N_out[indexes[0]];
    double n1 = N_out[indexes[1]];
    double n2 = N_out[indexes[2]];
    double n3 = N_out[indexes[3]];
    double n4 = N_out[indexes[4]];
    double n5 = N_out[indexes[5]];
    
    // mem_fence(CLK_GLOBAL_MEM_FENCE);
    double divisor = (1.0 - vel_n);
    if (fabs(divisor) < 1e-10) divisor = 1e-10;
    double rho = (n0 + n1 + n2 + 2.0 * (n3 + n4 + n5)) / divisor;
    
    // Write to output indices
    N_out[indexes[6]] = n3 + 2.0 / 3.0 * rho * vel_n;
    N_out[indexes[7]] = n4 - 0.5 * (n1 - n2) + (1.0 / 6.0) * rho * (vel_n + vel_t);
    N_out[indexes[8]] = n5 + 0.5 * (n1 - n2) + (1.0 / 6.0) * rho * (vel_n - vel_t);
}

kernel void collide(global double* N_in, global double* N_out, global double* tau, global int* is_wall){
    int xy = get_global_id(0);

    if (is_wall[xy]==1){
        int i;
        for (int q=0; q<lattice_q; q++){
            i = xyq(xy,q);
            N_out[i]=N_in[i];
        }
        return;
    }
    // flow_properties
    double rho = 0, u = 0, v = 0;
    double Nq;
    for (int q=0; q<lattice_q; q++){
        Nq = N_in[xyq(xy,q)];
        rho += Nq;
        u += Nq * lattice_cx[q];
        v += Nq * lattice_cy[q];
    }
    rho = rho < 1e-6 ? 1e-6 : rho;
    u/=rho;
    v/=rho;
    double u2 = u*u + v*v;

    // equilibrium
    double Neq;
    int i;
    for (int q=0; q<lattice_q; q++){
        i = xyq(xy, q); 
        double cu = (u * lattice_cx[q] + v * lattice_cy[q]);
        Neq = rho * lattice_w[q]*(1.0 + invcs2*cu + 0.5*invcs2*invcs2*cu*cu - 0.5*invcs2*u2);
        N_out[i] = N_in[i] - (N_in[i] - Neq)/tau[i];
    }
}


