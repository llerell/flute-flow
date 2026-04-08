constant int bc_vel_top[]    = {0, 1, 3, 4, 7, 8, 2, 5, 6};
constant int bc_vel_bottom[] = {0, 1, 3, 2, 6, 5, 4, 8, 7};
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


kernel void velocity_bc_top(global double* N_out, global int* idx, 
double vel_n, double vel_t){
    int i = get_global_id(0);
    int xy = idx[i];
    
    int indexes[9];

    for (int q = 0; q<9; q++){
        indexes[q] = xyq(xy, bc_vel_top[q]);
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

kernel void velocity_bc_bottom(global double* N_out, global int* idx, 
double vel_n, double vel_t){
    int i = get_global_id(0);
    int xy = idx[i];
    
    int indexes[9];

    for (int q = 0; q<9; q++){
        indexes[q] = xyq(xy, bc_vel_bottom[q]);
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

kernel void collide(global double* N_in, global double* N_out, global double* tau){
    int xy = get_global_id(0);

    // flow_properties
    double rho, u, v=0;
    double Nq;
    for (int q=0; q<lattice_q; q++){
        Nq = N_in[xyq(xy,q)];
        rho += Nq;
        u += Nq * lattice_cx[q];
        v += Nq * lattice_cy[q];
    }
    rho = rho < 0.000001 ? 0.000001 : rho;
    u/=rho;
    v/=rho;

    // equilibrium for moments
    
}

double Neq(){
    return 0.1;
}

