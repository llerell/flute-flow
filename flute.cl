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

kernel void velocity_bc_top(global double *N_out, global int *idx, 
    double vel_n, double vel_t){
    int i = get_global_id(0);
    int xy = idx[i];
    int B[lattice_q];
    for (int q = 0 ; q < lattice_q ; q = q + 1){
        B[q] = xyq(xy, bc_vel_top[q]);
    }
    double rho = N_out[B[0]] + N_out[B[1]] + N_out[B[2]] \
     + 2 * (N_out[B[3]] + N_out[B[4]] +N_out[B[5]] );
    rho = rho / (1 - vel_n);
    N_out[B[6]] = N_out[B[3]] + 2 * (rho * vel_n)/3;
    N_out[B[7]] = N_out[B[4]] - 0.5 * (N_out[B[1]] - N_out[B[2]]) + 1./6. * rho * (vel_n + vel_t) ;
    N_out[B[8]] = N_out[B[5]] + 0.5 * (N_out[B[1]] - N_out[B[2]]) + 1./6. * rho * (vel_n - vel_t) ;
}

kernel void velocity_bc_bottom(global double *N_out, global int *idx, 
  double vel_n, double vel_t){
    int i = get_global_id(0);
    int xy = idx[i];
    int B[lattice_q];
    for (int q = 0 ; q < lattice_q ; q = q + 1){
        B[q] = xyq(xy, bc_vel_bottom[q]);
    }
    double rho = N_out[B[0]] + N_out[B[1]] + N_out[B[2]] + 2 * (N_out[B[3]] + N_out[B[4]] +N_out[B[5]] );
    rho = rho / (1 - vel_n);
    N_out[B[6]] = N_out[B[3]] + 2 * (rho * vel_n)/3;
    N_out[B[7]] = N_out[B[4]] - 0.5 * (N_out[B[1]] - N_out[B[2]]) + 1./6. * rho * (vel_n + vel_t) ;
    N_out[B[8]] = N_out[B[5]] + 0.5 * (N_out[B[1]] - N_out[B[2]]) + 1./6. * rho * (vel_n - vel_t) ;
}
kernel void collide(global double* N_in, global double* N_out, global double* tau){
    int xy = get_global_id(0);


    // flow_properties
    double rho = 0, u = 0, v = 0;
    double Nq;
    for (int q=0; q<lattice_q; q++){
        Nq = N_in[xyq(xy,q)];
        rho += Nq;
        u += Nq * lattice_cx[q];
        v += Nq * lattice_cy[q];
    }
    
    rho = (fabs(rho) > 1e-10) ? rho : 1e-10;
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


kernel void compute_rho(global double* rho, int index, global double* N){
    double sum = 0;
    for (int q=0; q<lattice_q; q++){
        sum += N[xyq(index,q)];
    }
    rho[0] = sum;
}