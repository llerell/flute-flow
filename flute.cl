constant int bc_vel_top[] = {0, 1, 3, 4, 7, 8, 2, 5, 6};
constant int bc_vel_bottom[] = {0, 1, 3, 2, 6, 5, 4, 8, 7};
constant int lattice_q = 9;

constant int lattice_cx[] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constant int lattice_cy[] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constant double lattice_w[]= {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};

constant double icsq = 3.;

int xy_q_to_xyq(int xy, int q){
    return xy * lattice_q + q;
}

kernel void stream(global double *N_out, global double *N_in, global int *P){
    int i = get_global_id(0);
    N_out[i] = N_in[P[i]];
}

kernel void velocity_bc_top(global double *N_out, global int *idx, 
    double vel_n, double vel_t){
    int i = get_global_id(0);
    int xy = idx[i];
    int B[lattice_q];
    for (int q = 0 ; q < lattice_q ; q = q + 1){
        B[q] = xy_q_to_xyq(xy, bc_vel_top[q]);
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
        B[q] = xy_q_to_xyq(xy, bc_vel_bottom[q]);
    }
    double rho = N_out[B[0]] + N_out[B[1]] + N_out[B[2]] + 2 * (N_out[B[3]] + N_out[B[4]] +N_out[B[5]] );
    rho = rho / (1 - vel_n);
    N_out[B[6]] = N_out[B[3]] + 2 * (rho * vel_n)/3;
    N_out[B[7]] = N_out[B[4]] - 0.5 * (N_out[B[1]] - N_out[B[2]]) + 1./6. * rho * (vel_n + vel_t) ;
    N_out[B[8]] = N_out[B[5]] + 0.5 * (N_out[B[1]] - N_out[B[2]]) + 1./6. * rho * (vel_n - vel_t) ;
}

kernel void collide(global double *N, global double *tau){
    int P = get_global_id(0);//un point de l'espace
    double rho = 0.;
    double u = 0., v = 0.;
    for(int q = 0; q < lattice_q ; q += 1){
        int i = xy_q_to_xyq(P, q) ;
        rho += N[ i ];
        u += N[ i] *lattice_cx[q] ;
        v += N[ i] *lattice_cy[q] ;
    }
    u /= rho;
    v /= rho;
    for(int q = 0 ; q < lattice_q ; q ++){
        double vc = icsq * (v * lattice_cy[q] + u * lattice_cx[q] ) ;
        double neq = rho * lattice_w[q] * ( 1 + vc + vc*vc/2 - icsq/2 * (u*u + v*v) ) ; 
        int i = xy_q_to_xyq(P, q) ; 
        N[ i ] += (neq - N[i])/tau[ i ];
    }
}