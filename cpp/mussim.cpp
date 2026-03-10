/** 
#=========================================

# 6   2   5
#  \  |  /
# 3---0---1  
#  /  |  \
# 7   4   8

#=========================================
*/

#include "mussim.h"

const int LATTICE_Q = 9;
const int SIZE_X = 40;
const int SIZE_Y = 30;
const int* Cx = new int [LATTICE_Q] {0,0,0,-1,0,1,-1,-1,1};
const int* Cy = new int [LATTICE_Q] {0,0,1,0,-1,1,1,-1,-1};

void equilibrium_from_moments(float* rho, float* u, float* v){
    float* p(float*a, float*b){
        return a;
    }
}
int idx(int i, int j, int q) {
    return q + LATTICE_Q * (j + SIZE_Y * i);
}

void calc_permutation(int* P) {
    for (int i = 0; i < SIZE_X; i++) {
        for (int j = 0; j < SIZE_Y; j++) {
            for (int q = 0; q < LATTICE_Q; q++) {
                int x = i + Cx[q] % SIZE_X;
                int y = j + Cy[q] % SIZE_Y;
                P[idx(x,y,q)]=idx(i,j,q);
            }
        }
    }
}

void stream(float* Nm, int* P, float*N) {
    int size = SIZE_X * SIZE_Y * LATTICE_Q;
    for (int k=0; k<size; k++) {
        N[k] = Nm[P[k]];
    }
}