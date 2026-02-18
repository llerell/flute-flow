import time
import numpy as np
def thisisatest():
    n = 1000000

# Using lists
    A = [2*i+1 for i in range(n)]
    B = [2*i-1 for i in range(n)]
    tic = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
    C = [A[i]+B[i] for i in range(n)]
    toc = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
    print("Timewith lists: ", (toc-tic)//1000)


# Using numpy arrays
    Ap=np.array([2*i+1 for i in range(n)])
    Bp=np.array([2*i-1 for i in range(n)])

    tic = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
    Cp=Ap+Bp
    toc = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)

    print("Time with np arrays: ", (toc-tic)//1000)

# Using numpy arrays as lists
    tic = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)
    C2 = [Ap[i] + Bp[i] for i in range(n)]
    toc = time.clock_gettime_ns(time.CLOCK_PROCESS_CPUTIME_ID)

    print("Time with np arrays with for loop: ", (toc-tic)//1000)
if __name__ == "__main__":
    thisisatest()