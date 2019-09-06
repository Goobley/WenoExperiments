import numpy as np
from numba import njit, stencil, prange

@njit('float64[:,:,:](float64[:,:])', parallel=True)
def reconstruct_weno(q):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-36
    EnoCoeffL = np.array((( 1.875, -1.25 ,  0.375),
                          ( 0.375,  0.75 , -0.125),
                          (-0.125,  0.75 ,  0.375)))
    LinWL = np.array((5.0/16.0, 10.0/16.0, 1.0/16.0))
    EnoCoeffR = np.array((( 0.375, -1.25 ,  1.875),
                          (-0.125,  0.75 ,  0.375),
                          ( 0.375,  0.75 , -0.125)))
    LinWR = np.array((1.0/16.0, 10.0/16.0, 5.0/16.0))

    # Loop over each row in the q matrix - we parallelise over rows
    for row in prange(nRows):
        beta = np.empty(3)
        for i in range(2, nGrid-2):
            # Compute beta, the smoothness indicator for each intepolating polynomial
            beta[0] = 13./12.*(q[row, i-2] - 2.*q[row, i-1] + q[row, i])**2 + 0.25*(q[row, i-2] - 4.*q[row, i-1] + 3.*q[row, i])**2
            beta[1] = 13./12.*(q[row, i-1] - 2.*q[row, i] + q[row, i+1])**2 + 0.25*(q[row, i-1] - q[row, i+1])**2
            beta[2] = 13./12.*(q[row, i] - 2.*q[row, i+1] + q[row, i+2])**2 + 0.25*(3.*q[row, i] - 4.*q[row, i+1] + q[row, i+2])**2

            # Compute and normalise the non-linear weights
            nonLinWL = LinWL / (WenoEps + beta)**Pow
            nonLinWR = LinWR / (WenoEps + beta)**Pow
            nonLinWL /= np.sum(nonLinWL)
            nonLinWR /= np.sum(nonLinWR)

            # Compute the standard polynomial reconstructions
            enoIntpL = np.zeros(3)
            enoIntpR = np.zeros(3)
            for s in range(3):
                gridIdx = s + i - 2
                enoIntpL[s] = np.dot(q[row, gridIdx:gridIdx+3], EnoCoeffL[2-s]) 
                enoIntpR[s] = np.dot(q[row, gridIdx:gridIdx+3], EnoCoeffR[s]) 

            # Combine the different polynomial reconstrucitions weighted by their non-linear weights
            result[row, 0, i] = np.dot(nonLinWL, enoIntpL)
            result[row, 1, i] = np.dot(nonLinWR, enoIntpR)
    result[:, 0, :2] = q[:, :2]
    result[:, 1, :2] = q[:, :2]
    result[:, 0, -2:] = q[:, -2:]
    result[:, 1, -2:] = q[:, -2:]
    return result