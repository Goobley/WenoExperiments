import numpy as np
from numba import njit, stencil, prange

@njit('float64[:,:,:](float64[:,:], float64[:])', parallel=True, cache=True)
def reconstruct_weno(q, dx):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-36
    EnoCoeffL = np.array((( 11.0/6.0, -7.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0)))
    LinWL = np.array((0.3, 0.6, 0.1))
    EnoCoeffR = np.array((( 2.0/6.0,  -7.0/6.0,  11.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0)))
    LinWR = np.array((0.1, 0.6, 0.3))

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

@njit('float64[:,:,:](float64[:,:])', parallel=True, cache=True)
def reconstruct_weno_z(q):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-10
    EnoCoeffL = np.array((( 11.0/6.0, -7.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0)))
    LinWL = np.array((0.3, 0.6, 0.1))
    EnoCoeffR = np.array((( 2.0/6.0,  -7.0/6.0,  11.0/6.0),
                          (-1.0/6.0,   5.0/6.0,  2.0/6.0),
                          ( 2.0/6.0,   5.0/6.0, -1.0/6.0)))
    LinWR = np.array((0.1, 0.6, 0.3))

    # Loop over each row in the q matrix - we parallelise over rows
    for row in prange(nRows):
        beta = np.empty(3)
        betaZ = np.empty(3)
        for i in range(2, nGrid-2):
            # Compute beta, the smoothness indicator for each intepolating polynomial
            beta[0] = 13./12.*(q[row, i-2] - 2.*q[row, i-1] + q[row, i])**2 + 0.25*(q[row, i-2] - 4.*q[row, i-1] + 3.*q[row, i])**2
            beta[1] = 13./12.*(q[row, i-1] - 2.*q[row, i] + q[row, i+1])**2 + 0.25*(q[row, i-1] - q[row, i+1])**2
            beta[2] = 13./12.*(q[row, i] - 2.*q[row, i+1] + q[row, i+2])**2 + 0.25*(3.*q[row, i] - 4.*q[row, i+1] + q[row, i+2])**2
            tau5 = np.abs(beta[0] - beta[2])
            betaZ[:] = ((beta + WenoEps) / (beta + tau5 + WenoEps))

            # Compute and normalise the non-linear weights
            nonLinWL = LinWL / betaZ
            nonLinWR = LinWR / betaZ
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

@njit('float64[:,:,:](float64[:,:], float64[:])', parallel=True, cache=True)
def reconstruct_weno_nm(q, dx):
    nRows, nGrid = q.shape
    result = np.zeros((nRows, 2, nGrid))

    # Set up general and left/right coefficients.
    # The left/right coeffs are just flipped relative to each other
    Pow = 2
    WenoEps = 1e-34
    # EnoCoeffL = np.array(((-2.0/3.0,  -1.0/3.0,  2.0),
    #                       ( 2.0/3.0,   2.0/3.0, -1.0/3.0),
    #                       (-1.0/3.0,   2.0/3.0,  2.0/3.0)))
    # LinWL = np.array((0.3, 0.6, 0.1))
    # EnoCoeffR = np.array((( 2.0,      -1.0/3.0, -2.0/3.0),
    #                       (-1.0/3.0,   2.0/3.0,  2.0/3.0),
    #                       ( 2.0/3.0,   2.0/3.0, -1.0/3.0)))
    LinW = np.array((0.1, 0.6, 0.3))
    # length ratio designated rho
    rho = np.zeros(nGrid)
    rho[:-1] = dx[:-1] / (dx[:-1] + dx[1:])
    # print(rho)
    # rhoPrime is used for the double extrapolation performed for the first stencil
    # rhoPrime = np.zeros(nGrid)
    # rhoPrime[1:] = dx[1:] / (dx[:-1] + dx[1:])

    # Loop over each row in the q matrix - we parallelise over rows
    for row in prange(nRows):
        beta = np.empty((2,3))
        # betaZ = np.empty(3)
        stencil = np.empty((3, 3)) # coords are [left/right, stencilIdx, stencilEntry]
        for i in range(2, nGrid-2):
            # Compute beta, the smoothness indicator for each intepolating polynomial

            q00 = q[row, i-1] + rho[i-1] * (q[row, i-1] - q[row, i-2])
            # q00 = q[row, i-1] + rho[i-2] * (q[row, i-1] - q[row, i-2])
            q01 = rho[i] * q[row, i] + (1.0 - rho[i]) * q[row, i+1]
            q10 = rho[i-1] * q[row, i-1] + (1.0 - rho[i-1])* q[row, i]
            q11 = q[row, i+1] + rho[i+1] * (q[row, i+1] - q[row, i+2])
            # q11 = q[row, i+1] + rho[i+2] * (q[row, i+1] - q[row, i+2])
            q12 = rho[i+1] * q[row, i+1] + (1.0 - rho[i+1]) * q[row, i+2]
            q21 = rho[i-2] * q[row, i-2] + (1.0 - rho[i-2]) * q[row, i-1]


            # stencil[0, 0] = (2.0 * rhoPrime[i-1] + rho[i-2]) * q[row, i-1] + rhoPrime[i-1] * q[row, i-2]
            # stencil[0, 1] = (1.0 - rho[i-1]) * q[row, i-1] + rho[i-1] * q[row, i]
            # stencil[0, 2] = q[row, i]
            # stencil[1, 0] = stencil[0, 1]
            # stencil[1, 1] = q[row, i]
            # stencil[1, 2] = (1.0 - rho[i]) * q[row, i] + rho[i] * q[row, i+1]
            # stencil[2, 0] = stencil[1, 2]
            # stencil[2, 1] = q[row, i+1]
            # stencil[2, 2] = (1.0 - rho[i+1]) * q[row, i+1] + rho[i+1] * q[row, i+2]

            beta[0, 0] = 13.0/12.0*(2.0*q01 - 2.0*q11)**2 + 0.25*(4.0*q[row, i] - 2.0*q01 - 2.0*q11)**2
            beta[0, 1] = 13.0/12.0*(2.0*q01 - 4.0*q[row, i] + 2.0*q10)**2 + 0.25*(-2.0*q01 + 2.0*q10)**2
            beta[0, 2] = 13.0/12.0*(2.0*q10 - 4.0*q[row, i-1] + 2.0*q21)**2 + 0.25*(-6.0*q10 + 8.0*q[row, i-1] - 2.0*q21)**2

            beta[1, 0] = 13.0/12.0*(2.0*q10 - 2.0*q00)**2 + 0.25*(4.0*q[row, i] - 2.0*q10 - 2.0*q00)**2
            beta[1, 1] = 13.0/12.0*(2.0*q10 - 4.0*q[row, i] + 2.0*q01)**2 + 0.25*(-2.0*q10 + 2.0*q01)**2
            beta[1, 2] = 13.0/12.0*(2.0*q01 - 4.0*q[row, i+1] + 2.0*q12)**2 + 0.25*(-6.0*q01 + 8.0*q[row, i+1] - 2.0*q12)**2
            # tau5 = np.abs(beta[0] - beta[2])
            # betaZ[:] = ((beta + WenoEps) / (beta + tau5 + WenoEps))

            # Compute and normalise the non-linear weights
            nonLinWL = LinW / (WenoEps + beta[0, :])**Pow
            nonLinWR = LinW / (WenoEps + beta[1, :])**Pow
            nonLinWL /= np.sum(nonLinWL)
            nonLinWR /= np.sum(nonLinWR)

            # Compute the standard polynomial reconstructions
            enoIntpLR = np.zeros((2, 3))
            # for s in range(3):
            #     enoIntpL[s] = np.dot(stencil[s], EnoCoeffL[2-s]) 
            #     enoIntpR[s] = np.dot(stencil[s], EnoCoeffR[s]) 
            enoIntpLR[0, 0] = ( 6.0*q[row, i] - 1.0*q01         - 2.0*q11)
            enoIntpLR[0, 1] = (-1.0*q01       + 2.0*q[row, i]   + 2.0*q10)
            enoIntpLR[0, 2] = ( 2.0*q10       + 2.0*q[row, i-1] - 1.0*q21)
            enoIntpLR[1, 0] = ( 6.0*q[row, i] - 1.0*q10         - 2.0*q00)
            enoIntpLR[1, 1] = (-1.0*q10       + 2.0*q[row, i]   + 2.0*q01)
            enoIntpLR[1, 2] = ( 2.0*q01       + 2.0*q[row, i+1] - 1.0*q12)
            enoIntpLR /= 3.0

            # Combine the different polynomial reconstrucitions weighted by their non-linear weights
            result[row, 0, i] = np.dot(nonLinWL, enoIntpLR[0, :])
            result[row, 1, i] = np.dot(nonLinWR, enoIntpLR[1, :])
    result[:, 0, :2] = q[:, :2]
    result[:, 1, :2] = q[:, :2]
    result[:, 0, -2:] = q[:, -2:]
    result[:, 1, -2:] = q[:, -2:]
    return result