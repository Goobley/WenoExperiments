import numpy as np
from numba import njit
from SimSettings import Gas, Cons, Prim
from PrimCons import prim2cons, prim2flux, cons2prim

@njit('float64[:,:](float64[:,:,:])', parallel=True)
def lax_friedrichs_flux(primRecon):
    consRecon = np.zeros((Cons.NumVars, primRecon.shape[1], primRecon.shape[2]))
    fluxLR = np.zeros_like(consRecon)
    fluxLR[:, 0, :] = prim2flux(primRecon[:, 0, :])
    fluxLR[:, 1, :] = prim2flux(primRecon[:, 1, :])
    consRecon[:, 0, :] = prim2cons(primRecon[:, 0, :])
    consRecon[:, 1, :] = prim2cons(primRecon[:, 1, :])

    csR = np.sqrt(Gas.Gamma * primRecon[Prim.Pres, 0] / primRecon[Prim.Dens, 0])
    csL = np.sqrt(Gas.Gamma * primRecon[Prim.Pres, 1] / primRecon[Prim.Dens, 1])
    maxC = np.zeros_like(csL)
    # The left hand sound speed is the right hand extrapolation of the cell below,
    # and the left hand one is the right hand extrapolation of the value in the
    # next cell
    # maxC[1:] = 0.5 * (csL[:-1] + np.abs(primRecon[Prim.Velo, 1, :-1]) + csR[1:] + np.abs(primRecon[Prim.Velo, 0, 1:]))
    maxC[1:] = 0.5 * (csL[:-1] + np.abs(primRecon[Prim.Velo, 1, :-1]) + csR[1:] + np.abs(primRecon[Prim.Velo, 0, 1:]))

    flux = np.zeros((Cons.NumVars, primRecon.shape[2]))
    flux[:, 1:] = 0.5 * (fluxLR[:, 1, :-1] + fluxLR[:, 0, 1:] - maxC[1:] * (consRecon[:, 0, 1:] - consRecon[:, 1, :-1]))
    return flux