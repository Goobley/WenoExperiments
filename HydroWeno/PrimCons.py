import numpy as np
from numba import njit
from .SimSettings import Prim, Cons, Gas

SmallPressure = 1e-16

@njit('float64[:,:](float64[:,:])', cache=True)
def prim2cons(V):
    result = np.empty((Cons.NumVars, V.shape[1]))
    result[Cons.Dens] = V[Prim.Dens]
    result[Cons.Mome] = V[Prim.Dens] * V[Prim.Velo]
    eKin = 0.5 * V[Prim.Dens] * V[Prim.Velo]**2
    eInt = V[Prim.Pres] / (Gas.Gamma - 1.0)
    result[Cons.Ener] = eKin + eInt
    return result

@njit('float64[:](float64[:], float64[:])', cache=True)
def eos(dens, eInt):
    pres = np.maximum((Gas.Gamma - 1.0) * dens * eInt, SmallPressure)
    return pres

@njit('float64[:,:](float64[:,:])', cache=True)
def cons2prim(U):
    result = np.empty((Prim.NumVars, U.shape[1]))
    result[Prim.Dens] = U[Cons.Dens]
    result[Prim.Velo] = U[Cons.Mome] / U[Cons.Dens]
    eKin = 0.5 * result[Prim.Dens] * result[Prim.Velo]**2
    eInt = np.maximum(U[Cons.Ener] - eKin, SmallPressure)
    eInt /= U[Cons.Dens]
    pres = eos(U[Cons.Dens], eInt)
    result[Prim.Pres] = pres
    result[Prim.Eint] = eInt * U[Cons.Dens]
    return result

@njit('float64[:,:](float64[:,:])', cache=True)
def prim2flux(V):
    result = np.empty((Cons.NumVars, V.shape[1]))
    eKin = 0.5 * V[Prim.Dens] * V[Prim.Velo]**2
    eInt = V[Prim.Pres] / (Gas.Gamma - 1.0)
    ener = eKin + eInt

    result[Cons.Dens] = V[Prim.Dens] * V[Prim.Velo]
    result[Cons.Mome] = result[Cons.Dens]*V[Prim.Velo] + V[Prim.Pres]
    result[Cons.Ener] = V[Prim.Velo] * (ener + V[Prim.Pres])
    return result
