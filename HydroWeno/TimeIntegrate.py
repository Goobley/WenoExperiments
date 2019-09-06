import numpy as np
from Weno import reconstruct_weno
from Flux import lax_friedrichs_flux
from PrimCons import cons2prim
from BCs import apply_bcs
from SimSettings import Sim

def tvd_rk3(dt, prim, cons):
    dtx = dt / Sim.GrDx
    consNew = np.zeros_like(cons)

    apply_bcs(prim)
    primRecon = reconstruct_weno(prim)
    flux = lax_friedrichs_flux(primRecon)
    consNew[:, Sim.GriBeg:Sim.GriEnd] = cons[:, Sim.GriBeg:Sim.GriEnd] - dtx * (flux[:, Sim.GriBeg+1:Sim.GriEnd+1] - flux[:, Sim.GriBeg:Sim.GriEnd])
    primNew = cons2prim(consNew)

    apply_bcs(primNew)
    primRecon[:] = reconstruct_weno(primNew)
    flux[:] = lax_friedrichs_flux(primRecon)
    consNew[:, Sim.GriBeg:Sim.GriEnd] = 0.75 * cons[:, Sim.GriBeg:Sim.GriEnd] + 0.25 * consNew[:, Sim.GriBeg:Sim.GriEnd] - \
                                        0.25 * dtx * (flux[:, Sim.GriBeg+1:Sim.GriEnd+1] - flux[:, Sim.GriBeg:Sim.GriEnd]) 
    primNew[:] = cons2prim(consNew)

    apply_bcs(primNew)
    primRecon[:] = reconstruct_weno(primNew)
    flux[:] = lax_friedrichs_flux(primRecon)
    consNew[:, Sim.GriBeg:Sim.GriEnd] = 1/3 * cons[:, Sim.GriBeg:Sim.GriEnd] + 2/3 * consNew[:, Sim.GriBeg:Sim.GriEnd] - \
                                        2/3 * dtx * (flux[:, Sim.GriBeg+1:Sim.GriEnd+1] - flux[:, Sim.GriBeg:Sim.GriEnd]) 
    primNew[:] = cons2prim(consNew)

    return primNew, consNew
