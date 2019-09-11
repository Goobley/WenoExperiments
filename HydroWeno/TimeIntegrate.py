import numpy as np
from .Weno import reconstruct_weno, reconstruct_weno_z, reconstruct_weno_nm
from .Flux import lax_friedrichs_flux, lax_wendroff_flux, gforce_flux
from .PrimCons import cons2prim
from .SimSettings import Prim, Sim

def tvd_rk3(apply_bcs, reconstruct, flux_fn):
    def tvd_rk3_impl(dt, grid, prim, cons):
        # dx = np.ones(prim.shape[1]) * Sim.GrDx
        consNew = np.zeros_like(cons)
        GriBeg = grid.griBeg
        GriEnd = grid.griEnd
        dtx = (dt / grid.dx)[GriBeg:GriEnd]

        apply_bcs(grid, prim)
        primRecon = reconstruct(prim, grid.dx)
        flux = flux_fn(primRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = cons[:, GriBeg:GriEnd] - dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd])
        primNew = cons2prim(consNew)

        apply_bcs(grid, primNew)
        primRecon[:] = reconstruct(primNew, grid.dx)
        flux[:] = flux_fn(primRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = 0.75 * cons[:, GriBeg:GriEnd] + 0.25 * consNew[:, GriBeg:GriEnd] - 0.25 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd]) 
        primNew[:] = cons2prim(consNew)

        apply_bcs(grid, primNew)
        primRecon[:] = reconstruct(primNew, grid.dx)
        flux[:] = flux_fn(primRecon, grid.dx, dt)
        consNew[:, GriBeg:GriEnd] = 1/3 * cons[:, GriBeg:GriEnd] + 2/3 * consNew[:, GriBeg:GriEnd] - 2/3 * dtx * (flux[:, GriBeg+1:GriEnd+1] - flux[:, GriBeg:GriEnd]) 
        primNew[:] = cons2prim(consNew)

        return primNew, consNew
    return tvd_rk3_impl
