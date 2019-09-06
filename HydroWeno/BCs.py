from SimSettings import Prim, Sim, Gas

def apply_bcs(V):
    typ = 'Reflect'
    if typ == 'Reflect':
        for i in range(Sim.NumGhost):
            k0 = 2 * Sim.NumGhost - 1
            k1 = Sim.GriEnd - Sim.NumGhost

            V[:, i] = V[:, k0-i]
            V[Prim.Velo, i] = -V[Prim.Velo, k0-i]

            V[:, k1+k0-i] = V[:, k1+i]
            V[Prim.Velo, k1+k0-i] = -V[Prim.Velo, k1+i]
    if typ == 'LowerDiode':
        for i in range(Sim.NumGhost):
            k0 = 2 * Sim.NumGhost - 1
            k1 = GriEnd - Sim.NumGhost

            V[:, i] = V[:, k0-i]
            V[Prim.Velo, k1+k0-i] = min(0.0, V[Prim.Velo, k0-i])

            V[:, k1+k0-i] = V[:, k1+i]
            V[Prim.Velo, k1+k0-i] = -V[Prim.Velo, k1+i]
    elif typ == 'Fixed':
        V[Prim.Dens, :Sim.NumGhost] = 3.857143
        V[Prim.Velo, :Sim.NumGhost] = 2.629369
        V[Prim.Pres, :Sim.NumGhost] = 10.33333333
    
        for i in range(1, Sim.NumGhost+1):
            V[:, -i] = V[:, -Sim.NumGhost-1]
    
        V[Prim.Eint] = V[Prim.Pres] / ((Gas.Gamma-1.0) * V[Prim.Dens])
    elif typ == 'ZeroGrad':
        for i in range(Sim.NumGhost):
            V[:, i] = V[:, Sim.NumGhost]
            V[:, -(i+1)] = V[:, -Sim.NumGhost-1]
    
        V[Prim.Eint] = V[Prim.Pres] / ((Gas.Gamma-1.0) * V[Prim.Dens])