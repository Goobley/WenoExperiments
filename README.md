## WENO Experiments

A simple set of toy experiments investigating the application of explicit
hydrodynamics schemes using WENO reconstruction, and also applying this to heat
conduction.

Inspired by lots of places, but attempting to be a basic framework design for
solving n conservation laws in Python whilst abstracting BCs etc and retaining
performance through the use of numba.