#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from numpy import linalg as LA
import re
import os

# The script computes orbital angnular momentum matrix L from momentum matrix p and energies computed by JDFTx code
# and output L to totalE.L.
# It also computes g factor matrices as g = 2 + L S^-1
# It also writes diagonal elements of L and g matrices to Ldiag_gdiag.out
# and matrix elements of g factor for a few bands to gfac_mat.out

# Input paramters
dir_p = "./" # directory having totalE.momenta with more bands
             # dir_p should also have totalE.eigenvals and totalE.out
dir_s = "../" # directory having totalE.S with less bands 
              # band number is read from dir_s+'/totalE.out'
bstart_g = 24 # band range for g factor analysis
bend_g = 28 #

# Read totalE.out
def read_totalE_out(dir_):
  for line in open(dir_+"totalE.out"):
    if "nBands" in line:
      nb = int(re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)[1])
      nk = int(re.findall(r"[+-]?\d+\.\d*|[+-]?\d+", line)[2])
  return nk, nb

nk, nb_p = read_totalE_out(dir_p)
print("number of k points: ",nk)
print("number of bands for momenta: ",nb_p)
nk_s, nb_s = read_totalE_out(dir_s)
if nk != nk_s:
  print("numbers of k points are not the same")
  exit(1)
print("number of bands for spin: ",nb_s)

# Read momentum matrix p and energies e
p = np.fromfile(dir_p+"totalE.momenta",np.complex128).reshape(nk,3,nb_p,nb_p)
e = np.fromfile(dir_p+"totalE.eigenvals",np.float64).reshape(nk,nb_p)

# r_mn = -i p_mn / (e_m - e_n) with e_m - e_m != 0
degthr = 1e-8
r = np.copy(p[:,:,0:nb_s,0:nb_p])
for ik in range(nk):
  for idir in range(3):
    for ib in range(nb_s):
      for jb in range(nb_p):
        if (abs(e[ik,ib] - e[ik,jb]) > degthr):
          r[ik,idir,ib,jb] = p[ik,idir,ib,jb] / (e[ik,ib] - e[ik,jb]) # without prefactor -i

# L = r X p
L = np.zeros((nk_s,3,nb_s,nb_s),np.complex128)
L[:,0] = -1j * (np.einsum("kab,kbc->kac", r[:,1], p[:,2,0:nb_p,0:nb_s]) - np.einsum("kab,kbc->kac", r[:,2], p[:,1,0:nb_p,0:nb_s]))
L[:,1] = -1j * (np.einsum("kab,kbc->kac", r[:,2], p[:,0,0:nb_p,0:nb_s]) - np.einsum("kab,kbc->kac", r[:,0], p[:,2,0:nb_p,0:nb_s]))
L[:,2] = -1j * (np.einsum("kab,kbc->kac", r[:,0], p[:,1,0:nb_p,0:nb_s]) - np.einsum("kab,kbc->kac", r[:,1], p[:,0,0:nb_p,0:nb_s]))
L.tofile("totalE.L")


# Analysis g factor
# L + 2S = g S. So g = 2 + L S^-1 is a vector matrix
# Note that if g is not proportional to Identity, g will depend on unitary rotation

print('Analyse g factor for bands in range:',bstart_g,bend_g)
nb_g = bend_g - bstart_g
S = 0.5 * np.fromfile(dir_s+"totalE.S",np.complex128).reshape(nk_s,3,nb_s,nb_s) # JDFTx spin matrix is <1|pauli|2> without 0.5 factor
Sinv = LA.inv(S)
g = np.einsum("kiab,kibc->kiac", L[:,:,bstart_g:bend_g,0:nb_s], Sinv[:,:,0:nb_s,bstart_g:bend_g])
twoI = 2*np.eye(nb_g)
g = g + twoI[None,None,:]

# Output diagonal elements of L and g factor matrices
Ldiag = np.einsum("kdii->kdi", L[:,:,bstart_g:bend_g,bstart_g:bend_g]).real
gdiag = np.einsum("kdii->kdi", g).real
with open('Ldiag_gdiag.out', 'w') as f:
  f.write('#Diagonal elements of L and g at different k\n')
  f.write('#Band range: %4s %4s\n' % (bstart_g, bend_g))
  f.write('#Along x:\n')
  np.savetxt(f, np.concatenate((Ldiag[:,0], gdiag[:,0]), axis=1), fmt='%.3f')
  f.write('#Along y:\n')
  np.savetxt(f, np.concatenate((Ldiag[:,1], gdiag[:,1]), axis=1), fmt='%.3f')
  f.write('#Along z:\n')
  np.savetxt(f, np.concatenate((Ldiag[:,2], gdiag[:,2]), axis=1), fmt='%.3f')

# Output g factor matrices
g = g.reshape(nk_s,3,-1)
with open('gfac_mat.out', 'w') as f:
  f.write('#Along x:\n')
  np.savetxt(f,g[:,0],fmt='%.7e')
  f.write('#Along y:\n')
  np.savetxt(f,g[:,1],fmt='%.7e')
  f.write('#Along z:\n')
  np.savetxt(f,g[:,2],fmt='%.7e')