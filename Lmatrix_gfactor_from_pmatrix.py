#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from numpy import linalg as LA
import re
import os

# The script computes orbital angnular momentum matrix L from momentum matrix p and energies computed by JDFTx code
# and output L to totalE.L.
# It also compute g fator related quantities including
# energy changes of specific bands at different k points induced by B
# (file energy_change_Bext.out)
# and xpectation values of L, S and L+gs*S with gs=2.0023 along x,y,z
# (file angular_momenta_diag.out)

# Input paramters
dir_p = "./" # directory having totalE.momenta with more bands
             # dir_p should also have totalE.eigenvals and totalE.out
dir_s = "../" # directory having totalE.S with less bands 
              # band number is read from dir_s+'/totalE.out'
bstart_g = 24 # band range for g factor analysis
bend_g = 28 #
Bmag = 1 # The magnitude of test magnetic field in Tesla
         # from energy changes induced by a magnetic field, we can analyse g factors

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
p = np.fromfile(dir_p+"totalE.momenta",np.complex128).reshape(nk,3,nb_p,nb_p).swapaxes(2,3) # from Fortran to C
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
L.swapaxes(2,3).tofile("totalE.L") # from C to Fortran for JDFTx


# Analysis g factor
# Define g factor for band 1 and 2 as 
# 0.5 g_{i,12} B_i = Delta(E_2 - E_1)
# = [E_2(B_i) - E_1(B_i)] - [E_2(B_i=0) - E_1(B_i=0)] (suppose the band ordering is not changed),
# so g_{i,12} = Delta(E_2 - E_1) / (0.5 * B_i).
# In 1st order perturbation, g_{i,12} = (L+gs*S)^{exp}_{i,2} - (L+gs*S)^{exp}_{i,1},
# where exp means expectation value.
# Therefore, it is good to know E(B_i) and (L+gs*S)^{exp}_i

print('Analyse g factor for bands in range:',bstart_g,bend_g)

# JDFTx spin matrix is <1|pauli|2> without 0.5 factor
S = 0.5 * np.fromfile(dir_s+"totalE.S",np.complex128).reshape(nk_s,3,nb_s,nb_s).swapaxes(2,3) # from Fortran to C
Bmag = Bmag / 2.3505175675871e5 # convert to atomic unit

f_de = open("energy_change_Bext.out", "w")
f_de.write('#Energy changes induced by Bext at different k divided by 0.5 * B_i\n')
f_de.write('#g factor for band 1 and 2 is g_{i,12} = [ Delta E_2(B_i) - Delta E_1(B_i) ] / (0.5 * B_i)\n')
f_de.write('#Band range: %4s %4s\n' % (bstart_g, bend_g))
f_AM_diag = open("angular_momenta_diag.out", "w")
f_AM_diag.write('#Diagonal elements of L, S and L+gs*S at different k with gs=2.0023\n')
f_AM_diag.write('#g factor for band 1 and 2 is g_{i,12} = (L+gs*S)^{exp}_{i,2} - (L+gs*S)^{exp}_{i,1}\n')
f_AM_diag.write('#Band range: %4s %4s\n' % (bstart_g, bend_g))
s_dir = ['x','y','z']

for idir in range(3): # loop on x,y,z
  f_AM_diag.write('#Along '+s_dir[idir]+':\n')
  f_de.write('#Along '+s_dir[idir]+':\n')
  
  # Apply a magnetic field (suppose the band ordering is not changed)
  gs = 2.0023193043625635
  H = 0.5 * Bmag * (L[:,idir] + gs*S[:,idir]) # Total angular momentum
  for ik in range(nk):
    for ib in range(nb_s):
      H[ik,ib,ib] = H[ik,ib,ib] + e[ik,ib]
  ei,Ui = LA.eigh(H) # new eigenvalues and eigenvectors of Hamiltonian H(Bi)
  
  # Output energy changes induced by Bext
  dei = (ei[:,bstart_g:bend_g] - e[:,bstart_g:bend_g]) / (0.5*Bmag) # energy differences multiplied by 1 / (0.5 * B_i)
  np.savetxt(f_de, dei, fmt='%.4f')
  
  # Compute new L and S matrices and output diagonal elements of them and L+gs*S matrix
  Li_diag = np.einsum("kba,kbc,kca->ka", Ui.conj(), L[:,idir], Ui)[:,bstart_g:bend_g].real
  Si_diag = np.einsum("kba,kbc,kca->ka", Ui.conj(), S[:,idir], Ui)[:,bstart_g:bend_g].real
  np.savetxt(f_AM_diag, np.concatenate((Li_diag, Si_diag, Li_diag+gs*Si_diag), axis=1), fmt='%.4f')
  
f_de.close()
f_AM_diag.close()