import numpy as np
import time

"""This module calculates second order Moller Plesset energy
and calpulated Laplace tranfrom MP2
"""

#Perform integral transformation for AO basis to MO basis
def transform_integrals_einsum(ge2_ao, coeff_mat):
    """AO to MO integral transformation using einsum
    """
    start_time = time.clock()
    g2e_mo = np.einsum('PQRS, Pp->pQRS', g2e_ao, coeff_mat)
    g2e_mo = np.einsum('pQRS, Qq->pqRS', g2e_ao, coeff_mat)
    g2e_mo = np.einsum('pqRS, Rr->pqrS', g2e_ao, coeff_mat)
    g2e_mo = np.einsum('pqrS, Ss->pqrs', g2e_ao, coeff_mat)
    print(time.clock() - start_time, "seconds, AO to MO transformed")

    return g2e_mo

##################################################
#
# Compute MP2 energy
#
##################################################

def compute_mp2_energy(num_bf, nocc, g2e_mo, orb_e, ehf):
    """
    Computes the MP2 energy
    num_bf :: number of basis functions
    nocc  :: number of occupied orbitals
    g2e_mo :: two-electron integrals in MO basis
    orb_e ::: orbital energies as obtained from HF SCF procedure
    ehf :: HF energy
    """
    g2e_mo = transform_integrals_einsum(g2e_ao, coeff_mat)

    start_time = time.clock()
    E = 0.0
    for i in range(nocc):
      for j in range(nocc):
          for a in range(nocc, num_bf):
            for b in range(nocc, num_bf):
              E += g2e_mo[i, a, j, b]*(2*g2e_mo[i, a, j, b] - g2e_mo[i, b, j, a])/\
                         (orb_e[i] + orb_e[j] - orb_e[a] - orb_e[b])

    print('MP2 correlation energy: {:20.15f}'.format(E))
    print('Total MP2 energy: {:20.15f}\n'.format(E + ehf))
    print('MP2 energy calculated in  seconds',  time.clock() - start_time)
    return E

print('MP2 energy = ', compute_mp2_energy(num_bf, nocc, g2e_mo, orb_e, ehf))
print('HF energy = ', ehf)


#################################################
#
# Laplace MP2 implementation
#
#################################################
print("\nLaplace mp2 implementation \n ")

def laplace():

  ngrid = 5
  grid, weights = np.polynomial.laguerre.laggauss(ngrid)
  weights = weights*np.exp(grid)

  return grid, weights


def laplace_mp2_energy(orb_e, g2e_mo):

  print('Looping over %d grid points...' % ngrid)
  t_start2 = time.time()
  
  g2e_mo = transform_integrals_einsum(g2e_ao, coeff_mat)
  E_mp2_corr = 0.0
  grid, weights = laplace()

  for t, w in zip(grid, weights):
    for i in range(nocc):
      for j in range(nocc):
        for a in range(nocc, num_bf):
          for b in range(nocc, num_bf):
            t_tot = np.exp(orb_e[i]*t + orb_e[j]*t - orb_e[a]*t - orb_e[b]*t)
            E_mp2_corr -= (g2e_mo[i, a, j, b]*(2*g2e_mo[i, a, j, b] - g2e_mo[i, b, j, a]))* t_tot * w
  
  print('LT-MP2 energy calculated in %.3f seconds.\n' % (time.time() - t_start2))                          

  return E_mp2_corr



if __name__ == '__main__':

  from pyscf import gto, scf
  mol = gto.M(
      atom = [['H', (0, 0, 0)],
              ['H', (1, 0, 0)],
              ['H', (2, 0, 0)],
              ['H', (3, 0, 0)],
              ['H', (4, 0, 0)],
              ['H', (5, 0, 0)],
              ],
      basis = 'sto-3g',
      verbose = 0)
  
  num_bf = mol.nao_nr()

  # overlap, kinetic, nuclear attraction
  s = mol.intor('cint1e_ovlp_sph')
  t = mol.intor('cint1e_kin_sph')
  v = mol.intor('cint1e_nuc_sph')

  # The one-electron part of the H is the sum of the kinetic and nuclear integrals
  h = t + v

  # 2e integrals (electron repulsion integrals, "eri")
  eri = mol.intor('cint2e_sph')

  # ERI is stored as [pq, rs] 2D matrix
  #print ("ERI shape=", eri.shape)

  # Reshape it into a [p,q,r,s] 4D array
  eri = eri.reshape([num_bf,num_bf,num_bf,num_bf])
  #print ("ERI2 shape=", eri.shape)
  g2e_ao = eri

  # Perform HF calculation to obtain orbital energies and MO orbital coefficients.
  conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))
  #print(("conv, e, mo_e, mo, mo_occ", conv, e, mo_e, mo, mo_occ))
  #mo_e and mo aka, mo coefficients.
  nocc = list(mo_occ).count(2 or 1)
  coeff_mat = mo
  orb_e = mo_e
  ehf = e

  E_mp2_corr = laplace_mp2_energy(orb_e, g2e_mo)

  E_mp2_tot = ehf + E_mp2_corr

  print('LT-MP2  correlation energy:    %16.10f' % E_mp2_corr)
  print('LT-MP2 total energy:           %16.10f' % E_mp2_tot)

