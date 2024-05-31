"""
*
*     SOFTWARE NAME
*
*        File:  ani_preprocess.py
*
*     Authors: Deleted for purposes of anonymity
*
*     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
*
* The software and its source code contain valuable trade secrets and shall be maintained in
* confidence and treated as confidential information. The software may only be used for
* evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
* license agreement or nondisclosure agreement with the proprietor of the software.
* Any unauthorized publication, transfer to third parties, or duplication of the object or
* source code---either totally or in part---is strictly prohibited.
*
*     Copyright (c) 2024 Proprietor: Deleted for purposes of anonymity
*     All Rights Reserved.
*
* THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
* AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
* DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.
*
* NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
* IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
* LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
* FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
* OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
* ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
* TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGES.
*
* For purposes of anonymity, the identity of the proprietor is not given herewith.
* The identity of the proprietor will be given once the review of the
* conference submission is completed.
*
* THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
*
"""


import numpy as np
import dataloader as dl

def atomic_energy(labels):
  # --- computes the sum of atomic energies ---
  # --- to obtain cohesive/atomization energies ---
  dict = {
      1: -0.500607632585, # H
      6: -37.8302333826, # C
      7: -54.5680045287, # N
      8: -75.0362229210  # O
      }
  out = np.array([dict[key] for key in labels]).sum()
  return out

if __name__ == '__main__':
  # Path to the ANI-1x data set
  path_to_h5file = 'ani1x-release.h5'
  # List of keys to point to requested data
  data_keys = ['wb97x_dz.energy','wb97x_dz.forces'] # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
    # Example for extracting DFT/DZ energies and forces
  N_max = -np.inf
  N_data = 4956005
  data_set = "ani1x_wb97x_dz.npz"
  energy_scale = 627.50947406 # Hartree -> kcal/mol: 627.50947406
  for data in dl.iter_data_buckets(path_to_h5file, keys=data_keys):
    N = len(data['atomic_numbers'])
    if N > N_max:
      N_max = N
      #atoms = Atoms(positions=data['coordinates'][0])
      #atoms.set_atomic_numbers(data['atomic_numbers'])
      #write('test.xyz', atoms)
  #exit()
  R = np.zeros(shape=(N_data, N_max, 3))
  E = np.zeros(shape=(N_data, ))
  F = np.zeros(shape=(N_data, N_max, 3))
  N = np.zeros(shape=(N_data, ))
  Z = np.zeros(shape=(N_data, N_max))
  counter_global = 0
  for data in dl.iter_data_buckets(path_to_h5file, keys=data_keys):
    R_data = data['coordinates']
    Z_data = data['atomic_numbers']
    E_data = data['wb97x_dz.energy']
    F_data = data['wb97x_dz.forces']
    for i_data in range(len(R_data)):
      N[counter_global] = len(Z_data)
      Z[counter_global, :len(Z_data)] = Z_data
      R[counter_global, :len(Z_data)] = R_data[i_data]
      E[counter_global] = E_data[i_data] - atomic_energy(Z_data)
      F[counter_global, :len(Z_data)] = F_data[i_data]
      counter_global += 1
  F *= energy_scale
  E *= energy_scale
  N = N.astype(int)
  Z = Z.astype(int)
  print(N[:392])
  print(Z[:392])
  np.savez(data_set, N=N, E=E, R=R, F=F, Z=Z)