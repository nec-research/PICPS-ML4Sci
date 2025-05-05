"""
*          NAME OF THE PROGRAM THIS FILE BELONGS TO
*
*   file: ani_preprocess.py
*
* Authors: Makoto Takamoto (makoto.takamoto@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) 2025, All rights reserved.
*     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
*
*          PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT
ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution or non-profit organization or self (called "Licensee" or "You" in this Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor.
RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and condi\
tions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF LICENSOR’S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies \
permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials and to use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publicatio\
n. All publication materials mentioning features or use of this software must explicitly include an acknowledgement the software was developed by NEC Laboratories Europe GmbH.
COPYRIGHT: The Software is owned by Licensor.
PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and y\
ou may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement. Neither the name of NEC Laboratories Europe GmbH nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior \
written permission.
You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of L\
icensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement.
EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs administered by law.
SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Germany without reference to conflict of laws principles.  You consent to the personal jurisdiction of the courts of this country and waive their rights to venue outside of Germany.
ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.
*     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
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