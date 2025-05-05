"""
*          NAME OF THE PROGRAM THIS FILE BELONGS TO
*
*   file: ani-1x_new.py
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
import argparse
from pathlib import Path
import pickle
import random

import lmdb
import numpy as np
from tqdm import tqdm

from arrays_to_graphs import AtomsToGraphs

random.seed(2023)

npzname = 'ani1x_wb97x_dz.npz'
Data_Size = {
    '50': 50,
    '100': 100,
    '1K': 1000,
    '10K': 10000,
    }

def write_to_lmdb(data_path, db_path, vt_num=10000, size="10K"):
    a2g = AtomsToGraphs(
        max_neigh=50, #1000,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=True,
        device='cpu'
    )

    data_file = Path(data_path) / npzname
    Path(data_path).mkdir(parents=True, exist_ok=True)
    if not data_file.is_file():
        print('data does not exist')
        ValueError
    #    download(molecule, data_path)
    all_data = np.load(data_file)
    n_points = all_data['N'].shape[0]
    atomic_numbers = all_data['Z'] + 1  #  atomic number start from 0?
    atomic_numbers = atomic_numbers.astype(np.int64)
    positions = all_data['R']
    force = all_data['F']
    energy = all_data['E'][:,None]
    lengths = np.ones(3)[None, :] * 30.
    angles = np.ones(3)[None, :] * 90.

    n_tr, n_vl, n_ts = n_points - vt_num*2, vt_num, vt_num
    n_tr = min(Data_Size[size], n_tr)
    n_vl = max(min(n_tr, n_vl), 100)

    for dataset_size, train_size, val_size, test_size in zip([size], [n_tr], [n_vl], [n_ts]):
        print(f'processing dataset with size {dataset_size}.')
        indx = [i for i in range(n_points)]
        random.shuffle(indx)
        indx = np.array(indx)
        train, val, test = indx[:n_tr], indx[n_tr:n_tr + n_vl], indx[-n_ts:]
        ranges = [train, val, test]
        print('size of train, val, test: ', train.shape[0], val.shape[0], test.shape[0])

        norm_stats = {
            'e_mean': energy[train].mean(),
            'e_std': energy[train].std(),
            'f_mean': force[train].mean(),
            'f_std': force[train].std(),
        }
        save_path = Path(db_path) / dataset_size
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / 'metadata', norm_stats)
        
        for spidx, split in enumerate(['train', 'val', 'test']):
            print(f'processing split {split}.')
            save_path = Path(db_path) / dataset_size / split
            save_path.mkdir(parents=True, exist_ok=True)
            db = lmdb.open(
                str(save_path / 'data.lmdb'),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )
            for i, idx in enumerate(tqdm(ranges[spidx])):
                natoms = np.array([positions.shape[1]] * 1, dtype=np.int64)
                data = a2g.convert(natoms, positions[idx], atomic_numbers[idx],
                                   lengths, angles, energy[idx], force[idx])

                data.sid = 0
                data.fid = idx
                txn = db.begin(write=True)
                txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()

            # Save count of objects in lmdb.
            txn = db.begin(write=True)
            txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
            txn.commit()

            db.sync()
            db.close()

            # nequip
            if split == 'train':
                _idx = train
            elif split == 'val':
                _idx = val
            else:
                _idx = test

            data = {
                'z': atomic_numbers,
                'E': energy[_idx],
                'F': force[_idx],
                'R': all_data['R'][_idx]
            }
            np.savez(save_path / 'nequip_npz', **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='../DATAPATH/ani-1x')
    parser.add_argument("--db_path", type=str, default='../DATAPATH/ani-1x')
    parser.add_argument("--size", type=str, default='100K')
    parser.add_argument("--vt_num", type=float, default=10000)
    args = parser.parse_args()
    write_to_lmdb(args.data_path, args.db_path, args.vt_num, args.size)
