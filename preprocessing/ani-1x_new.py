"""
*
*     SOFTWARE NAME
*
*        File:  ani-1x_new.py
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
