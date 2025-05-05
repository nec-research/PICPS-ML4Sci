"""
*          NAME OF THE PROGRAM THIS FILE BELONGS TO
*
*   file: ase_utils.py
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
from pathlib import Path
import copy
import logging
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory

from mdsim.common.registry import registry
from mdsim.common.utils import setup_imports, setup_logging
from mdsim.datasets import data_list_collater

def atoms_to_batch(atoms):
    atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
    positions = torch.Tensor(atoms.get_positions())
    cell = torch.Tensor(atoms.get_cell()).view(1, 3, 3)
    natoms = positions.shape[0]

    return Data(
        cell=cell,
        pos=positions,
        atomic_numbers=atomic_numbers,
        natoms=natoms,
    )

def data_to_atoms(data):
    numbers = data.atomic_numbers
    positions = data.pos
    cell = data.cell.squeeze()
    atoms = Atoms(numbers=numbers, 
                  positions=positions.cpu().detach().numpy(), 
                  cell=cell.cpu().detach().numpy(),
                  pbc=[True, True, True])
    return atoms

def batch_to_atoms(batch):
    n_systems = batch.natoms.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    # tags = torch.split(batch.tags, natoms)
    cells = batch.cell
    if batch.y is not None:
        energies = batch.y.tolist()
    else:
        energies = [None] * n_systems

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            cell=cells[idx].cpu().detach().numpy(),
            pbc=[True, True, True],
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects


class OCPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, config_yml=None, checkpoint=None,
                 test_data_src=None, energy_units_to_eV=1.):
        """
        OCP-ASE Calculator. The default unit for energy is eV.

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint (str):
                Path to trained checkpoint.
        """
        setup_imports()
        setup_logging()
        Calculator.__init__(self)

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint is not None

        if config_yml is not None:
            if isinstance(config_yml, str):
                config = yaml.safe_load(open(config_yml, "r"))

                if "includes" in config:
                    for include in config["includes"]:
                        # Change the path based on absolute path of config_yml
                        path = os.path.join(
                            config_yml.split("configs")[0], include
                        )
                        include_config = yaml.safe_load(open(path, "r"))
                        config.update(include_config)
            else:
                config = config_yml
        else:
            # Loads the config from the checkpoint directly
            config = torch.load(checkpoint, map_location=torch.device("cpu"))[
                "config"
            ]

            config["trainer"] = "forces"
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint

        for cfg in config['dataset']:
            cfg['src'] = test_data_src
        config['dataset'].append({'src': test_data_src, 'name': 'test'})

        self.trainer = registry.get_trainer_class(
            config.get("trainer", "energy")
        )(
            task=config["task"],
            model=config["model"],
            dataset=config["dataset"],
            optimizer=config["optim"],
            identifier=config["identifier"],
            timestamp_id=config.get("timestamp_id", None),
            run_dir=config.get("run_dir", None),
            is_debug=config.get("is_debug", False),
            print_every=config.get("print_every", 100),
            seed=config.get("seed", 0),
            logger=config.get("logger", "wandb"),
            local_rank=config["local_rank"],
            amp=config.get("amp", False),
            cpu=config.get("cpu", False),
            slurm=config.get("slurm", {}),
            no_energy=config.get("no_energy", False),
            simulate=True
        )

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        self.energy_units_to_eV = energy_units_to_eV

    def load_checkpoint(self, checkpoint_path):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = atoms_to_batch(atoms)
        batch = data_list_collater([data_object], otf_graph=True)

        predictions = self.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )

        self.results["energy"] = predictions["energy"].item() * self.energy_units_to_eV
        self.results["forces"] = predictions["forces"].cpu().numpy() * self.energy_units_to_eV


class OCPCalculator_Ensemble(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, config_ymls=None, checkpoints=None,
                 test_data_src=None, energy_units_to_eV=1.):
        """
        OCP-ASE Calculator. The default unit for energy is eV.

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint (str):
                Path to trained checkpoint.
        """
        setup_imports()
        setup_logging()
        Calculator.__init__(self)

        # Either the config path or the checkpoint path needs to be provided
        assert config_ymls or checkpoints is not None

        configs = []
        if config_ymls is not None:
            for config_yml in config_ymls:
                if isinstance(config_yml, str):
                    config = yaml.safe_load(open(config_yml, "r"))

                    if "includes" in config:
                        for include in config["includes"]:
                            # Change the path based on absolute path of config_yml
                            path = os.path.join(
                                config_yml.split("configs")[0], include
                            )
                            include_config = yaml.safe_load(open(path, "r"))
                            config.update(include_config)
                else:
                    config = config_yml

                configs.append(config)
        else:
            # Loads the config from the checkpoint directly
            for checkpoint in checkpoints:
                config = torch.load(checkpoint, map_location=torch.device("cpu"))[
                    "config"
                ]
                config["trainer"] = "forces"
                config["model_attributes"]["name"] = config.pop("model")
                config["model"] = config["model_attributes"]

                configs.append(config)

        for config in configs:
            # Calculate the edge indices on the fly
            config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        self.configs = copy.deepcopy(configs)

        for config, checkpoint in zip(self.configs, checkpoints):
            config["checkpoint"] = checkpoint

            for cfg in config['dataset']:
                cfg['src'] = test_data_src
            config['dataset'].append({'src': test_data_src, 'name': 'test'})

        self.trainers = []
        for config in configs:
            trainer = registry.get_trainer_class(
                config.get("trainer", "energy")
            )(
                task=config["task"],
                model=config["model"],
                dataset=config["dataset"],
                optimizer=config["optim"],
                identifier=config["identifier"],
                timestamp_id=config.get("timestamp_id", None),
                run_dir=config.get("run_dir", None),
                is_debug=config.get("is_debug", False),
                print_every=config.get("print_every", 100),
                seed=config.get("seed", 0),
                logger=config.get("logger", "wandb"),
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
                slurm=config.get("slurm", {}),
                no_energy=config.get("no_energy", False),
                simulate=True
            )
            self.trainers.append(trainer)

        if checkpoints is not None:
            self.load_checkpoint(checkpoints)

        self.energy_units_to_eV = energy_units_to_eV

    def load_checkpoint(self, checkpoint_paths):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            for trainer, checkpoint_path in zip(self.trainers, checkpoint_paths):
                trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = atoms_to_batch(atoms)
        batch = data_list_collater([data_object], otf_graph=True)

        energies, forces = [], []
        for trainer in self.trainers:
            predictions = trainer.predict(
                batch, per_image=False, disable_tqdm=True
            )

            energies.append(predictions["energy"].item() * self.energy_units_to_eV)
            forces.append(predictions["forces"].cpu().numpy() * self.energy_units_to_eV)

        self.results["energy"] = sum(energies) / len(self.trainers)
        self.results["forces"] = sum(forces) / len(self.trainers)


class NeuralMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 start_time=0,
                 verbose=True,
                 **kwargs):
        if start_time == 0:
            header = True
        else:
            header = False
        super().__init__(header=header, *args, **kwargs)
        """
        Logger uses ps units.
        """
        self.start_time = start_time
        self.verbose = verbose
        if verbose:
            print(self.hdr)
        self.natoms = self.atoms.get_number_of_atoms()

    def __call__(self):
        if self.start_time > 0 and self.dyn.get_time() == 0:
            return
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        mean_mv = np.mean(self.atoms.get_momenta())
        mean_M = np.mean(self.atoms.get_masses())
        ebulkK = 0.5 * mean_mv**2 / mean_M
        temp = (ekin/self.natoms - ebulkK) / (1.5 * units.kB)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000*units.fs) + self.start_time
            dat = (t,)
        else:
            dat = ()
        dat += (epot+ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress() / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt % dat)

class Simulator:
    def __init__(self, 
                 atoms, 
                 integrator,
                 T_init,
                 start_time=0,
                 save_dir='./log',
                 restart=False,
                 save_frequency=100,
                 min_temp=0.1,
                 max_temp=100000):
        self.atoms = atoms
        self.integrator = integrator
        self.save_dir = Path(save_dir)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.natoms = self.atoms.get_number_of_atoms()

        # intialize system momentum 
        if not restart:
            assert (self.atoms.get_momenta() == 0).all()
            MaxwellBoltzmannDistribution(self.atoms, T_init * units.kB)
        
        # attach trajectory dump 
        self.traj = Trajectory(self.save_dir / 'atoms.traj', 'a', self.atoms)
        self.integrator.attach(self.traj.write, interval=save_frequency)
        
        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, self.atoms, 
                                        self.save_dir / 'thermo.log', 
                                        start_time=start_time, mode='a'), 
                               interval=save_frequency)
        
    def run(self, steps):
        early_stop = False
        step = 0
        for step in tqdm(range(steps)):
            self.integrator.run(1)
            
            ekin = self.atoms.get_kinetic_energy()
            mean_mv = np.mean(self.atoms.get_momenta())
            mean_M = np.mean(self.atoms.get_masses())
            ebulkK = 0.5 * mean_mv ** 2 / mean_M
            temp = (ekin / self.natoms - ebulkK) / (1.5 * units.kB)
            if temp < self.min_temp or temp > self.max_temp:
                print(f'Temprature {temp:.2f} is out of range: \
                        [{self.min_temp:.2f}, {self.max_temp:.2f}]. \
                        Early stopping the simulation.')
                early_stop = True
                break
            
        self.traj.close()
        return early_stop, (step+1)