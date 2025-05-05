"""
*          NAME OF THE PROGRAM THIS FILE BELONGS TO
*
*   file: simulate.py
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
import yaml
import json
import argparse
import os
import time 
import subprocess
import random
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from ase import units
from ase.io import Trajectory
from ase.md.verlet import VelocityVerlet

try:
    from nequip.ase.nequip_calculator import nequip_calculator
except:
    pass

try: 
    from deepmd.calculator import DP
    from mdsim.common.deepmd_utils import get_param_count, dp_test
except:
    pass

import mdsim.md.integrator as md_integrator
from mdsim.md.ase_utils import data_to_atoms, OCPCalculator, Simulator
from mdsim.common.utils import load_config
from mdsim.datasets.lmdb_dataset import LmdbDataset

def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runcmd(cmd_list):
    return subprocess.run(cmd_list, universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
def eval_and_init(config):
    # load model.
    model_dir = config["model_dir"]
    save_name = config["save_name"]
    if not config['no_evaluate']:
        print('get test metrics.')
    if config['sim_type'] == 'ocp':
        model_ckpt = str(Path(model_dir) / 'checkpoints/best_checkpoint.pt')
        model_config = str(Path(model_dir) / 'checkpoints/config.yml')
        
        if 'test_dataset_src' not in config:
            config['test_dataset_src'] = config['dataset_src']
        calculator = OCPCalculator(config_yml=model_config, checkpoint=model_ckpt,
                                   test_data_src=config['test_dataset_src'],
                                   energy_units_to_eV=config["energy_units_to_eV"])
        if config['no_evaluate']:
            test_metrics = {}
        else:
            test_metrics = calculator.trainer.validate('test', max_points=config["max_test_points"])
        test_metrics['num_params'] = sum(p.numel() for p in calculator.trainer.model.parameters())
    elif config['sim_type'] == 'dp':
        # deploy model.
        if not (Path(model_dir) / 'frozen_model.pb').exists():
            os.system(f'dp freeze -c {model_dir}/ -o {model_dir}/frozen_model.pb')
        param_count = get_param_count(f'{model_dir}/frozen_model.pb')
        if config['no_evaluate']:
            test_metrics = {}
        else:
            data_config = config['dp_data_path']
            test_metrics = dp_test(model=f'{model_dir}/frozen_model.pb', system=data_config, numb_test=10000)
        calculator = DP(model=str(Path(model_dir) / 'frozen_model.pb'))
        test_metrics.update({'num_params': param_count})
        
    elif config['sim_type'] == 'nequip':
        # deploy model.
        if not Path(f'nequip-deploy build --train-dir {model_dir}/ {model_dir}/deployed_model.pth').exists():
            os.system(f'nequip-deploy build --train-dir {model_dir}/ {model_dir}/deployed_model.pth')
        data_config = config['nequip_data_config']        
        if config['no_evaluate']:
            test_metrics = {}
        else:
            # call nequip evaluation script.
            os.system(f'nequip-evaluate --train-dir {model_dir} --dataset-config {data_config} \
                    --log {model_dir}/{save_name}/test_metric.log --batch-size 4')
            
            with open(f'{model_dir}/{save_name}/test_metric.log', 'r') as f:
                test_log = f.read().splitlines()
                for i, line in enumerate(test_log):
                    if 'Final result' in line:
                        test_log = test_log[(i+1):]
                        break
                test_metrics = {}
                for line in test_log:
                    k, v = line.split('=')
                    k = k.strip()
                    v = float(v.strip())
                    test_metrics[k] = v
        
            with open(str(Path(model_dir)  / 'log'), 'r') as f:
                lines = f.read().splitlines()
                test_metrics['num_params'] = int(lines[1].split(' ')[-1])
        
        calculator = nequip_calculator(Path(model_dir) / 'deployed_model.pth', device='cuda',
                                       energy_units_to_eV=1.)
    else:
        raise NotImplementedError()
    
    print(test_metrics)
    with open(Path(config["model_dir"]) / config['save_name'] / 'test_metric.json', 'w') as f:
        json.dump(test_metrics, f)

    return calculator, test_metrics
    
def simulate(config, calculator, test_metrics):
    
    (Path(config["model_dir"]) / config['save_name']).mkdir(parents=True, exist_ok=True)
    trajectory_path = (Path(config["model_dir"]) / config['save_name'] / 'atoms.traj')
    thermo_log_path = (Path(config["model_dir"]) / config['save_name'] / 'thermo.log')
    
    RESTART = False
    if trajectory_path.exists():
        if not thermo_log_path.exists():
            raise ValueError('trajectory exists but thermo.log does not exist.')
        history = Trajectory(trajectory_path)
        if len(history) > 0 and not config['purge']:
            atoms = history[-1]
            with open(thermo_log_path, 'r') as f:
                last_line = f.read().splitlines()[-1]
            simulated_time = [float(x) for x in last_line.split(' ') if x][0]
            simulated_step = int(simulated_time / config['integrator_config']['timestep'] * 1000)
            RESTART = True
            print(f'Found existing simulation. Simulated time: {simulated_time} ps')
        else:
            os.remove(trajectory_path)
            os.remove(thermo_log_path)

    if not RESTART:
        test_dataset = LmdbDataset({'src': config['dataset_src']})
        if 'init_idx' in config:
            init_idx = config['init_idx']
        else:
            init_idx = random.randint(0, len(test_dataset))
            
        init_data = test_dataset[init_idx]    
        atoms = data_to_atoms(init_data)
        simulated_time = 0
        simulated_step = 0
        print('Start simulation from scratch.')
        
    if simulated_step > config['steps']:
        print(f'Simulated step {simulated_step} > {config["steps"]}. Simulation already complete.')
        return
    
    save_dir = Path(config["model_dir"]) / config['save_name'] 
    
    # set calculator.   
    if config['plumed']:
        # plumed setting is explicitly for alanine dipeptide.
        from ase.calculators.plumed import Plumed
        if RESTART:
            plumed_meta_cmd = f"metad: METAD ARG=phi,psi PACE=500 HEIGHT=1.2 SIGMA=0.35,0.35 FILE={str(save_dir)}/HILLS RESTART=YES BIASFACTOR=6.0"
        else:
            plumed_meta_cmd = f"metad: METAD ARG=phi,psi PACE=500 HEIGHT=1.2 SIGMA=0.35,0.35 FILE={str(save_dir)}/HILLS BIASFACTOR=6.0"
        setup = [f"UNITS LENGTH=A TIME={1/(1000 * units.fs)} ENERGY={units.mol/units.kJ}",
         "MOLINFO STRUCTURE=./alanine_dipeptide_files/ala2.pdb",
         "phi: TORSION ATOMS=@phi-2 ",
         "psi: TORSION ATOMS=@psi-2 ",
         plumed_meta_cmd,
         f"PRINT ARG=phi,psi FILE={str(save_dir)}/COLVAR STRIDE=10"]
        atoms.set_calculator(
            Plumed(calc=calculator,
                   input=setup,
                   timestep=config['integrator_config']['timestep'] * units.fs,
                   atoms=atoms,
                   kT=1.0))
    else:
        atoms.set_calculator(calculator)
    
    # adjust units.
    config["integrator_config"]["timestep"] *= units.fs
    if config["integrator"] in ['NoseHoover', 'NoseHooverChain', 'NoseHooverMid', 'NoseHooverChainMid']:
        config["integrator_config"]["temperature"] *= units.kB
        
    # set up simulator.
    if config["integrator"] == "VelocityVerlet":
        integrator = VelocityVerlet(atoms, **config["integrator_config"])
    else:
        integrator = getattr(md_integrator, config["integrator"])(
            atoms, **config["integrator_config"])
    simulator = Simulator(atoms, integrator, config["T_init"],
                          restart=RESTART,
                          start_time=simulated_time,
                          save_dir=Path(config["model_dir"]) / config['save_name'], 
                          save_frequency=config["save_freq"])
    
    # run simulation.
    start_time = time.time()    
    early_stop, step = simulator.run(config["steps"] - simulated_step)
    elapsed = time.time() - start_time
    test_metrics['running_time'] = elapsed
    test_metrics['early_stop'] = early_stop
    test_metrics['simulated_frames'] = step

    with open(Path(config["model_dir"]) / config['save_name'] / 'test_metric.json', 'w') as f:
        json.dump(test_metrics, f)

def main(config):
    seed_everywhere(config['seed'])
    save_name = 'md'
    if config["identifier"] is not None:
        save_name = 'md_' + config["identifier"] + '_' + str(config["seed"])
    if 'init_idx' in config:
        save_name = save_name + '_init_' + str(config['init_idx'])
    config['save_name'] = save_name
    os.makedirs(Path(config["model_dir"]) / save_name, exist_ok=True)
    with open(os.path.join(Path(config["model_dir"]) / save_name, 'config.yml'), 'w') as yf:
        yaml.dump(config, yf, default_flow_style=False)
        
    calculator, test_metrics = eval_and_init(config)
    simulate(config, calculator, test_metrics)
    
if __name__ == '__main__':       
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yml", required=True, type=Path)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--identifier", type=str)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init_idx", type=int, 
                        help='the index of the initial state selected from the init dataset.')
    parser.add_argument("--deepmd", action='store_true')
    parser.add_argument("--nequip", action='store_true')
    parser.add_argument("--plumed", action='store_true',
                        help='MetaDynamics. only used for alanine dipeptide.')
    parser.add_argument("--no_evaluate", action='store_true', 
                        help='if <True>, skip force error evaluation and directly start simulation.')
    parser.add_argument("--purge", action='store_true',
                        help='if <True>, remove the previous run if exists.')
    parser.add_argument("--energy_units_to_eV", type=float)

    args, override_args = parser.parse_known_args()
    config, _, _ = load_config(args.config_yml)
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(overrides)
    if args.deepmd and args.nequip:
        raise ValueError('cannot set both <deepmd> and <nequip> to <True>.')
    elif args.nequip:
        config['sim_type'] = 'nequip'
    elif args.deepmd:
        config['sim_type'] = 'dp'
    main(config)
