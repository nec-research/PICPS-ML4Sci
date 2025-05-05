"""
*          NAME OF THE PROGRAM THIS FILE BELONGS TO
*
*   file: main.py
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
import copy
import logging
import os
import yaml
import time
import _pickle as cPickle
import warnings
warnings.filterwarnings("ignore")


import submitit

from mdsim.common import distutils
from mdsim.common.flags import flags
from mdsim.common.registry import registry
from mdsim.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
    compose_data_cfg
)


class Runner(submitit.helpers.Checkpointable):
    def __init__(self, if_optuna=False, if_robust_loss=None, coef_rob=None, coef_cyc=None, delta=None,
                 lr=None, rr=None, loss=None, Cf=None, Ce=None, consis_case=None):
        self.config = None
        self.if_optuna = if_optuna
        self.if_robust_loss = if_robust_loss
        self.coef_rob = coef_rob
        self.coef_cyc = coef_cyc
        self.delta = delta
        self.lr = lr
        self.rr = rr
        self.loss = loss
        self.Cf = Cf
        self.Ce = Ce
        self.consis_case = consis_case

    def __call__(self, config):
        setup_logging()
        self.config = copy.deepcopy(config)

        if config['distributed']:
            distutils.setup(config)

        try:
            setup_imports()
            
            # compose dataset configs.
            train_data_cfg = config['dataset']
            train_data_cfg = compose_data_cfg(train_data_cfg)
            config['dataset'] = [
                train_data_cfg,
                {'src': os.path.join(os.path.dirname(train_data_cfg['src']), 'val')},
                {'src': os.path.join(os.path.dirname(train_data_cfg['src']), 'test')}
            ]
            
            if self.if_optuna:
                if self.loss is not None:
                    config["optim"].update({"loss_energy": self.loss})
                    config["optim"].update({"loss_force": self.loss})
                if self.rr is not None:
                    config["optim"].update({"weight_decay": self.rr})
                if self.lr is not None:
                    config["optim"].update({"lr_initial": self.lr})
                if self.Cf is not None:
                    config["optim"].update({"force_coefficient": self.Cf})
                if self.Ce is not None:
                    config["optim"].update({"energy_coefficient": self.Ce})
                if self.consis_case is not None:
                    config["optim"].update({"consis_case": self.consis_case})

            self.config = copy.deepcopy(config)
            
            # initialize trainer.
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
                #is_debug=config.get("is_debug", False),
                print_every=config.get("print_every", 100),
                seed=config.get("seed", 0),
                logger=config.get("logger", "wandb"),
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
                slurm=config.get("slurm", {}),
                no_energy=config.get("no_energy", False),
                is_debug=self.if_optuna,
                if_robust_loss=self.if_robust_loss,
                coef_rob=self.coef_rob,
                coef_cyc=self.coef_cyc,
                delta=self.delta,
            )

            # save config.
            if not self.if_optuna:
                with open(os.path.join(self.trainer.config["cmd"]["checkpoint_dir"], 'config.yml'), 'w') as yf:
                    yaml.dump(self.config, yf, default_flow_style=False)

            self.task = registry.get_task_class(config["mode"])(self.config)
            self.task.setup(self.trainer)
            start_time = time.time()
            self.task.run()
            distutils.synchronize()
            if distutils.is_master():
                logging.info(f"Total time taken: {time.time() - start_time}")
                #logging.info(f"Total time taken: {(time.time() - start_time)//60}")
        finally:
            if config['distributed']:
                distutils.cleanup()

        if self.config["task"].get("if_eval_fe", False):
            cPickle.dump(self.trainer.predictions_ef, open('predictions_ef.pickle', 'wb'))
            import numpy as np
            np.save('pred_f', self.trainer.predictions_ef['forces'])
            np.save('pred_e', self.trainer.predictions_ef['energy'])
        else:
            print('best test result...')
            print(self.trainer.test_metric)

        if self.if_optuna:
            return self.trainer.test_metric

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    setup_logging() 
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    if args.nequip:
        os.system(f'nequip-train {args.config_yml}')
    else:
        config = build_config(args, override_args)
        if args.submit:  # Run on cluster
            slurm_add_params = config.get(
                "slurm", None
            )  # additional slurm arguments
            if args.sweep_yml:  # Run grid search
                configs = create_grid(config, args.sweep_yml)
            else:
                configs = [config]

            logging.info(f"Submitting {len(configs)} jobs")
            executor = submitit.AutoExecutor(
                folder=args.logdir / "%j", slurm_max_num_timeout=3
            )
            executor.update_parameters(
                name=args.identifier,
                mem_gb=args.slurm_mem,
                timeout_min=args.slurm_timeout * 60,
                slurm_partition=args.slurm_partition,
                gpus_per_node=args.num_gpus,
                cpus_per_task=(config["optim"]["num_workers"] + 1),
                tasks_per_node=(args.num_gpus if args.distributed else 1),
                nodes=args.num_nodes,
                slurm_additional_parameters=slurm_add_params,
            )
            for config in configs:
                config["slurm"] = copy.deepcopy(executor.parameters)
                config["slurm"]["folder"] = str(executor.folder)
            jobs = executor.map_array(Runner(), configs)
            logging.info(
                f"Submitted jobs: {', '.join([job.job_id for job in jobs])}"
            )
            log_file = save_experiment_log(args, jobs, configs)
            logging.info(f"Experiment log saved to: {log_file}")

        else:  # Run locally
            Runner()(config)
