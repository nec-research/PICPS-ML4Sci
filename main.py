"""
*
*     SOFTWARE NAME
*
*        File:  main.py
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
