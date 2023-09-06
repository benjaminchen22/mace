###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import tensor_dict_to_device, to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    device: torch.device,
    log_errors: str,
    rank: int,
    global_rank: int,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
):
    lowest_loss = np.inf
    patience_counter = 0
    swa_start = True

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("Started training")
    for epoch in range(start_epoch, max_num_epochs):

        # Required for shuffling data in DistributedDataParallel
        sampler = train_loader.sampler
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(epoch)

        # Train
        for batch in train_loader:
            _, opt_metrics = take_step(
                model=model,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            logger.log(opt_metrics)

        # Validate
        if epoch % eval_interval == 0:
            if ema is not None:
                with ema.average_parameters():
                    valid_loss, eval_metrics = evaluate(
                        model=model,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        device=device,
                    )
            else:
                valid_loss, eval_metrics = evaluate(
                    model=model,
                    loss_fn=loss_fn,
                    data_loader=valid_loader,
                    device=device,
                )
            eval_metrics["mode"] = "eval"
            eval_metrics["epoch"] = epoch
            logger.log(eval_metrics)
            lr = lr_scheduler.optimizer.param_groups[0]['lr']

            if log_errors == "PerAtomRMSE":
                error_e = eval_metrics["rmse_e_per_atom"] * 1e3
                error_f = eval_metrics["rmse_f"] * 1e3
                logging.info(
                        f"GPU {global_rank} | Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, lr={lr:.2e}"
                )
            elif log_errors == "TotalRMSE":
                error_e = eval_metrics["rmse_e"] * 1e3
                error_f = eval_metrics["rmse_f"] * 1e3
                logging.info(
                        f"GPU {global_rank} | Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, lr={lr:.2e}"
                )
            elif log_errors == "PerAtomMAE":
                error_e = eval_metrics["mae_e_per_atom"] * 1e3
                error_f = eval_metrics["mae_f"] * 1e3
                logging.info(
                        f"GPU {global_rank} | Epoch {epoch}: loss={valid_loss:.4f}, MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A, lr={lr:.2e}"
                )
            elif log_errors == "TotalMAE":
                error_e = eval_metrics["mae_e"] * 1e3
                error_f = eval_metrics["mae_f"] * 1e3
                logging.info(
                        f"GPU {global_rank} | Epoch {epoch}: loss={valid_loss:.4f}, MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A, lr={lr:.2e}"
                )

            if valid_loss >= lowest_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(
                        f"Stopping optimization after {patience_counter} epochs without improvement"
                    )
                    break
            else:
                lowest_loss = valid_loss
                patience_counter = 0
                if global_rank == 0 or global_rank is None: # global_rank is None for CPU

                    if global_rank == 0:
                        m = model.module
                    else:
                        m = model

                    # Save model.module isntead of model, as model is
                    # DistributedDataParallel
                    if ema is not None:
                        with ema.average_parameters():
                            checkpoint_handler.save(
                                state=CheckpointState(m, optimizer, lr_scheduler),
                                epochs=epoch,
                            )
                    else:
                        checkpoint_handler.save(
                            state=CheckpointState(m, optimizer, lr_scheduler),
                            epochs=epoch,
                        )
                        m.to('cpu')
                        torch.save(m)

        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR):
                lr_scheduler.step()
            else:
                lr_scheduler.step(valid_loss)  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on SWA")
                swa_start = False
                

                ##################################################################
                # BC: Experimental; set LR to a low value that will be annealed to the 
                # SWA LR so that when we change loss function there will be no jump in the 
                # losses
                optim = lr_scheduler.optimizer
                swa_optim = swa.scheduler.optimizer
                
                for g, swa_g in zip(optim.param_groups, swa_optim.param_groups):
                    logging.info(f'Setting lr to {swa_g["swa_lr"] / 100}')
                    g['lr'] =  swa_g['swa_lr'] / 100
                ###################################################################    
            loss_fn = swa.loss_fn
            # model.module or model gives same results
            swa.model.update_parameters(model)
            swa.scheduler.step()

    logging.info("Training complete")


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:

    start_time = time.time()
    batch = batch.to(device)
    model = model.to(device)
    optimizer.zero_grad()
    output = model(batch, training=True)
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    total_loss = 0.0
    delta_es_list = []
    delta_es_per_atom_list = []
    delta_fs_list = []
    fs_list = []

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        model = model.to(device)
        output = model(batch, training=False)
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device("cpu"))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()

        delta_es_list.append(batch.energy - output["energy"])
        delta_es_per_atom_list.append(
            (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
        )
        delta_fs_list.append(batch.forces - output["forces"])
        fs_list.append(batch.forces)

    avg_loss = total_loss / len(data_loader)

    delta_es = to_numpy(torch.cat(delta_es_list, dim=0))
    delta_es_per_atom = to_numpy(torch.cat(delta_es_per_atom_list, dim=0))
    delta_fs = to_numpy(torch.cat(delta_fs_list, dim=0))
    fs = to_numpy(torch.cat(fs_list, dim=0))

    aux = {
        "loss": avg_loss,
        # Mean absolute error
        "mae_e": compute_mae(delta_es),
        "mae_e_per_atom": compute_mae(delta_es_per_atom),
        "mae_f": compute_mae(delta_fs),
        "rel_mae_f": compute_rel_mae(delta_fs, fs),
        # Root-mean-square error
        "rmse_e": compute_rmse(delta_es),
        "rmse_e_per_atom": compute_rmse(delta_es_per_atom),
        "rmse_f": compute_rmse(delta_fs),
        "rel_rmse_f": compute_rel_rmse(delta_fs, fs),
        # Q_95
        "q95_e": compute_q95(delta_es),
        "q95_f": compute_q95(delta_fs),
        # Time
        "time": time.time() - start_time,
    }

    return avg_loss, aux
