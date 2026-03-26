import torch_data_generator as data_generator

import time
import gc
import os
import sys
import shutil
import argparse

import numpy as np
import yaml
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import wandb

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

# Registry of all architectures available in segmentation_models_pytorch.
# Keep in sync with the same dict in torch_processor.py.
ARCHITECTURES = {
    'unet':        smp.Unet,
    'unet-plus':   smp.UnetPlusPlus,
    'manet':       smp.MAnet,
    'linknet':     smp.Linknet,
    'fpn':         smp.FPN,
    'pspnet':      smp.PSPNet,
    'deeplabv3':   smp.DeepLabV3,
    'deeplabv3+':  smp.DeepLabV3Plus,
    'pan':         smp.PAN,
}


def ask_architecture(default: str = 'unet') -> str:
    """
    Interactively prompt the user to select a segmentation architecture.
    The YAML's current value is shown as the default; press Enter to keep it.
    """
    names = list(ARCHITECTURES.keys())
    print("\nAvailable architectures:")
    for i, name in enumerate(names, 1):
        tag = "  ← current" if name == default else ""
        print(f"  [{i}] {name}{tag}")

    while True:
        choice = input(f"\nSelect architecture [Enter to keep '{default}']: ").strip()
        if choice == '':
            return default
        if choice in ARCHITECTURES:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print(f"  Invalid input. Enter a number 1–{len(names)} or an architecture name.")


def collect_arguments():
    """
    Collect command line arguments.
    """
    argument_parser = argparse.ArgumentParser(description='Torch train run.')
    argument_parser.add_argument('-p', '--project_name', required=False, type=str, default="model", help='project name in wandb')
    argument_parser.add_argument('-d', '--data_path', required=True, type=str, help='input data file')
    argument_parser.add_argument('-c', '--config_path', required=True, type=str, help='input')
    argument_parser.add_argument('-a', '--alb_config_path', required=False, type=str, default="", help='albumentations')
    argument_parser.add_argument('-o', '--output_path', required=True, type=str, help='output')
    arguments = vars(argument_parser.parse_args())

    return arguments["project_name"], arguments["data_path"], arguments["config_path"], \
           arguments["alb_config_path"], arguments["output_path"]


def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA out of memory." in exception.args[0]
    )


def is_cudnn_snafu(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def is_out_of_cpu_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_model(model_param: dict, training_param: dict, num_classes: int):
    print("Init model...")

    arch_name = model_param['modelname']
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch_name}'. Available: {list(ARCHITECTURES.keys())}")
    model = ARCHITECTURES[arch_name](
        encoder_name=model_param['backbone'],
        classes=num_classes,
        encoder_weights=model_param['encoder_weights'],
    )

    if model_param['loss'] == 'cc':
        loss_fn = nn.NLLLoss(ignore_index=-100)
    elif model_param['loss'] == 'lovasz':
        loss_fn = smp.losses.LovaszLoss(mode='multiclass', ignore_index=-100)
    elif model_param['loss'] == 'dice':
        loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=-100)
    else:
        raise ValueError(f"Unknown loss '{model_param['loss']}'. Choose from: cc, lovasz, dice")

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=model_param['learning_rate'])])

    if model_param['learning_rate_schedule'] == 'plateau':
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=training_param['lr_reduction_factor'],
                                                               patience=training_param['lr_plateau'])

    elif model_param['learning_rate_schedule'] == 'one_cycle':
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=0.001,
                                                        steps_per_epoch=1,
                                                        epochs=training_param['epochs'],
                                                        final_div_factor=1e6)

    return model, loss_fn, optimizer, schedular


def get_metrics(predictions: torch.Tensor,
                targets: torch.Tensor,
                mode: str = 'multiclass',
                threshold: float = 0.5,
                num_classes=2) -> dict:
    """Create metrics to monitor network performance."""
    tp, fp, fn, tn = smp.metrics.get_stats(predictions.argmax(dim=1),
                                           targets,
                                           mode=mode,
                                           ignore_index=-100,
                                           num_classes=num_classes)

    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").detach().cpu().numpy()
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").detach().cpu().numpy()
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro").detach().cpu().numpy()
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").detach().cpu().numpy()
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise").detach().cpu().numpy()

    return iou_score, f1_score, f2_score, accuracy, recall


def train_loop(data_gen, model, loss_fn, scaler, optimizer, device, use_amp, num_classes, progress, task_id):

    epoch_loss, epoch_iou, epoch_f1, epoch_f2, epoch_acc, epoch_recall = [], [], [], [], [], []
    softmax_fn = nn.LogSoftmax(dim=1)

    for tbatch_idx, (inputs, targets) in enumerate(data_gen):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            predictions = model.forward(inputs)
            if isinstance(loss_fn, torch.nn.NLLLoss):
                loss = loss_fn(softmax_fn(predictions), targets.squeeze())
            else:
                loss = loss_fn(predictions, targets.squeeze())

        if not torch.isfinite(loss):
            console.print(f'[yellow]Warning: non-finite loss ({loss.item():.4f}) at batch {tbatch_idx}, skipping.[/yellow]')
            optimizer.zero_grad()
            progress.update(task_id, advance=1)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        iou_score, f1_score, f2_score, accuracy, recall = get_metrics(softmax_fn(predictions),
                                                                       targets.squeeze(),
                                                                       num_classes=num_classes)
        epoch_loss.append(loss.detach().cpu().numpy())
        epoch_iou.append(iou_score)
        epoch_f1.append(f1_score)
        epoch_f2.append(f2_score)
        epoch_acc.append(accuracy)
        epoch_recall.append(recall)

        progress.update(task_id, advance=1,
                        loss=float(np.mean(epoch_loss)),
                        iou=float(np.mean(epoch_iou)))

    n = tbatch_idx + 1
    return {'training_loss':     np.sum(epoch_loss)   / n,
            'training_iou_score': np.sum(epoch_iou)   / n,
            'training_f1_score':  np.sum(epoch_f1)    / n,
            'training_f2_score':  np.sum(epoch_f2)    / n,
            'training_accuracy':  np.sum(epoch_acc)   / n,
            'training_recall':    np.sum(epoch_recall) / n}


def validation_loop(data_gen, model, loss_fn, device, use_amp, num_classes, progress, task_id):

    epoch_loss, epoch_iou, epoch_f1, epoch_f2, epoch_acc, epoch_recall = [], [], [], [], [], []
    softmax_fn = nn.LogSoftmax(dim=1)

    for tbatch_idx, (inputs, targets) in enumerate(data_gen):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                predictions = model.forward(inputs)
                if isinstance(loss_fn, torch.nn.NLLLoss):
                    loss = loss_fn(softmax_fn(predictions), targets.squeeze())
                else:
                    loss = loss_fn(predictions, targets.squeeze())

            iou_score, f1_score, f2_score, accuracy, recall = get_metrics(softmax_fn(predictions),
                                                                           targets.squeeze(),
                                                                           num_classes=num_classes)
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_iou.append(iou_score)
            epoch_f1.append(f1_score)
            epoch_f2.append(f2_score)
            epoch_acc.append(accuracy)
            epoch_recall.append(recall)

            progress.update(task_id, advance=1,
                            loss=float(np.mean(epoch_loss)),
                            iou=float(np.mean(epoch_iou)))

    n = tbatch_idx + 1
    return {'validation_loss':     np.sum(epoch_loss)   / n,
            'validation_iou_score': np.sum(epoch_iou)   / n,
            'validation_f1_score':  np.sum(epoch_f1)    / n,
            'validation_f2_score':  np.sum(epoch_f2)    / n,
            'validation_accuracy':  np.sum(epoch_acc)   / n,
            'validation_recall':    np.sum(epoch_recall) / n}


def save_model(model, output_path: str):
    torch.save(model.state_dict(), output_path)


def get_config_from_yaml(config_path: str) -> dict:
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.load(stream=param_file, Loader=yaml.SafeLoader)
    return parameters['model'], parameters['sampler'], parameters['training']


def main():
    project_name, data_path, config_path, albumentations_path, output_path = collect_arguments()

    model_parameters, sampler_parameters, training_parameters = get_config_from_yaml(config_path)

    # Let the user confirm or change the architecture before training starts.
    current_arch = model_parameters.get('modelname', 'unet')
    chosen_arch = ask_architecture(default=current_arch)
    model_parameters['modelname'] = chosen_arch

    wandb.login()
    logger = wandb.init(project=project_name)
    shutil.copyfile(config_path, os.path.join(output_path, project_name + '_' + logger.name + '.yaml'))

    print("Init data loaders...")
    datawrapper = data_generator.PtDataLoader(data_path,
                                              albumentations_path,
                                              sampler_parameters,
                                              training_parameters)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    use_amp = training_parameters['mixed_precision']
    num_classes = len(np.unique(list(sampler_parameters['training']['label_map'].values())))

    model, loss_fn, optimizer, schedular = init_model(model_parameters,
                                                      training_parameters,
                                                      num_classes)

    wandb.config = {"learning_rate": model_parameters['learning_rate'],
                    "backbone": model_parameters["backbone"],
                    "encoder_weights": model_parameters["encoder_weights"],
                    "epochs": training_parameters['epochs'],
                    "training_batch_size": training_parameters['training_batch_size']}
    wandb.watch(model)

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    model.to(device)

    # ── Startup panel ────────────────────────────────────────────────────────
    config_text = (
        f"[bold]Project:[/bold] {project_name}  |  "
        f"[bold]Run:[/bold] {logger.name}\n"
        f"[bold]Arch:[/bold] {model_parameters['modelname']} "
        f"({model_parameters['backbone']})  |  "
        f"[bold]Loss:[/bold] {model_parameters['loss']}  |  "
        f"[bold]LR:[/bold] {model_parameters['learning_rate']}  |  "
        f"[bold]Epochs:[/bold] {training_parameters['epochs']}  |  "
        f"[bold]Early-stop patience:[/bold] {training_parameters['stop_plateau']}"
    )
    console.print(Panel(config_text, title="[bold cyan]Training started[/bold cyan]", expand=False))

    _progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold]{task.description:<12}[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("loss=[cyan]{task.fields[loss]:.4f}[/cyan]  iou=[green]{task.fields[iou]:.4f}[/green]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    best_val_loss = np.inf
    no_improve_counter = 0
    save_paths_list = []
    total_epochs = training_parameters['epochs']

    for epoch_indx in range(total_epochs):
        start_time = time.time()
        model.train()

        with Progress(*_progress_columns, console=console, transient=True) as progress:
            train_task = progress.add_task(
                "Training", total=len(datawrapper.training_generator), loss=0.0, iou=0.0)
            train_results = train_loop(datawrapper.training_generator,
                                       model, loss_fn, scaler, optimizer,
                                       device, use_amp, num_classes,
                                       progress, train_task)

            model.eval()
            val_task = progress.add_task(
                "Validating", total=len(datawrapper.validation_generator), loss=0.0, iou=0.0)
            validation_results = validation_loop(datawrapper.validation_generator,
                                                 model, loss_fn,
                                                 device, use_amp, num_classes,
                                                 progress, val_task)

        if model_parameters['learning_rate_schedule'] == 'plateau':
            schedular.step(validation_results['validation_loss'])
        elif model_parameters['learning_rate_schedule'] == 'one_cycle':
            schedular.step()

        update_dict = {**train_results, **validation_results}
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        update_dict['learning_rate'] = current_lr
        wandb.log(update_dict)

        # Keep only the last 10 epoch checkpoints
        if len(save_paths_list) > 10:
            os.remove(save_paths_list[0])
            save_paths_list.pop(0)

        epoch_path = os.path.join(output_path, project_name + '_{}_epoch_{}.pt'.format(logger.name, epoch_indx + 1))
        save_paths_list.append(epoch_path)
        save_model(model, epoch_path)

        is_best = update_dict['validation_loss'] < best_val_loss
        if is_best:
            best_val_loss = update_dict['validation_loss']
            no_improve_counter = 0
            save_model(model, os.path.join(output_path, project_name + '_{}_best_model.pt'.format(logger.name)))
        else:
            no_improve_counter += 1

        # ── Epoch summary line ────────────────────────────────────────────
        elapsed = time.time() - start_time
        time_str = f'{elapsed / 60:.1f}m' if elapsed >= 60 else f'{elapsed:.0f}s'

        status = Text('✓ new best', style='bold green') if is_best else \
                 Text(f'no improve {no_improve_counter}/{training_parameters["stop_plateau"]}', style='dim')

        summary = (
            f"[bold]Epoch {epoch_indx + 1:>3}/{total_epochs}[/bold]  "
            f"train_loss=[cyan]{train_results['training_loss']:.4f}[/cyan]  "
            f"val_loss=[cyan]{validation_results['validation_loss']:.4f}[/cyan]  "
            f"val_iou=[green]{validation_results['validation_iou_score']:.4f}[/green]  "
            f"lr={current_lr:.2e}  "
            f"[{time_str}]  "
        )
        console.print(summary, status)

        if no_improve_counter >= training_parameters['stop_plateau']:
            console.print(Panel(
                f"No improvement for [bold]{no_improve_counter}[/bold] epochs. "
                f"Best val loss: [cyan]{best_val_loss:.4f}[/cyan]",
                title="[bold yellow]Early stopping[/bold yellow]", expand=False))
            break

        datawrapper.on_epoch_end()

    console.print(Panel(
        f"Training complete.  Best val loss: [cyan]{best_val_loss:.4f}[/cyan]  "
        f"after [bold]{epoch_indx + 1}[/bold] epochs.",
        title="[bold green]Done[/bold green]", expand=False))

    wandb.finish()


if __name__ == '__main__':
    sys.exit(main())
