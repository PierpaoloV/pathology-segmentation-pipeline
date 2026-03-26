import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml

from ..async_tile_processor import async_tile_processor

# Registry of all architectures available in segmentation_models_pytorch.
# Add new entries here to make them available everywhere (training + inference).
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


class torch_processor(async_tile_processor):
    """
    Unified PyTorch inference processor.

    Handles both single-model and ensemble inference automatically:
      - model_path points to a .pt file  → single model
      - model_path points to a directory → ensemble (mean of all .pt files)

    The architecture is read from the accompanying YAML config. This means
    the same processor class is used for all architectures; no need to choose
    a different --custom_processor per model family.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.softmax_fn = nn.LogSoftmax(dim=1)

    def get_config_from_yaml(self, config_path: str):
        with open(file=config_path, mode='r') as f:
            parameters = yaml.load(stream=f, Loader=yaml.SafeLoader)
        return parameters['model'], parameters['sampler'], parameters['training']

    def _build_model(self, model_parameters: dict, n_classes: int):
        arch_name = model_parameters.get('modelname', 'unet')
        if arch_name not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{arch_name}' in YAML. "
                f"Available: {list(ARCHITECTURES.keys())}"
            )
        return ARCHITECTURES[arch_name](
            encoder_name=model_parameters['backbone'],
            classes=n_classes,
            encoder_weights=model_parameters['encoder_weights'],
        )

    def _load_single(self, pt_path: str):
        config_path = pt_path.split("_epoch")[0].split("_best")[0] + '.yaml'
        model_parameters, sampler_parameters, _ = self.get_config_from_yaml(config_path)
        n_classes = len(np.unique(list(sampler_parameters['training']['label_map'].values())))

        model = self._build_model(model_parameters, n_classes)
        state_dict = torch.load(pt_path, map_location=self._device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self._device)
        print(f"Loaded: {os.path.basename(pt_path)}  "
              f"arch={model_parameters.get('modelname', 'unet')}  "
              f"backbone={model_parameters['backbone']}")
        return model

    def _load_network_model(self):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.isdir(self._model_path):
            pt_files = sorted([f for f in os.listdir(self._model_path) if f.endswith('.pt')])
            if not pt_files:
                raise FileNotFoundError(f"No .pt files found in {self._model_path}")
            models = [self._load_single(os.path.join(self._model_path, f)) for f in pt_files]
            print(f"Ensemble of {len(models)} models ready.")
            return models
        else:
            return self._load_single(self._model_path)

    def _predict_tile_batch(self, tile_batch=None, info=None):
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)

        tile_batch = torch.from_numpy(tile_batch).to(self._device)

        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                if isinstance(self._model, list):
                    preds = [m.predict(tile_batch) for m in self._model]
                    result = torch.mean(torch.stack(preds), dim=0)
                else:
                    result = self._model.predict(tile_batch)

        result = self.softmax_fn(result)
        result = result.detach().cpu().numpy()

        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)

        return result

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info', '', 1))
