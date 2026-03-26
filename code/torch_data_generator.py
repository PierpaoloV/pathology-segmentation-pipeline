import digitalpathology.generator.batch.simplesampler as sampler

import torch 
import os
import numpy as np 
import skimage.color
import albumentations 
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.composition import Compose
from functools import partial 

class HED_normalize(ImageOnlyTransform):
    def __init__(self, sigmas, biases, **params):
        super(HED_normalize, self).__init__(**params)
        self.biases = biases
        self.sigmas = sigmas

    def color_norm_hed(self, img, sigmas, biases):
        patch = img.astype(np.float32) / 255.0
        patch_hed = skimage.color.rgb2hed(rgb=patch)

        # Augment the Haematoxylin channel.
        patch_hed[:, :, 0] *= 1.0 + sigmas[0]
        patch_hed[:, :, 0] += biases[0]

        # Augment the Eosin channel.
        patch_hed[:, :, 1] *= 1.0 + sigmas[1]
        patch_hed[:, :, 1] += biases[1]

        patch_hed[:, :, 2] *= 1.0 + sigmas[2]
        patch_hed[:, :, 2] += biases[2]

        # Convert back to RGB color coding.
        patch_rgb = skimage.color.hed2rgb(hed=patch_hed)
        patch_transformed = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
        return np.asarray(patch_transformed * 255, dtype=np.uint8)

    def apply(self, img, sigmas, biases, **params):
        return self.color_norm_hed(img, sigmas, biases)

    def get_params_dependent_on_targets(self, params):
        sigmas = [np.random.uniform(-self.sigmas[i], self.sigmas[i]) for i in range(len(self.sigmas))]
        biases = [np.random.uniform(-self.biases[i], self.biases[i]) for i in range(len(self.biases))]
        return {'sigmas': sigmas, 'biases': biases}

    def get_transform_init_args_names(self):
        return ("sigmas", "biases")

    @property
    def targets_as_params(self):
        return ["image"]

class PTDataGenerator(torch.utils.data.Dataset):
    """_summary_

    Args:
        torch (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # def __init__(self, data_gen: sampler.SimpleSampler, augmentations_pipeline=None: Compose, batch_size=1: int, num_classes=2: int):
    def __init__(self, 
                 data_gen,   
                 augmentations_pipeline=None, 
                 batch_size=1, 
                 num_classes=2,
                 reset_samplers=True):
        """
        

        Args:
            data_gen (sampler.SimpleSampler): _description_
            augmentations_pipeline (_type_, optional): _description_. Defaults to None:Compose.
            batch_size (_type_, optional): _description_. Defaults to 1:int.
            num_classes (_type_, optional): _description_. Defaults to 2:int.

        Returns:
            _type_: _description_
        """
        
        self._data_gen = data_gen
        self._num_classes = num_classes
        if augmentations_pipeline: 
            self._aug_fn = partial(self.augment_fn, transform=augmentations_pipeline)
        else: 
            self._aug_fn = None
            
        self._batch_size = batch_size
        self._counter = 0
        self._reset = reset_samplers

    def __getitem__(self, index):
        batch, masks = self._preprocess_batch(index)
        return batch, masks
    
    def __len__(self):
        return self._data_gen._iterations // self._batch_size
    
    def on_epoch_end(self):
        if self._reset:
            self._data_gen.step()
    
    def augment_fn(self, patch, mask, weight, transform):
        transformed = transform(image=patch, masks=[mask, weight])
        return transformed["image"], transformed["masks"][0], transformed["masks"][1]    
    
    def _preprocess_batch(self, index):
        patch, mask, weight = self._data_gen[index]
        if self._aug_fn:
            patch, mask, weight = self._aug_fn(patch, mask, weight)

        patch = patch.astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).type(torch.float32)
        mask = torch.from_numpy(mask).type(torch.int64)
        mask[weight == 0] = -100
        patch = patch.permute(2, 0, 1)
        return patch, mask[None]

class PtDataLoader(object):
    def __init__(self, 
                 data_path: str,
                 albumentations_path: str, 
                 sampler_param: dict, 
                 training_param: dict):
         
        
        self.data_path = data_path
        self.sampler_param = sampler_param
        self.training_param= training_param
        if os.path.isfile(albumentations_path):
            self.transforms = albumentations.load(albumentations_path, data_format='yaml')
        else:
            self.transforms = None

        if self.transforms: 
            print("Will apply augmentations...")
        self.num_classes = len(np.unique(list(sampler_param['training']['label_map'].values())))
        
        self.training_sampler = None 
        self.training_set = None 
        self.validation_sampler = None 
        self.validation_set = None
        
        self.init_simple_sampler()

        print("{} images in the training sampler".format(len(self.training_sampler._patch_samplers.items())))
        print("{} images in the validation sampler".format(len(self.validation_sampler._patch_samplers.items())))

        print(f"Starting batch generators with a batch-size of {self.training_param['training_batch_size']} for training and {self.training_param['validation_batch_size']} for validation")
        
        self.training_set = PTDataGenerator(self.training_sampler, 
                                            augmentations_pipeline=self.transforms, 
                                            batch_size=1,
                                            num_classes=self.num_classes)
        
        self.validation_set = PTDataGenerator(self.validation_sampler,
                                              augmentations_pipeline=None, 
                                              batch_size=1,
                                              num_classes=self.num_classes, 
                                              reset_samplers=False)
        
        self.training_generator = torch.utils.data.DataLoader(self.training_set,
                                                              batch_size=self.training_param['training_batch_size'],
                                                              num_workers=self.training_param['workers'],
                                                              prefetch_factor=4,
                                                              pin_memory=True,
                                                              persistent_workers=True)

        self.validation_generator = torch.utils.data.DataLoader(self.validation_set,
                                                                batch_size=self.training_param['validation_batch_size'],
                                                                num_workers=self.training_param['workers'],
                                                                prefetch_factor=4,
                                                                pin_memory=True,
                                                                persistent_workers=True)

    
    def init_simple_sampler(self):

        self.training_sampler = sampler.SimpleSampler(patch_source_filepath=self.data_path,
                                                      **self.sampler_param['training'], 
                                                      partition='training')
        
        self.validation_sampler = sampler.SimpleSampler(patch_source_filepath=self.data_path,
                                                        **self.sampler_param['validation'],
                                                        partition='validation')

        self.training_sampler.step()
        self.validation_sampler.step()
        
    def on_epoch_end(self):
        """
        On the end over every epoch we want to reset the indicis, so we have other patches.
        """
        self.training_set.on_epoch_end()
        self.validation_set.on_epoch_end()