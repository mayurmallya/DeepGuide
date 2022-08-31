import os
import torch
import pickle
import pandas as pd
import numpy as np
import monai
from monai.transforms import (
    LoadImaged,
    Compose,
    AsChannelFirstd,
    AddChanneld,
    Orientationd,
    RandFlipd,
    NormalizeIntensityd,
    ScaleIntensityd,
    RandRotate90d,
    Resized,
    ToTensord,
    MapTransform,
    RandScaleIntensityd,
    RandShiftIntensityd,
    apply_transform,
    CropForegroundd,
    RandRotated,
    Transpose,
    RandSpatialCropd,
    RandZoomd,
    CenterSpatialCropd,
)
from monai.transforms.utils import rescale_array, rescale_instance_array
from torch.utils.data import WeightedRandomSampler



from typing import Optional, Sequence, Mapping, Hashable, Any, Dict
from monai.config import KeysCollection
from monai.transforms.inverse import InvertibleTransform

class Transposed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Transpose`.
    """

    def __init__(
        self, keys: KeysCollection, indices: Optional[Sequence[int]], allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = Transpose(indices)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
            # if None was supplied then numpy uses range(a.ndim)[::-1]
            indices = self.transform.indices or range(d[key].ndim)[::-1]
            self.push_transform(d, key, extra_info={"indices": indices})
        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            fwd_indices = np.array(transform[InverseKeys.EXTRA_INFO]["indices"])
            inv_indices = np.argsort(fwd_indices)
            inverse_transform = Transpose(inv_indices)
            # Apply inverse
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d



class RadIter(monai.data.CacheDataset):
    def __init__(self, csv_file, data_path, data_transform, modality, label_dict = {'G':0, 'O':1, 'A':2}, shuffle=True):
        self.transform = data_transform
        self.image_path = data_path
        self.df = pd.read_csv(csv_file)
        self.label_dict = label_dict
        self.modality = modality
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def  __getitem__(self, idx):
        label = self.label_dict[self.df.loc[idx, 'class']]
        radpath_ID = self.df.loc[idx, 'CPM_RadPath_2019_ID']

        T1    = os.path.join(self.image_path, radpath_ID, radpath_ID+'_t1.nii.gz') # (240, 240, 155)
        T1c   = os.path.join(self.image_path, radpath_ID, radpath_ID+'_t1ce.nii.gz')
        T2    = os.path.join(self.image_path, radpath_ID, radpath_ID+'_t2.nii.gz')
        FLAIR = os.path.join(self.image_path, radpath_ID, radpath_ID+'_flair.nii.gz')

        if self.modality == 'all':
            image = [T1, T1c, T2, FLAIR]
        elif self.modality == 't1':
            image = [T1]
        elif self.modality == 't2':
            image = [T2]
        elif self.modality == 't1ce':
            image = [T1c]
        elif self.modality == 'flair':
            image = [FLAIR]

        data = {'image': image, 'gt':label, 'radpath_ID':radpath_ID}

        if self.transform is not None:
            data = apply_transform(self.transform, data)
        return data



def get_data(data_root, label_root, fold, modality, **kwargs):
    print("DataIter:: fold = {}".format(fold))

    if modality == 'all':
        print('\n Transform AddChanneld not needed in this case')
        raise NotImplementedError
    
    train_transform = Compose(
        [   
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Transposed(keys="image", indices=[0,3,1,2]),
            #Resized(keys=["image"], spatial_size=input_size),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True), # Normalize
            
            # Overfit
            RandFlipd(keys=["image"], prob=0.25, spatial_axis=0), # Mirroring axis0
            RandFlipd(keys=["image"], prob=0.25, spatial_axis=1), # Mirroring axis1
            RandFlipd(keys=["image"], prob=0.25, spatial_axis=2), # Mirroring axis2
            RandRotated(keys=["image"], range_x=0.25, range_y=0.25, range_z=0.25, prob=0.5, keep_size=True), # Rotation
            RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9, max_zoom=1.1, keep_size=True), # Scaling
            RandSpatialCropd(keys=["image"], roi_size=[140,220,220], random_center=False, random_size=False), # Cropping
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5), # Color augmentation


            ToTensord(keys=["image"]),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Transposed(keys="image", indices=[0,3,1,2]),
            #Resized(keys=["image"], spatial_size=input_size),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            
            #CenterSpatialCropd(keys=["image"], roi_size=[140,220,220]),
            #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=["image"]),
        ]
    )

    test_transform = val_transform

    label_dict = {'G':0, 'O':1, 'A':2}

    train = RadIter(csv_file=os.path.join(label_root, 'split_{}_train.csv'.format(fold)),
                      data_path = data_root,
                      label_dict = label_dict,
                      data_transform = train_transform,
                      modality = modality,
                      shuffle = True)

    val   = RadIter(csv_file=os.path.join(label_root, 'split_{}_val.csv'.format(fold)),
                      data_path = data_root,
                      label_dict = label_dict,
                      data_transform = val_transform,
                      modality = modality,
                      shuffle = False)

    test = RadIter(csv_file=os.path.join(label_root, 'split_{}_test.csv'.format(fold)),
                      data_path = data_root,
                      label_dict = label_dict,
                      data_transform = test_transform,
                      modality = modality,
                      shuffle = False)

    return train, val, test



def data_create(data_root, label_root, batch_size, fold, modality, num_workers=8, **kwargs):

    train, val, test = get_data(data_root, label_root, fold = fold, modality=modality, **kwargs)

    # weighted random sampler
    label_dict = {'G':0, 'O':1, 'A':2}
    csv_file = os.path.join(label_root, 'split_{}_train.csv'.format(fold))
    df = pd.read_csv(csv_file, usecols=['class'])
    y_train = df['class']
    labels = [label_dict[t] for t in y_train]
    labels = np.array(labels)
    class_sample_count = np.array([len(np.where(labels==t)[0]) for t in np.unique(labels)]) 
    class_sample_probabilities = 1./class_sample_count
    sample_probabilities = np.array([class_sample_probabilities[t] for t in labels])
    sample_probabilities = torch.from_numpy(sample_probabilities)
    sampler = WeightedRandomSampler(weights = sample_probabilities.type('torch.DoubleTensor'), num_samples = len(sample_probabilities), replacement = True)


    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler = sampler)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    

    return (train_loader, val_loader, test_loader)
