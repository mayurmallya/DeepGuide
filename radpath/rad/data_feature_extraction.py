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
    def __init__(self, csv_file, data_path, data_transform, modality, label_dict = {'G':0, 'O':1, 'A':2}, shuffle=False): 
        self.transform = data_transform
        self.image_path = data_path
        self.df = pd.read_csv(csv_file)
        self.label_dict = label_dict
        self.modality = modality

        if shuffle:
            raise AssertionError
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
    
    
    val_transform = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Transposed(keys="image", indices=[0,3,1,2]),
            #Resized(keys=["image"], spatial_size=input_size),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )

    test_transform = val_transform
    train_transform = val_transform # since this is feature extraction, we don't want random augmentations
    

    label_dict = {'G':0, 'O':1, 'A':2}

    train = RadIter(csv_file=os.path.join(label_root, 'split_{}_train.csv'.format(fold)),
                      data_path = data_root,
                      label_dict = label_dict,
                      data_transform = train_transform,
                      modality = modality,
                      shuffle = False) # changing from True

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



def data_create_feature_extraction(data_root, label_root, batch_size, fold, modality, num_workers=8, **kwargs):

    train, val, test = get_data(data_root, label_root, fold = fold, modality=modality, **kwargs)

    assert batch_size == 1
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return (train_loader, val_loader, test_loader)
