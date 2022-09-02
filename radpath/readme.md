# RadPath

Example images of the RadPath dataset. The image shows the inferior radiology-based multisequence MRI on the left and superior histopathology WSI on the right.

![](./img_radpath.png)

RadPath is a publicly available dataset of multimodal brain tumor images from the [RadPath 2020](https://miccai.westus2.cloudapp.azure.com/competitions/1) challenge and can be downloaded from [here](http://miccai2020-data.eastus.cloudapp.azure.com/). We use the training-validation-testing splits provided in the `./labels` folder.

---

## Radiology

### Requirements

Use the `env_rad.yml` conda environment for training the modality-specific classifier for radiology as follows:

```
cd radpath
conda env create -f env_rad.yml
conda activate env_rad
```

### Models

1) Train the baseline classifier of radiology as follows:

`python train.py -b 10 -e 600 -m t1 -f 1 --early_stopping --dropout 0.1 -x exp_t1`

2) To extract the latent representations, use the `latent_represenetations.ipynb` notebook

## Pathology

We use of the [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README_old.md) repository to train our WSI classifier. 

1) Patch extraction is done at 256 x 256 resolution using `create_patches.py` as follows:

`python create_patches.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch`

2) Feature extraction is done using `extract_features.py` as follows:

`python extract_features.py --data_dir DIR_TO_PATCHES --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512`

3) Model training is done using `main.py` as follows:

```python -u main.py --drop_out --early_stopping --lr 2e-4 --reg 1e-5 --seed 1 --opt adam --k 1 --split_dir /PATH/TO/SPLITS --exp_code exp_ID --weighted_sample --bag_loss ce --task radpath --model_type clam_sb --log_data --data_root_dir /PATH/TO/DATA --max_epochs 200 --model_size small --bag_weight 0.7 --B 25 --inst_loss ce```

For more details, please see the official [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README_old.md) repository.

4) To extract the latent representations, use the `latent_representations.ipynb` from the `path` folder. The `env_path.yml` file can be used for the conda environment.


## Guidance

1) Run `guidance_model.ipynb` to map from the latent space of inferior radiology to superior pathology modality.

2) Finally, run `guided_model.ipynb` for the proposed guided model, G(I)+I.

