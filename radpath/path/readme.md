# Pathology

We use of the [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README_old.md) repository to train our WSI classifier. 

1) Patch extraction is done at 256 x 256 resolution using `create_patches.py`.

`python create_patches.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch`

2) Feature extraction is done using `extract_features.py`

`python extract_features.py --data_dir DIR_TO_PATCHES --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512`

3) Model training is done using `main.py`

```python -u main.py --drop_out --early_stopping --lr 2e-4 --reg 1e-5 --seed 1 --opt adam --k 1 --split_dir /PATH/TO/SPLITS --exp_code exp_ID --weighted_sample --bag_loss ce --task radpath --model_type clam_sb --log_data --data_root_dir /PATH/TO/DATA --max_epochs 200 --model_size small --bag_weight 0.7 --B 25 --inst_loss ce```
