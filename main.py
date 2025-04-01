import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.configs import Config
from src.data_preprocessing.split_data import generate_splits, load_cases_split
from src.data_preprocessing.transformations import get_brats24_transform
from src.data_preprocessing.custom_dataset import BraTS24GliDataset

from src.utils.visualize import visualize_mri

if __name__ == "__main__":
    args = Config()
    # generate_splits(args)  # Only when no splits exist
    tr_cases, val_cases = load_cases_split(Path(args.split_folds_dir + '/split_fold_1.csv'))

    train_transformation = get_brats24_transform(args, 'train')
    valid_transformation = get_brats24_transform(args, 'valid')

    train_dataset = BraTS24GliDataset(
        data_dir=args.trainval_dir,
        mode='train',
        case_names=tr_cases,
        transforms=train_transformation
    )

    valid_dataset = BraTS24GliDataset(
        data_dir=args.trainval_dir,
        mode='valid',
        case_names=val_cases,
        transforms=valid_transformation
    )

    # image, seg, idx, name, affine = train_dataset[1]
    # visualize_mri(args, image, seg, name)

   