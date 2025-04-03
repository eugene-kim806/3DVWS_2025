import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.configs import Config

from src.data_preprocessing.split_data import generate_splits, load_cases_split
from src.data_preprocessing.transformations import get_brats24_transform
from src.data_preprocessing.custom_dataset import BraTS24GliDataset

from src.model.unet3D import Custom3DUNet

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet3D_model = Custom3DUNet(args=args).to(device)

    ## Debug Test
    # dummy_input = torch.randn(2, 4, 96, 96, 96).to(device)
    # output = unet3D_model(dummy_input).argmax(dim=1)
    # cpu_output = np.array(output.cpu().detach().numpy())
    # print("Input shape: ", dummy_input.shape)
    # print("Output shape: ", output.shape)
    # print("Unique mask: ", np.unique(cpu_output))

    

   