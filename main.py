import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.configs import Config
from src.data_preprocessing.split_data import generate_splits, load_cases_split
from src.data_preprocessing.transformations import get_brats24_transform
# from .src.data_preprocessing.custom_dataset import BraTS24GliDataset

if __name__ == "__main__":
    args = Config()
    generate_splits(args)
    tr_cases, val_cases = load_cases_split(Path(args.split_folds_dir + '/split_fold_1.csv'))

    train_transformation = get_brats24_transform(args, 'train')
    valid_transformation = get_brats24_transform(args, 'valid')

    # train_dataset = BraTS24GliDataset(
    #     data_dir=args.trainval_dir,
    #     mode='train',
    #     case_names=tr_cases,
    #     transforms=train_transformation
    # )

    # valid_dataset = BraTS24GliDataset(
    #     data_dir=args.trainval_dir,
    #     mode='valid',
    #     case_names=val_cases,
    #     transforms=valid_transformation
    # )

    # # print(train_dataset[0])
    # image, seg, idx, name, affine = train_dataset[0]
    # print(image.shape)
    # print(seg.shape)
    # print(idx)
    # print(name)
    # print(affine)

    # # Define the slice indices for each view
    # sagittal_slice = image.shape[1] // 2
    # coronal_slice = image.shape[2] // 2
    # axial_slice = image.shape[3] // 2

    # # Plot the slices for each channel and the segmentation masks
    # fig, axes = plt.subplots(6, 3, figsize=(15, 30))

    # for i in range(4):
    #     # Extract the slices
    #     sagittal_view = image[i, sagittal_slice, :, :]
    #     coronal_view = image[i, :, coronal_slice, :]
    #     axial_view = image[i, :, :, axial_slice]

    #     # Plot the sagittal view
    #     axes[i, 0].imshow(sagittal_view, cmap='gray')
    #     axes[i, 0].imshow(seg[sagittal_slice, :, :], cmap='jet', alpha=0.5)
    #     axes[i, 0].set_title(f'Channel {i+1} - Sagittal View')

    #     # Plot the coronal view
    #     axes[i, 1].imshow(coronal_view, cmap='gray')
    #     axes[i, 1].imshow(seg[:, coronal_slice, :], cmap='jet', alpha=0.5)
    #     axes[i, 1].set_title(f'Channel {i+1} - Coronal View')

    #     # Plot the axial view
    #     axes[i, 2].imshow(axial_view, cmap='gray')
    #     axes[i, 2].imshow(seg[:, :, axial_slice], cmap='jet', alpha=0.5)
    #     axes[i, 2].set_title(f'Channel {i+1} - Axial View')

    # # Extract the segmentation slices
    # sagittal_seg = seg[:, sagittal_slice, :, :]
    # coronal_seg = seg[:, :, coronal_slice, :]
    # axial_seg = seg[:, :, :, axial_slice]

    # # Plot the segmentation masks
    # axes[4, 0].imshow(sagittal_seg)
    # axes[4, 0].set_title('Segmentation - Sagittal View')

    # axes[4, 1].imshow(coronal_seg)
    # axes[4, 1].set_title('Segmentation - Coronal View')

    # axes[4, 2].imshow(axial_seg)
    # axes[4, 2].set_title('Segmentation - Axial View')

    # # Plot the grayscale segmentation masks
    # axes[5, 0].imshow(sagittal_seg, cmap='gray')
    # axes[5, 0].set_title('Segmentation Gray - Sagittal View')

    # axes[5, 1].imshow(coronal_seg, cmap='gray')
    # axes[5, 1].set_title('Segmentation Gray - Coronal View')

    # axes[5, 2].imshow(axial_seg, cmap='gray')
    # axes[5, 2].set_title('Segmentation Gray - Axial View')

    # plt.tight_layout()
    # plt.savefig('test.png') 

