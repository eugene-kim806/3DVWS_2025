from dataclasses import dataclass

@dataclass
class Config:
    home_dir: str = '/home/ek3dw25/3d-glioma-segmentation/BraTS2024_GLI_post_treatment'
    verbose: bool = True
    num_workers: int = 6

    # Data
    dataset: str = 'BraTS2024-GLI'
    data_dir: str = home_dir + '/data'
    raw_data_dir: str = data_dir + '/raw'
    trainval_dir: str = raw_data_dir + '/BraTS2024-BraTS-GLI-TrainingData'
    test_dir: str = raw_data_dir + '/BraTS2024-BraTS-GLI-ValidatingData'
    split_folds_dir: str = data_dir + '/split_folds'
    num_folds: int = 5

    # Data Augmentation
    patch_size: int = 96
    pos_ratio: float = 2.0
    neg_ratio: float = 0.5

    # Save
    example_dir: str = '/home/ek3dw25/3d-glioma-segmentation/BraTS2024_GLI_post_treatment/examples'
