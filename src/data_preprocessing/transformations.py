import numpy as np
import monai.transforms as transforms
from monai.transforms.transform import MapTransform

class RobustZScoreNormalization(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key] > 0

            lower = np.percentile(d[key][mask], 0.2)
            upper = np.percentile(d[key][mask], 99.8)

            d[key][mask & (d[key] < lower)] = lower
            d[key][mask & (d[key] > upper)] = upper

            y = d[key][mask]
            d[key] -= y.mean()
            d[key] /= y.std()

        return d


def get_brats24_transform(args, mode:str):
    """
    Returns the transformation pipeline for BraTS 2024 dataset.
    """
    assert mode in ["train", "valid", "test"], f"Unknown mode: {mode}"
    mods_list = ["t1c", "t1n", "t2f", "t2w"]
    keys = mods_list + ["label"] if mode in ["train", "valid"] else mods_list
    transform = [
        transforms.AddChanneld(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        RobustZScoreNormalization(keys=mods_list),
        transforms.ConcatItemsd(keys=mods_list, name="image", dim=0),
        transforms.DeleteItemsd(keys=mods_list),
    ]

    if mode == "train":
        transform += [
                    transforms.RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=[args.patch_size] * 3,
                        pos=args.pos_ratio,
                        neg=args.neg_ratio,
                        num_samples=1,
                    ),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    transforms.RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.33),
                    transforms.RandGaussianSmoothd(keys="image", prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                    transforms.RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.3)),
                    transforms.EnsureTyped(keys=["image", "label"]),
                    ]
    return transforms.Compose(transform)

