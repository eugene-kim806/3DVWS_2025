import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path

class BraTS24GliDataset(Dataset):
    def __init__(self, data_dir:str, mode:str, case_names:list=[], transforms=None):
        super().__init__()
        assert mode in ['train', 'valid', 'test'], f"Unknown mode: {mode}"
        self.mode = mode
        self.data_path = Path(data_dir)
        self.case_names = case_names
        self.transforms = transforms
        self.original_affine = None

    def _load_nifti(self, index:int):
        name = self.case_names[index]  # e.g., 'BraTS-GLI-02903-100'
        case_path = self.data_path / name

        # Load NifTi images & Get image data
        t1c, t1n, t2f, t2w = [
            nib.load(case_path / f'{name}-{modality}.nii.gz').get_fdata() 
            for modality in ['t1c', 't1n', 't2f', 't2w']]
        
        item = {'t1c': t1c,
                't1n': t1n,
                't2f': t2f,
                't2w': t2w}
        
        # Load ground truth segmentation label (if not test mode; no ground truth exists for test dataset)
        if self.mode in ['train', 'valid']:
            seg = nib.load(case_path / f'{name}-seg.nii.gz')
            self.original_affine = seg.affine
            item['label'] = seg.get_fdata()  # add label to the item dict

        return item, name
    
    def __len__(self):
        return len(self.case_names)
    
    def __getitem__(self, index:int):
        item, name = self._load_nifti(index=index)
        if self.transforms:
            # after random crop to patch size, output is list
            # number of sample is 1, so just return the dict
            item = self.transforms(item)
        
        if self.mode == 'train':
            item = item[0]

        return item['image'], item['label'], index, name, self.original_affine
