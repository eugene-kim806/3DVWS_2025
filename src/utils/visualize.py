import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_mri(args, image, seg, name):
    """
    Visualizes the MRI scan with segmentation overlays for three primary views.
    
    Parameters:
    - image: (4, 96, 96, 96) numpy array, MRI scan with 4 channels.
    - seg: (1, 96, 96, 96) numpy array, segmentation mask.
    - name: str, name of the patient/timepoint.
    """
    save_example_dir = Path(args.example_dir)
    save_example_dir.mkdir(parents=True, exist_ok=True)
    save_vis_img_dir = save_example_dir / 'data_loading_check'
    save_vis_img_dir.mkdir(parents=True, exist_ok=True)

    sagittal_slice = image.shape[1] // 2
    coronal_slice = image.shape[2] // 2
    axial_slice = image.shape[3] // 2
    
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    
    for i in range(4):  # Iterate over the 4 channels
        sagittal_view = image[i, sagittal_slice, :, :]
        coronal_view = image[i, :, coronal_slice, :]
        axial_view = image[i, :, :, axial_slice]
        
        # Sagittal View
        axes[i, 0].imshow(sagittal_view, cmap='gray')
        axes[i, 0].imshow(seg[0, sagittal_slice, :, :], cmap='jet', alpha=0.5)
        axes[i, 0].set_title(f'Channel {i+1} - Sagittal')
        
        # Coronal View
        axes[i, 1].imshow(coronal_view, cmap='gray')
        axes[i, 1].imshow(seg[0, :, coronal_slice, :], cmap='jet', alpha=0.5)
        axes[i, 1].set_title(f'Channel {i+1} - Coronal')
        
        # Axial View
        axes[i, 2].imshow(axial_view, cmap='gray')
        axes[i, 2].imshow(seg[0, :, :, axial_slice], cmap='jet', alpha=0.5)
        axes[i, 2].set_title(f'Channel {i+1} - Axial')
    
    # Extract segmentation masks for grayscale display
    sagittal_seg = seg[0, sagittal_slice, :, :]
    coronal_seg = seg[0, :, coronal_slice, :]
    axial_seg = seg[0, :, :, axial_slice]
    
    # Colored segmentation masks
    axes[4, 0].imshow(sagittal_seg, cmap='jet')
    axes[4, 0].set_title('Segmentation - Sagittal')
    
    axes[4, 1].imshow(coronal_seg, cmap='jet')
    axes[4, 1].set_title('Segmentation - Coronal')
    
    axes[4, 2].imshow(axial_seg, cmap='jet')
    axes[4, 2].set_title('Segmentation - Axial')
    
    # Grayscale segmentation masks
    axes[5, 0].imshow(sagittal_seg, cmap='gray')
    axes[5, 0].set_title('Segmentation Gray - Sagittal')
    
    axes[5, 1].imshow(coronal_seg, cmap='gray')
    axes[5, 1].set_title('Segmentation Gray - Coronal')
    
    axes[5, 2].imshow(axial_seg, cmap='gray')
    axes[5, 2].set_title('Segmentation Gray - Axial')
    
    plt.tight_layout()
    plt.savefig(save_vis_img_dir / f'{name}_visualization.png')
