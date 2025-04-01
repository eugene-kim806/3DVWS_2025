import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
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

    # Define label names and corresponding colors
    label_classes = {
        0: "Background",
        1: "Non-Enhancing Tumor Core (NETC)",
        2: "Surrounding Non-Enhancing FLAIR Hyperintensity (SNFH)",
        3: "Enhancing Tumor (ET)",
        4: "Resection Cavity (RC)"
    }

    label_colors = [
        (0, 0, 0, 0),        # 0 - Black (Background, Transparent)
        (0.8, 0, 0, 1),      # 1 - Red (NETC)
        (0, 0.8, 0, 1),      # 2 - Green (SNFH)
        (0, 0, 0.8, 1),      # 3 - Blue (ET)
        (0.8, 0.8, 0, 1)     # 4 - Yellow (RC)
    ]

    cmap = ListedColormap(label_colors)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, len(label_colors) + 0.5, 1), ncolors=len(label_colors))

    sagittal_slice = image.shape[1] // 2
    coronal_slice = image.shape[2] // 2
    axial_slice = image.shape[3] // 2

    fig, axes = plt.subplots(6, 3, figsize=(15, 30))

    # MRI Image with Segmentation Overlay
    for i in range(4):
        sagittal_view = image[i, sagittal_slice, :, :]
        coronal_view = image[i, :, coronal_slice, :]
        axial_view = image[i, :, :, axial_slice]

        axes[i, 0].imshow(sagittal_view, cmap='gray')
        axes[i, 0].imshow(seg[0, sagittal_slice, :, :], cmap=cmap, norm=norm, alpha=0.5)
        axes[i, 0].set_title(f'Channel {i+1} - Sagittal')

        axes[i, 1].imshow(coronal_view, cmap='gray')
        axes[i, 1].imshow(seg[0, :, coronal_slice, :], cmap=cmap, norm=norm, alpha=0.5)
        axes[i, 1].set_title(f'Channel {i+1} - Coronal')

        axes[i, 2].imshow(axial_view, cmap='gray')
        axes[i, 2].imshow(seg[0, :, :, axial_slice], cmap=cmap, norm=norm, alpha=0.5)
        axes[i, 2].set_title(f'Channel {i+1} - Axial')

    # Pure Segmentation Visualization (No grayscale)
    sagittal_seg = seg[0, sagittal_slice, :, :]
    coronal_seg = seg[0, :, coronal_slice, :]
    axial_seg = seg[0, :, :, axial_slice]

    axes[4, 0].imshow(sagittal_seg, cmap=cmap, norm=norm)
    axes[4, 0].set_title('Segmentation - Sagittal')

    axes[4, 1].imshow(coronal_seg, cmap=cmap, norm=norm)
    axes[4, 1].set_title('Segmentation - Coronal')

    axes[4, 2].imshow(axial_seg, cmap=cmap, norm=norm)
    axes[4, 2].set_title('Segmentation - Axial')

    # Grayscale Segmentation for Reference
    axes[5, 0].imshow(sagittal_seg, cmap='gray')
    axes[5, 0].set_title('Segmentation Gray - Sagittal')

    axes[5, 1].imshow(coronal_seg, cmap='gray')
    axes[5, 1].set_title('Segmentation Gray - Coronal')

    axes[5, 2].imshow(axial_seg, cmap='gray')
    axes[5, 2].set_title('Segmentation Gray - Axial')

    # Set the super title for the entire figure
    fig.suptitle('Visualizations of Loaded Data', fontsize=16, y=0.995)

    # Add Legend for Segmentation Classes
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=color, label=label) for color, label in zip(label_colors[1:], list(label_classes.values())[1:])]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(save_vis_img_dir / f'{name}_visualization.png')
