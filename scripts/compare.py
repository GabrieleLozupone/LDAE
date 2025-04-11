#!/Users/gabrielelozupone/miniconda3/envs/brain-diffae/bin/python

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from monai.transforms import GaussianSharpen


def compare_3d_volumes_with_diff(
        original_data: np.ndarray,
        reconstructed_data: np.ndarray,
        original_title: str = 'Original',
        reconstructed_title: str = 'Reconstructed',
        diff_title: str = 'Difference',
        voxel_sizes=(1.0, 1.0, 1.0)
):
    """
    Interactive visualization of two 3D volumes side by side (original vs. reconstructed),
    plus a difference map (|original - reconstructed|).

    Provides a slice slider and plane selection radio buttons.

    Args:
        original_data (np.ndarray): 3D array of shape [D, W, H].
        reconstructed_data (np.ndarray): 3D array of shape [D, W, H].
        voxel_sizes (tuple): (size_along_D, size_along_W, size_along_H),
                             used to preserve aspect ratios for each plane.
    """

    # --- 1) Create a difference volume (absolute difference) ---
    diff_data = np.abs(original_data - reconstructed_data)

    # --- 2) Initial Setup ---
    plane = 'axial'

    def get_max_slice(data_shape, plane_name):
        """Returns the maximum slice index for the given plane."""
        if plane_name == 'axial':
            return data_shape[2] - 1
        elif plane_name == 'sagittal':
            return data_shape[0] - 1
        elif plane_name == 'coronal':
            return data_shape[1] - 1

    max_slice = get_max_slice(original_data.shape, plane)
    slice_idx = max_slice // 2

    def get_slice(data, plane_name, idx):
        if plane_name == 'axial':
            slice_2d = data[:, :, idx]
            aspect = voxel_sizes[1] / voxel_sizes[0]
        elif plane_name == 'sagittal':
            slice_2d = data[idx, :, :]
            aspect = voxel_sizes[2] / voxel_sizes[1]
        else:
            slice_2d = data[:, idx, :]
            aspect = voxel_sizes[2] / voxel_sizes[0]

        slice_2d = np.rot90(slice_2d, k=-1)
        return slice_2d, aspect

    # --- 3) Initial slices for display ---
    slice_original, aspect = get_slice(original_data, plane, slice_idx)
    slice_reconstructed, _ = get_slice(reconstructed_data, plane, slice_idx)
    slice_diff, _ = get_slice(diff_data, plane, slice_idx)

    # --- 4) Set up the figure and axes ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    im1 = axes[0].imshow(slice_original, cmap='gray', origin='lower', aspect=aspect)
    axes[0].set_title(original_title)

    im2 = axes[1].imshow(slice_reconstructed, cmap='gray', origin='lower', aspect=aspect)
    axes[1].set_title(reconstructed_title)

    im3 = axes[2].imshow(slice_diff, cmap='jet', origin='lower', aspect=aspect,
                         vmin=0, vmax=diff_data.max())
    axes[2].set_title(diff_title)

    cbar = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Difference')

    # --- 5) Slider for selecting the slice index ---
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Slice',
        valmin=0,
        valmax=max_slice,
        valinit=slice_idx,
        valfmt='%d'
    )

    # --- 6) Radio buttons for selecting the slicing plane ---
    ax_radio = plt.axes([0.05, 0.5, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ('axial', 'sagittal', 'coronal'))

    # --- 7) Callbacks ---
    def update_slice(val):
        nonlocal slice_idx
        slice_idx = int(val)

        slice_orig, aspect_val = get_slice(original_data, plane, slice_idx)
        slice_recon, _ = get_slice(reconstructed_data, plane, slice_idx)
        slice_d, _ = get_slice(diff_data, plane, slice_idx)

        im1.set_data(slice_orig)
        im2.set_data(slice_recon)
        im3.set_data(slice_d)

        im1.set_aspect(aspect_val)
        im2.set_aspect(aspect_val)
        im3.set_aspect(aspect_val)

        im1.axes.figure.canvas.draw_idle()
        im2.axes.figure.canvas.draw_idle()
        im3.axes.figure.canvas.draw_idle()

    def update_plane(label):
        nonlocal plane, max_slice, slice_idx, slider
        plane = label
        max_slice = get_max_slice(original_data.shape, plane)

        if slice_idx > max_slice:
            slice_idx = max_slice // 2

        slider.ax.cla()
        slider_new = Slider(
            ax=slider.ax,
            label='Slice',
            valmin=0,
            valmax=max_slice,
            valinit=slice_idx,
            valfmt='%d'
        )
        slider_new.on_changed(update_slice)
        slider = slider_new

        update_slice(slice_idx)

    slider.on_changed(update_slice)
    radio.on_clicked(update_plane)

    plt.show()


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Visualize and compare NIfTI images.")
    parser.add_argument('original', type=str, help="Path to the original NIfTI file.")
    parser.add_argument('reconstructed', type=str, help="Path to the reconstructed NIfTI file.")
    args = parser.parse_args()

    # Load the NIfTI images
    original_img = nib.load(args.original)
    reconstructed_img = nib.load(args.reconstructed)

    original_data = original_img.get_fdata()
    reconstructed_data = reconstructed_img.get_fdata()

    compare_3d_volumes_with_diff(original_data, reconstructed_data, original_title='Original',
                                 reconstructed_title='Manipulated')


if __name__ == "__main__":
    main()
