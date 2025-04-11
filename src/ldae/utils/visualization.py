# Copyright (c) 2025 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
#
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
#
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def get_slices(volume, num_slices=3, total_slices_per_plane=None, offset=None):
    """
    Return axial, coronal, sagittal slices from 'volume', with 'num_slices' slices per plane.
    Slices are selected around the center slice (e.g., 80 out of 160), with spacing defined by 'offset'.
    If total_slices_per_plane is not provided, it is inferred from the volume shape.
    If offset is not provided, it defaults to a reasonable value based on the center.
    """ 
    dims = np.array(volume.shape)
    if total_slices_per_plane is None:
        total_slices_per_plane = dims
    center_slices = [total_slices // 2 for total_slices in total_slices_per_plane]
    if offset is None:
        offset = center_slices[0] // 2
    if num_slices == 1:
        idxs = [[center] for center in center_slices]
    else:
        idxs = [
            np.array([center_slices[0] + offset * (i - (num_slices // 2)) for i in range(num_slices)], dtype=int),
            np.array([center_slices[1] + offset * (i - (num_slices // 2)) for i in range(num_slices)], dtype=int),
            np.array([center_slices[2] + offset * (i - (num_slices // 2)) for i in range(num_slices)], dtype=int)
        ]
    for i in range(3):
        idxs[i] = np.clip(idxs[i], 0, dims[i] - 1)
    axial = [(volume[:, :, i], i) for i in idxs[2]]
    coronal = [(volume[:, i, :], i) for i in idxs[1]]
    sagittal = [(volume[i, :, :], i) for i in idxs[0]]
    return axial, coronal, sagittal

def pad_to_square(arr):
    """
    Pad a 2D array 'arr' so it becomes NxN, with zeros around the edges.
    The original array is centered in the new square.
    """
    h, w = arr.shape
    s = max(h, w)
    padded = np.zeros((s, s), dtype=arr.dtype)
    start_h = (s - h) // 2
    start_w = (s - w) // 2
    padded[start_h:start_h + h, start_w:start_w + w] = arr
    return padded

def plot_comparison_columns_by_view(slices_list, titles, num_slices=3):
    """
    slices_list: a list of 3 sets of slices: [reconstructed, manipulated, difference]
    titles: e.g., ['Reconstructed', 'Manipulated', 'Difference']
    """
    nrows = 3                # Reconstructed / Manipulated / Difference
    ncols = 3 * num_slices   # Axial x N, Coronal x N, Sagittal x N

    # Adjust figure size and create a GridSpec with an extra column for the colorbar
    fig = plt.figure(figsize=(26, 8))  # Increased width for the colorbar
    gs = GridSpec(nrows, ncols + 3, figure=fig, wspace=0.05, hspace=0.2, 
                  width_ratios=[1]*num_slices + [0.2] + [1]*num_slices + [0.2] + [1]*num_slices + [0.2])

    views = ['Axial', 'Coronal', 'Sagittal']
    diff_mappable = None  # To store the mappable for the colorbar

    for view_idx, view in enumerate(views):
        col_offset = view_idx * (num_slices + 1)  # Add 1 for the gap column
        for slice_idx in range(num_slices):
            col_idx = col_offset + slice_idx

            for row_idx, (slices, title) in enumerate(zip(slices_list, titles)):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                slice_img, slice_num = slices[view_idx][slice_idx]
                slice_img = pad_to_square(slice_img)
                rotated_img = np.rot90(slice_img)
                cmap = 'jet' if title == 'Difference' else 'gray'
                im = ax.imshow(rotated_img, cmap=cmap)

                # Store the mappable for the 'Difference' row
                if title == 'Difference':
                    diff_mappable = im

                ax.axis('off')
                ax.set_aspect('equal', adjustable='box')

                # Label the top row with the view title
                if row_idx == 0 and slice_idx == round(num_slices / 2) - 1:
                    ax.set_title(f'{view}', fontsize=14, pad=10)

                # Add row titles on the left side
                if col_idx == 0 and slice_idx == 0:
                    ax.text(-0.15, 0.5, title, fontsize=12, va='center', ha='center', 
                            transform=ax.transAxes, rotation=90)

    # Add gaps to separate the slicing planes
    for gap_col in [num_slices, 2*num_slices + 1]:
        for row in range(nrows):
            ax = fig.add_subplot(gs[row, gap_col])
            ax.axis('off')

    # Add a full-height colorbar as a "4th plot"
    if diff_mappable is not None:
        ax = fig.add_subplot(gs[:, -1])
        ax.axis('off')
        cbar_ax = fig.add_subplot(gs[:, -1])  # Span all rows in the last column
        cbar = fig.colorbar(diff_mappable, cax=cbar_ax, aspect=10)
        cbar.set_label('Absolute Difference', fontsize=12, labelpad=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_difference(vol1, vol2, title1, title2, offset=10):
    difference = np.abs(vol1 - vol2)
    slices_vol1 = get_slices(vol1, offset=offset)
    slices_vol2 = get_slices(vol2, offset=offset)
    slices_diff = get_slices(difference, offset=offset)

    plot_comparison_columns_by_view(
        [slices_vol1, slices_vol2, slices_diff],
        [title1, title2, "Difference"]
    )