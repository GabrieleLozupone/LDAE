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

from monai.transforms import MapTransform
import torch


class SwapDimensionsBasedOnSlicingPlane(MapTransform):
    def __init__(self, keys, slicing_plane="sagittal"):
        super().__init__(keys)
        self.slicing_plane = slicing_plane

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]
            if self.slicing_plane == 'coronal':
                x = torch.permute(x, (0, 2, 1, 3))
            elif self.slicing_plane == 'axial':
                x = torch.permute(x, (0, 3, 1, 2))
            else:
                if self.slicing_plane != 'sagittal':
                    print("Invalid slicing plane, using sagittal")
                # If 'sagittal', no permutation is applied as default
            d[key] = x
        return d
