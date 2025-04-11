# --------------------------------------------------------------------------------
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

import lightning as L
from src.ldae.utils.nets_utils import (
    replace_classifier_with_identity,
    convert_first_layer_rgb_to_grayscale,
    freeze_backbone,
    initialize_weights
)


def import_object(import_path):
    module_path, attr_name = import_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[attr_name])
    return getattr(module, attr_name)


def import_weights(weights_path):
    # Split the weights path to find the correct module
    module_path, attr_name = weights_path.rsplit('.', 1)
    try:
        # Attempt to import the module directly
        module = __import__(module_path, fromlist=[attr_name])
        return getattr(module, attr_name)
    except ImportError:
        # Handle special cases where weights are defined in submodules
        # For torchvision, weights are often in a submodule with the model name
        parent_module_path, submodule_name = module_path.rsplit('.', 1)
        parent_module = __import__(parent_module_path, fromlist=[submodule_name])
        submodule = getattr(parent_module, submodule_name)
        return getattr(submodule, attr_name)


def instantiate_backbone(class_path, weights):
    # Import the backbone class
    BackboneClass = import_object(class_path)
    # Handle weights
    if weights:
        # Import weights enum value
        weights_value = import_weights(weights)
        return BackboneClass(weights=weights_value)
    else:
        return BackboneClass()


class BackboneBaseModule(L.LightningModule):
    def __init__(
            self,
            net_class_path,
            weights=None,
            freeze_perc=0.5,
            grayscale=True
    ):
        super().__init__()
        # Instantiate the backbone model
        self.backbone = instantiate_backbone(net_class_path, weights)
        print("Backbone model loaded successfully.")

        # Optionally convert first layer to grayscale
        if grayscale:
            convert_first_layer_rgb_to_grayscale(self.backbone)

        # Replace classifier with identity
        replace_classifier_with_identity(self.backbone)

        # Freeze the backbone partially
        freeze_backbone(self.backbone, freeze_perc)

        if weights is None:
            print("No weights provided. The model will be loaded with weights according to kaiming method.")
            initialize_weights(self.backbone, init_type='kaiming')

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, x):
        return self.backbone(x)
