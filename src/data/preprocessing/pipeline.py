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

import ants
import antspynet
import os

def extract_brain(image, modality):
    return antspynet.utilities.brain_extraction(image, modality=modality, verbose=False)

def preprocess_image(img_path, template=ants.image_read(ants.get_ants_data('mni')), pbar=None):
    """
    Preprocesses a T1-weighted MRI image by performing bias field correction, brain extraction,
    and registration to a template space.

    Args:
        img_path (str): Path to the input MRI image file
        template (ANTsImage, optional): Template image for registration. Defaults to MNI template.
        pbar (tqdm, optional): Progress bar for tracking preprocessing steps. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - ANTsImage: The preprocessed image after registration
            - str: Path to the saved preprocessed image file
    """
    
    if pbar == None:
        print(f"Loading image {os.path.basename(img_path)}")
    else:
        pbar.set_description(f"Loading image {os.path.basename(img_path)}")
    # Load the image
    img = ants.image_read(img_path)

    if pbar == None:
        print(f"Applying bias field correction {os.path.basename(img_path)}")
    else:
        pbar.set_description(f"Applying bias field correction {os.path.basename(img_path)}")
    # Apply bias field correction
    img = ants.n4_bias_field_correction(img)

    if pbar == None:
        print(f"Extracting brain {os.path.basename(img_path)}")
    else:
        pbar.set_description(f"Extracting brain {os.path.basename(img_path)}")
    # Extract the brain
    seg = extract_brain(image=img, modality="t1")
    brain = img * seg

    if pbar == None:
        print(f"Registering brain to template {os.path.basename(img_path)}")
    else:
        pbar.set_description(f"Registering brain to template {os.path.basename(img_path)}")
    # Register the brain to the template
    brain_reg = ants.registration(fixed=template, moving=brain, type_of_transform='SyN')

    if pbar == None:
        print(f"Saving processed image {os.path.basename(img_path)}")
    else:
        pbar.set_description(f"Saving processed image {os.path.basename(img_path)}")
    mri_filename = os.path.basename(img_path)
    mri_parent_dir = os.path.dirname(img_path)
    ants.image_write(brain_reg['warpedmovout'], os.path.join(mri_parent_dir, 'processed_' + mri_filename))
    return brain_reg['warpedmovout'], os.path.join(mri_parent_dir, 'processed_' + mri_filename)
