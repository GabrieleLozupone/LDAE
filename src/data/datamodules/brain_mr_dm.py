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
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.data.image_reader import NumpyReader
import numpy as np
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
    Compose,
    Lambdad,
    ToTensord,
    RandFlipd,
    RandAffined,
    RandShiftIntensityd,
    RandAdjustContrastd,
    ThresholdIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
)
from src.data.transforms import SwapDimensionsBasedOnSlicingPlane
from monai.utils import set_determinism
from rich.columns import Columns
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import GroupShuffleSplit


class BrainMRDataModule(L.LightningDataModule):
    def __init__(self,
                 csv_path: str = "/data/dataset.csv",
                 classes=None,
                 target: str = 'diagnosis',
                 balance_classes: bool = False,
                 fake_3d: bool = False,
                 slicing_plane: str = 'axial',
                 resize_to=(128, 160, 128),
                 apply_augmentation: bool = False,
                 batch_size: int = 2,
                 num_workers: int = 16,
                 seed: int = 42,
                 val_size: float = 0.2,
                 test_size: float = 0.1,
                 return_all_info: bool = False,
                 load_latents: bool = False,
                 load_images: bool = True):
        super().__init__()
        self.csv_path = csv_path
        self.target = target
        self.balance_classes = balance_classes

        self.fake_3d = fake_3d
        self.slicing_plane = slicing_plane
        self.resize_to = resize_to
        self.apply_augmentation = apply_augmentation

        self.batch_size = batch_size
        self.classes = classes
        self.num_workers = num_workers
        self.seed = seed
        self.val_size = val_size
        self.test_size = test_size
        self.return_all_info = return_all_info

        self.load_latents = load_latents
        self.load_images = load_images
        self.save_hyperparameters()

    def __split_train_test__(self, df):
        splitter = GroupShuffleSplit(test_size=self.test_size, random_state=self.seed)
        groups = df['subject']

        train_idx, test_val_idx = next(splitter.split(df, groups=groups))
        df_train_val = df.iloc[train_idx]
        groups_train_val = df_train_val['subject']

        val_splitter = GroupShuffleSplit(test_size=self.val_size, random_state=self.seed)
        train_idx, val_idx = next(val_splitter.split(df_train_val, groups=groups_train_val))
        train_df = df_train_val.iloc[train_idx]
        val_df = df_train_val.iloc[val_idx]
        test_df = df.iloc[test_val_idx]

        return train_df, val_df, test_df

    def __sanity_check_split__(self):
        """
        Check if the split is done correctly.
        :return:
        """
        train_subjects = self.train_df['subject'].unique()
        val_subjects = self.val_df['subject'].unique()
        test_subjects = self.test_df['subject'].unique()
        assert len(set(train_subjects).intersection(val_subjects)) == 0, "Train and Val subjects intersect"
        assert len(set(train_subjects).intersection(test_subjects)) == 0, "Train and Test subjects intersect"
        assert len(set(val_subjects).intersection(test_subjects)) == 0, "Val and Test subjects intersect"
        # Print sanity check passed in green
        console = Console()
        console.print("\nSanity check split passed. No subject intersection between sets.\n", style="bold green")

    def __print_info__(self):
        console = Console()

        def create_table(df, name):
            samples_per_class = df[self.target].value_counts()
            total_samples = len(df)
            subjects_per_class = df.groupby(self.target)['subject'].nunique()
            total_subjects = df['subject'].nunique()

            # Create a Rich Table
            table = Table(title=f"{name} Data Info", style="bold white")

            # Add columns with header color
            table.add_column("Class", justify="center", header_style="bold magenta")
            table.add_column("Samples", justify="center", header_style="bold blue")
            table.add_column("Subjects", justify="center", header_style="bold green")

            # Add rows with white color
            for class_name in samples_per_class.index:
                table.add_row(str(class_name), str(samples_per_class[class_name]), str(subjects_per_class[class_name]),
                              style="white")

            # Add total row at the end, also in white
            table.add_row("Total", str(total_samples), str(total_subjects), style="white")

            return table

        # Generate tables for Train, Validation, and Test data
        train_table = create_table(self.train_df, "Train")
        val_table = create_table(self.val_df, "Validation")
        test_table = create_table(self.test_df, "Test")

        # Display all tables side by side
        console.print("\n")
        console.print(Columns([train_table, val_table, test_table]))

    def __data_split__(self):
        """
        Prepare the data for training. This function will load the data, balance the classes, and split the data into
        train, val, test. The balancing is done by subjects first, then by images in order to have a fair split. The
        split is done by subjects first, then by images in order to avoid data leakage.
        :return:
        """
        # Set deterministic behavior for reproducibility
        set_determinism(42)
        # Load data
        df = pd.read_csv(self.csv_path)

        # Select only the classes we want
        if self.classes is not None:
            df = df[df[self.target].isin(self.classes)]
        else:
            self.classes = df[self.target].unique()
            # If a class have less than 10% of the majority class, remove it
            majority_class = df[self.target].value_counts().idxmax()
            for c in self.classes:
                if df[df[self.target] == c].shape[0] < 0.1 * df[df[self.target] == majority_class].shape[0]:
                    df = df[df[self.target] != c]
            self.classes = df[self.target].unique()

        # Balance classes by subjects first, then by images if balance_classes is True
        if self.balance_classes:
            subjects_df = df.drop_duplicates(subset='subject')
            balanced_subjects_df = (subjects_df.groupby(self.target, group_keys=False)
                                    .apply(
                lambda x: x.sample(subjects_df[self.target].value_counts().min(), random_state=self.seed))
                                    .reset_index(drop=True))
            balanced_df = df[df['subject'].isin(balanced_subjects_df['subject'])]
            df = (balanced_df.groupby(self.target, group_keys=False).apply(lambda x: x.sample(balanced_df[self.target].
                                                                                              value_counts().min(),
                                                                                              random_state=self.seed))
                  .reset_index(drop=True))

        # Split data into train, val, test
        self.train_df, self.val_df, self.test_df = self.__split_train_test__(df)

        # Print info
        self.__print_info__()

        # Sanity check
        self.__sanity_check_split__()

    def setup(self, stage: str):
        # Split the data
        self.__data_split__()

        # Get the dict for MONAI
        self.train_data = [{
            "image": row['path'],
            "label": row[self.target],
            "path": row['path'],
            "latent": row['latent_path'],
            "age": row['age'],
            "subject_id": row['subject']} for _, row in self.train_df.iterrows()
        ]
        self.val_data = [{
            "image": row['path'],
            "label": row[self.target],
            "path": row['path'],
            "latent": row['latent_path'],
            "age": row['age'],
            "subject_id": row['subject']} for _, row in self.val_df.iterrows()
        ]
        self.test_data = [{
            "image": row['path'],
            "label": row[self.target],
            "path": row['path'],
            "latent": row['latent_path'],
            "age": row['age'],
            "subject_id": row['subject']} for _, row in self.test_df.iterrows()
        ]

        # Define mapping from class to index
        self.label_to_idx_map = {label: idx for idx, label in enumerate(sorted(self.classes))}
        print(f"Label to index map: {self.label_to_idx_map}")

        # Define augmentation transform
        if self.apply_augmentation:
            aug = Compose([
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                RandAffined(
                    keys=["image"],
                    rotate_range=(  # Small rotations for slight orientation changes
                        (-5, 5),  # X-axis rotation
                        (-5, 5),  # Y-axis rotation
                        (-5, 5)  # Z-axis rotation
                    ),
                    translate_range=(-2, 2),
                    scale_range=(-0.02, 0.02),
                    spatial_size=self.resize_to,
                    prob=0.1,
                ),
                RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
                RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
                ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
            ])
            print("\nAugmentation is applied.\n")
        else:
            aug = Compose([])

        # Slice the 3D image based on the slicing plane
        if self.fake_3d:
            swap = SwapDimensionsBasedOnSlicingPlane(keys=["image"], slicing_plane=self.slicing_plane)
        else:
            swap = Compose([])

        if self.load_images:
            image_transform = Compose([
                LoadImaged(keys=["image"], reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"]),
                Spacingd(pixdim=1.5, keys=['image']),
                ResizeWithPadOrCropd(spatial_size=self.resize_to, mode='minimum', keys=['image']),
                ScaleIntensityd(minv=0, maxv=1, keys=['image']),
                aug,  # Include aug
                swap,  # Include swap
                ToTensord(keys=["image"]),
            ])
            image_test_transform = Compose([
                LoadImaged(keys=["image"], reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"]),
                Spacingd(pixdim=1.5, keys=['image']),
                ResizeWithPadOrCropd(spatial_size=self.resize_to, mode='minimum', keys=['image']),
                ScaleIntensityd(minv=0, maxv=1, keys=['image']),
                swap,  # Include swap
                ToTensord(keys=["image"]),
            ])
        else:
            image_transform = Compose([])
            image_test_transform = Compose([])

        # Define latent transform
        if self.load_latents:
            latent_transform = Compose([
                LoadImaged(keys=['latent'], reader=NumpyReader(npz_keys=['data'])),
                EnsureChannelFirstd(keys=['latent'], channel_dim=0),
                ToTensord(keys=['latent']),
            ])
        else:
            latent_transform = Compose([])

        # Define train transforms
        self.train_transform = Compose([
            image_transform,  # Include image_transform
            latent_transform,  # Include latent_transform
            Lambdad(keys=["label"], func=lambda x: self.label_to_idx_map[x]),
            ToTensord(keys=["label"]),
            ToTensord(keys=["age"]),
        ])

        self.test_transform = Compose([
            image_test_transform,  # Include image_test_transform
            latent_transform,  # Include latent_transform
            Lambdad(keys=["label"], func=lambda x: self.label_to_idx_map[x]),
            ToTensord(keys=["label"]),
            ToTensord(keys=["age"]),
        ])

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = Dataset(data=self.train_data, transform=self.train_transform)
            self.val_ds = Dataset(data=self.val_data, transform=self.test_transform)
            if self.load_images:
                print(f"Train samples size: {self.train_ds[0]['image'].shape}\n")
            if self.load_latents:
                print(f"Train latent size: {self.train_ds[0]['latent'].shape}\n")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = Dataset(data=self.test_data, transform=self.test_transform)
            if self.load_images:
                print(f"Test samples size: {self.test_ds[0]['image'].shape}")

        if stage == "predict":
            self.predict_ds = Dataset(data=self.test_data, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)
