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
import torch.nn as nn
import lightning as L
import torchmetrics
import torch
from .backbone_base import BackboneBaseModule


class LitClassification3DBase(L.LightningModule):
    """
    This is a base class for 3D classification tasks. It's used to train, validate, and test a model on 3D data.
    """
    def __init__(self,
                 backbone_args,
                 embedding_dim=768,
                 num_classes=2,
                 load_size=224,
                 lr=1e-4,
                 weight_decay=1e-3,
                 epochs=30,
                 warmup_epochs=5,
                 warmup_start_lr=1e-5):
        super().__init__()
        # Load the backbone
        if backbone_args is not None:
            self.backbone = BackboneBaseModule(**backbone_args)
        # Set up the model parameters
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.load_size = load_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        # Define the metrics
        self._init_metrics()
        # Define the loss
        self.loss = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def _init_metrics(self):
        """Initializes metrics for binary or multiclass classification."""
        if self.num_classes == 2:
            # Train metrics
            self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
            # Validation metrics
            self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
            self.val_sensitivity = torchmetrics.classification.BinaryRecall()
            self.val_specificity = torchmetrics.classification.BinarySpecificity()
            self.val_mcc = torchmetrics.classification.BinaryMatthewsCorrCoef()
            # Test metrics
            self.test_accuracy = torchmetrics.classification.BinaryAccuracy()
            self.test_precision = torchmetrics.classification.BinaryPrecision()
            self.test_recall = torchmetrics.classification.BinaryRecall()
            self.test_mcc = torchmetrics.classification.BinaryMatthewsCorrCoef()
            self.test_f1 = torchmetrics.classification.BinaryF1Score()
            self.test_auc = torchmetrics.classification.BinaryAUROC()

        else:
            # Train metrics
            self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
            # Validation metrics
            self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_mcc = torchmetrics.classification.MatthewsCorrCoef(task="multiclass", num_classes=self.num_classes)
            # Test metrics
            self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_mcc = torchmetrics.classification.MatthewsCorrCoef(task="multiclass", num_classes=self.num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        predictions = self(x)
        loss = self.loss(predictions, y)
        if self.num_classes == 2:
            predictions = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.train_accuracy.update(predictions, y)
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        train_acc = self.train_accuracy.compute()
        self.log("train_acc", train_acc, prog_bar=True, sync_dist=True)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss(logits, y)
        predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        self.val_accuracy.update(predictions, y)
        if self.num_classes == 2:
            self.val_specificity.update(predictions, y)
            self.val_sensitivity.update(predictions, y)
        self.val_mcc.update(predictions, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        val_mcc = self.val_mcc.compute()
        # Convert val_mcc to torch.float32 to silence warnings
        val_mcc = val_mcc.clone().detach().float()
        self.log("val_acc", val_acc, prog_bar=True, sync_dist=True)
        self.log("val_mcc", val_mcc, prog_bar=True, sync_dist=True)
        self.val_accuracy.reset()
        self.val_mcc.reset()

        if self.num_classes == 2:
            val_spec = self.val_specificity.compute()
            val_sens = self.val_sensitivity.compute()
            self.log("val_specificity", val_spec, prog_bar=True, sync_dist=True)
            self.log("val_sensitivity", val_sens, prog_bar=True, sync_dist=True)
            self.val_specificity.reset()
            self.val_sensitivity.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss(logits, y)

        # Convert to regular tensors for metrics
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        # Ensure y is a regular tensor
        if hasattr(y, "as_tensor"):
            y = y.as_tensor()  # Convert MetaTensor to regular tensor

        self.test_accuracy.update(predictions, y)
        self.test_mcc.update(predictions, y)

        if self.num_classes == 2:
            self.test_f1.update(predictions, y)
            # Make sure to convert probabilities to regular tensor if needed
            prob_tensor = probabilities[:, 1]
            if hasattr(prob_tensor, "as_tensor"):
                prob_tensor = prob_tensor.as_tensor()
            self.test_auc.update(prob_tensor, y)
            self.test_recall.update(predictions, y)
            self.test_precision.update(predictions, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        test_acc = self.test_accuracy.compute()
        test_mcc = self.test_mcc.compute()
        # Convert test_mcc to torch.float32 to silence warnings
        test_mcc = test_mcc.clone().detach().float()
        self.log("test_acc", test_acc, prog_bar=True, sync_dist=True)
        self.log("test_mcc", test_mcc, prog_bar=True, sync_dist=True)
        self.test_accuracy.reset()
        self.test_mcc.reset()
        if self.num_classes == 2:
            test_f1 = self.test_f1.compute()
            test_auc = self.test_auc.compute()
            test_recall = self.test_recall.compute()
            test_precision = self.test_precision.compute()
            self.log("test_f1", test_f1, prog_bar=True, sync_dist=True)
            self.log("test_auc", test_auc, prog_bar=True, sync_dist=True)
            self.log("test_recall", test_recall, prog_bar=True, sync_dist=True)
            self.log("test_precision", test_precision, prog_bar=True, sync_dist=True)
            self.test_specificity.reset()
            self.test_sensitivity.reset()
            self.test_f1.reset()
            self.test_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1.0e-07)
        return optimizer

    def forward(self, x):
        """This will be specialized by the child class."""
        raise NotImplementedError("You need to implement this method in the child class.")

