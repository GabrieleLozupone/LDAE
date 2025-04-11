import random

import numpy as np
import torch
import torchmetrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_binary_classification_metrics(model, dataloader, device, latents_mean, latents_std):
    """
    Computes binary classification metrics using torchmetrics:
    Accuracy, Sensitivity, Specificity, F1, MCC, AUC, and Confusion Matrix.

    Args:
        model (nn.Module): Trained model (output dim = 1).
        dataloader (DataLoader): DataLoader for dataset evaluation.
        device (torch.device): CPU or GPU device.
        latents_mean, latents_std: Normalization parameters.

    Returns:
        dict: Metrics dictionary.
    """
    model.eval()

    accuracy_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    sensitivity_metric = torchmetrics.classification.BinaryRecall().to(device)
    specificity_metric = torchmetrics.classification.BinarySpecificity().to(device)
    f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    mcc_metric = torchmetrics.classification.BinaryMatthewsCorrCoef().to(device)
    auc_metric = torchmetrics.classification.BinaryAUROC().to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Normalize input
            x_batch = normalize(x_batch, latents_mean, latents_std)

            # Compute logits and probabilities
            logits = model(x_batch).view(-1)
            probs = torch.sigmoid(logits)

            accuracy_metric.update(probs, y_batch)
            sensitivity_metric.update(probs, y_batch)
            specificity_metric.update(probs, y_batch)
            f1_metric.update(probs, y_batch)
            mcc_metric.update(probs, y_batch)
            auc_metric.update(probs, y_batch)

            preds = (probs > 0.5).long()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        'accuracy': accuracy_metric.compute().item(),
        'sensitivity': sensitivity_metric.compute().item(),
        'specificity': specificity_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'mcc': mcc_metric.compute().item(),
        'auc': auc_metric.compute().item(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

    return metrics


def normalize(z, mean, std):
    return (z - mean) / std


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, latents_mean, latents_std):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        # Ensure target is of shape [batch_size, 1]
        batch_y = batch_y.to(device).float().unsqueeze(1)

        batch_x = normalize(batch_x, latents_mean, latents_std)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device, latents_mean, latents_std, is_regression=False, age_min=None, age_max=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().unsqueeze(1)

            batch_x = normalize(batch_x, latents_mean, latents_std)
            outputs = model(batch_x)

            if is_regression:
                preds = outputs
            else:
                preds = (torch.sigmoid(outputs) > 0.5).long()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())

    if is_regression:
        # Denormalize predictions and labels
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        preds_denorm = all_preds * (age_max - age_min) + age_min
        labels_denorm = all_labels * (age_max - age_min) + age_min
        mae = mean_absolute_error(labels_denorm, preds_denorm)
        rmse = np.sqrt(mean_squared_error(labels_denorm, preds_denorm))
        return mae, rmse, preds_denorm, labels_denorm
    else:
        accuracy = accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        return accuracy, mcc, all_preds, all_labels


def main(train_on_age=False, skip_training=False, epochs=1000, suffix="ldae", train_data_path=None, test_data_path=None, train_on_baseline_sessions=False, embeddings_dim=768, target_classes=[0, 1]):
    # Set to True for age regression, False for disease classification.
    if train_on_age:
        print("------ Training age regressor ------")
    else:
        print("------ Training disease classifier ------")

    # ------------------------------
    # Load the data from disk
    # ------------------------------
    train_data = torch.load(train_data_path,
        map_location='cpu')
    test_data = torch.load(
        test_data_path,
        map_location='cpu')

    # Extract arrays from training data
    train_embeddings = train_data['embeddings'].numpy()
    train_labels = train_data['labels'].numpy()
    train_ages = train_data['age'].numpy()

    # For safety, align the training arrays by taking the minimum length
    n_train = min(len(train_embeddings), len(train_labels), len(train_ages))
    if len(train_embeddings) != n_train or len(train_labels) != n_train or len(train_ages) != n_train:
        print(f"Warning: mismatched training array lengths; using the first {n_train} samples for all arrays.")
        train_embeddings = train_embeddings[:n_train]
        train_labels = train_labels[:n_train]
        train_ages = train_ages[:n_train]

    # Similarly, align the test arrays
    test_embeddings = test_data['embeddings'].numpy()
    test_labels = test_data['labels'].numpy()
    test_ages = test_data['age'].numpy()
    n_test = min(len(test_embeddings), len(test_labels), len(test_ages))
    if len(test_embeddings) != n_test or len(test_labels) != n_test or len(test_ages) != n_test:
        print(f"Warning: mismatched test array lengths; using the first {n_test} samples for all arrays.")
        test_embeddings = test_embeddings[:n_test]
        test_labels = test_labels[:n_test]
        test_ages = test_ages[:n_test]

    if train_on_baseline_sessions and not train_on_age:
        baseline_indices = [i for i, session in enumerate(train_data['session_number']) if session == 0]
        train_embeddings = train_embeddings[baseline_indices]
        train_labels = train_labels[baseline_indices]
        train_ages = train_ages[baseline_indices]

        baseline_indices = [i for i, session in enumerate(test_data['session_number']) if session == 0]
        test_embeddings = test_embeddings[baseline_indices]
        test_labels = test_labels[baseline_indices]
        test_ages = test_ages[baseline_indices]

    # ------------------------------
    # Split training data into train and validation using sklearn
    # ------------------------------
    # Note: we split embeddings, labels, and ages to keep them aligned.
    (train_embeddings, val_embeddings,
     train_labels, val_labels,
     train_ages, val_ages) = train_test_split(
        train_embeddings, train_labels, train_ages,
        test_size=0.3, random_state=42,
    )

    # ------------------------------
    # Optionally filter data based on target classes.
    # In the classification branch we filter for specific classes (e.g. [0, 1]).
    # In the age regression branch you may choose to use all samples.
    # ------------------------------
    if not train_on_age:
        # Filter training data
        train_mask = np.isin(train_labels, target_classes)
        train_embeddings = train_embeddings[train_mask]
        train_labels = train_labels[train_mask]
        train_ages = train_ages[train_mask]  # Keeping ages aligned even if not used
        # Filter validation data
        val_mask = np.isin(val_labels, target_classes)
        val_embeddings = val_embeddings[val_mask]
        val_labels = val_labels[val_mask]
        val_ages = val_ages[val_mask]
        # Filter test data
        test_mask = np.isin(test_labels, target_classes)
        test_embeddings = test_embeddings[test_mask]
        test_labels = test_labels[test_mask]
        test_ages = test_ages[test_mask]

    # ------------------------------
    # Print distributions and lengths before NaN removal
    # ------------------------------
    if not train_on_age:
        print("Training data distribution:")
        for cls in target_classes:
            print(f"  Class {cls}: {np.sum(train_labels == cls)} samples")
        print("Validation data distribution:")
        for cls in target_classes:
            print(f"  Class {cls}: {np.sum(val_labels == cls)} samples")
        print("Test data distribution:")
        for cls in target_classes:
            print(f"  Class {cls}: {np.sum(test_labels == cls)} samples")
    else:
        print(f"Training data count (for age regression): {len(train_embeddings)} samples")
        print(f"Validation data count (for age regression): {len(val_embeddings)} samples")
        print(f"Test data count (for age regression): {len(test_embeddings)} samples")

    print(
        f"Length of the training data before removing NaN values: {(len(train_embeddings), len(train_labels), len(train_ages))}")
    print(
        f"Length of the validation data before removing NaN values: {(len(val_embeddings), len(val_labels), len(val_ages))}")
    print(
        f"Length of the test data before removing NaN values: {(len(test_embeddings), len(test_labels), len(test_ages))}")

    # ------------------------------
    # Remove samples with NaN values in the ages
    # ------------------------------
    if np.isnan(train_ages).any():
        print("Found NaN values in training ages; removing corresponding samples.")
        nan_indices = np.argwhere(np.isnan(train_ages)).flatten()
        train_embeddings = np.delete(train_embeddings, nan_indices, axis=0)
        train_labels = np.delete(train_labels, nan_indices, axis=0)
        train_ages = np.delete(train_ages, nan_indices, axis=0)

    if np.isnan(val_ages).any():
        print("Found NaN values in validation ages; removing corresponding samples.")
        nan_indices = np.argwhere(np.isnan(val_ages)).flatten()
        val_embeddings = np.delete(val_embeddings, nan_indices, axis=0)
        val_labels = np.delete(val_labels, nan_indices, axis=0)
        val_ages = np.delete(val_ages, nan_indices, axis=0)

    if np.isnan(test_ages).any():
        print("Found NaN values in test ages; removing corresponding samples.")
        nan_indices = np.argwhere(np.isnan(test_ages)).flatten()
        test_embeddings = np.delete(test_embeddings, nan_indices, axis=0)
        test_labels = np.delete(test_labels, nan_indices, axis=0)
        test_ages = np.delete(test_ages, nan_indices, axis=0)

    print(
        f"Length of the training data after removing NaN values: {(len(train_embeddings), len(train_labels), len(train_ages))}")
    print(
        f"Length of the validation data after removing NaN values: {(len(val_embeddings), len(val_labels), len(val_ages))}")
    print(
        f"Length of the test data after removing NaN values: {(len(test_embeddings), len(test_labels), len(test_ages))}")

    # ------------------------------
    # Device Setup
    # ------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ------------------------------
    # Task-Specific Dataset Preparation
    # ------------------------------
    if train_on_age:
        # For age regression, we normalize the ages using the global min and max
        global_age_min = np.min(np.concatenate([train_ages, val_ages, test_ages]))
        global_age_max = np.max(np.concatenate([train_ages, val_ages, test_ages]))
        train_ages_norm = (train_ages - global_age_min) / (global_age_max - global_age_min)
        val_ages_norm = (val_ages - global_age_min) / (global_age_max - global_age_min)
        test_ages_norm = (test_ages - global_age_min) / (global_age_max - global_age_min)
        # Save age normalization parameters
        torch.save(torch.tensor(global_age_min), f"linear_probe/normalization/{suffix}_age_min.pt")
        torch.save(torch.tensor(global_age_max), f"linear_probe/normalization/{suffix}_age_max.pt")
        # Create datasets (target is age)
        train_dataset = TensorDataset(torch.from_numpy(train_embeddings).float(),
                                      torch.from_numpy(train_ages_norm).float())
        val_dataset = TensorDataset(torch.from_numpy(val_embeddings).float(),
                                    torch.from_numpy(val_ages_norm).float())
        test_dataset = TensorDataset(torch.from_numpy(test_embeddings).float(),
                                     torch.from_numpy(test_ages_norm).float())
    else:
        # For classification, use the labels.
        # Map the labels to 0, 1, ... (if needed)
        unique_classes = np.unique(train_labels)
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        train_labels_mapped = np.array([class_mapping[label] for label in train_labels])
        val_labels_mapped = np.array([class_mapping[label] for label in val_labels])
        test_labels_mapped = np.array([class_mapping[label] for label in test_labels])
        # Create datasets
        train_dataset = TensorDataset(torch.from_numpy(train_embeddings).float(),
                                      torch.from_numpy(train_labels_mapped).long())
        val_dataset = TensorDataset(torch.from_numpy(val_embeddings).float(),
                                    torch.from_numpy(val_labels_mapped).long())
        test_dataset = TensorDataset(torch.from_numpy(test_embeddings).float(),
                                     torch.from_numpy(test_labels_mapped).long())

    postfix = "baseline" if train_on_baseline_sessions else "full"

    # ------------------------------
    # Compute Normalization Parameters for Embeddings
    # ------------------------------
    latents_mean = np.mean(train_embeddings, axis=0)
    latents_std = np.std(train_embeddings, axis=0)
    latents_mean = torch.from_numpy(latents_mean).float().to(device)
    latents_std = torch.from_numpy(latents_std).float().to(device)
    torch.save(latents_mean, f"linear_probe/normalization/{suffix}_semantic_latents_mean_{postfix}.pt")
    torch.save(latents_std, f"linear_probe/normalization/{suffix}_semantic_latents_std_{postfix}.pt")

    # ------------------------------
    # Create DataLoaders
    # ------------------------------
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------
    # Model, Optimizer, and Loss Setup
    # ------------------------------
    model = nn.Linear(embeddings_dim, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) if train_on_age else torch.optim.Adam(model.parameters(),
                                                                                                   lr=1e-3)
    loss_fn = nn.MSELoss() if train_on_age else nn.BCEWithLogitsLoss()

    # ------------------------------
    # Training Loop with Validation for Model Selection
    # ------------------------------
    if train_on_age:
        best_metric = float('inf')  # Lower RMSE is better.
        best_model_path = f"linear_probe/models/{suffix}_best_age_regressor.ckpt"
    else:
        best_metric = 0.0  # Higher MCC is better.
        best_model_path = f"linear_probe/models/{suffix}_best_disease_classifier{postfix}.ckpt"

    if not skip_training:
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, latents_mean, latents_std)

            # Evaluate on the validation set
            if train_on_age:
                train_mae, train_rmse, _, _ = evaluate(model, train_loader, device, latents_mean, latents_std,
                                                       is_regression=True, age_min=global_age_min,
                                                       age_max=global_age_max)
                val_mae, val_rmse, _, _ = evaluate(model, val_loader, device, latents_mean, latents_std,
                                                   is_regression=True, age_min=global_age_min, age_max=global_age_max)
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, "
                      f"Train MAE: {train_mae:.2f}, Train RMSE: {train_rmse:.2f}, "
                      f"Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}")
                current_metric = val_rmse  # Lower is better.
            else:
                train_acc, train_mcc, _, _ = evaluate(model, train_loader, device, latents_mean, latents_std)
                val_acc, val_mcc, _, _ = evaluate(model, val_loader, device, latents_mean, latents_std)
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Train MCC: {train_mcc:.4f}, "
                      f"Val Acc: {val_acc:.4f}, Val MCC: {val_mcc:.4f}")
                current_metric = val_mcc  # Higher is better.

            # Save best model based on the validation metric.
            if (train_on_age and current_metric < best_metric) or (not train_on_age and current_metric > best_metric):
                best_metric = current_metric
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved as {best_model_path}")

    # ------------------------------
    # Testing: Load the Best Model and Evaluate on the Test Set
    # ------------------------------
    print("\n----- Testing Best Model -----")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from {best_model_path}")

    if train_on_age:
        test_mae, test_rmse, _, _ = evaluate(model, test_loader, device, latents_mean, latents_std,
                                             is_regression=True, age_min=global_age_min, age_max=global_age_max)
        print(f"Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
    else:
        test_acc, test_mcc, _, _ = evaluate(model, test_loader, device, latents_mean, latents_std)
        print(f"Test Acc: {test_acc:.4f}, Test MCC: {test_mcc:.4f}")

        metrics = compute_binary_classification_metrics(model, test_loader, device, latents_mean, latents_std)

        print("----- Binary Classification Metrics -----")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Sensitivity:         {metrics['sensitivity']:.4f}")
        print(f"Specificity:            {metrics['specificity']:.4f}")
        print(f"F1-score:          {metrics['f1']:.4f}")
        print(f"MCC:               {metrics['mcc']:.4f}")
        print(f"ROC AUC:           {metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


if __name__ == "__main__":
    train_data_path = "<path_to_your_training_data>"
    test_data_path = "<path_to_your_test_data>"
    # Set to True for age regression or False for disease classification.
    main(train_on_age=False, skip_training=False, epochs=200, suffix="ldae_ad_vs_cn",
         train_data_path=train_data_path, test_data_path=test_data_path, train_on_baseline_sessions=False, embeddings_dim=768, target_classes=[0, 1])
