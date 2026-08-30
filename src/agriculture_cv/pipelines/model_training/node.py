import copy

import mlflow
import pandas as pd
import plotly.express as px
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from .dataset_creation import PlantVillageDataset, _get_transforms


def _train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct_preds += torch.sum(predicted == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def _eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            correct_preds += torch.sum(predicted == labels).item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def train_model_and_track_model(train_df, val_df, params: dict):
    mlflow.log_params(params)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, params["num_classes"])

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    train_transform, val_transform = _get_transforms(params["image_size"])

    train_dataset = PlantVillageDataset(train_df, transform=train_transform)
    val_dataset = PlantVillageDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(params["epochs"]):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            step=epoch,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="PlantVillage_ResNet18",
        serialization_format="pickle",
    )
    return model


def evaluate_model(
    trained_model_checkpoint: nn.Module,
    test_data: pd.DataFrame,
    params: dict,
    label_mapping: dict,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    trained_model_checkpoint.to(device)
    trained_model_checkpoint.eval()

    _, test_transform = _get_transforms(params["image_size"])
    test_dataset = PlantVillageDataset(test_data, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = trained_model_checkpoint(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    id_to_class = {i: name for i, name in enumerate(label_mapping)}
    class_names = [id_to_class[i] for i in sorted(id_to_class.keys())]

    report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )

    mlflow.log_metrics(
        {
            "test_accuracy": report["accuracy"],
            "test_f1_macro": report["macro avg"]["f1-score"],
            "test_f1_weighted": report["weighted avg"]["f1-score"],
        }
    )

    cm = confusion_matrix(all_targets, all_preds)

    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        labels=dict(x="Predito", y="Real", color="Quantidade"),
        title="Matriz de Confusão - Conjunto de Teste",
        text_auto=True,
        color_continuous_scale="Blues",
    )
    return report, fig
