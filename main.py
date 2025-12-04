import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision import transforms

from datamodule import KFoldDataModule, SubtypeDataset

plt.rcParams["pdf.fonttype"] = 42
plt.switch_backend("agg")


class LitClassifier(L.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, model_name="resnet34"):
        super().__init__()
        self.save_hyperparameters()

        # Use timm to create model
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.train_acc(preds, labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.val_acc(preds, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch[0]
        logits = self(images)
        probs = F.softmax(logits, dim=1)

        targets = batch[1]
        return probs, targets

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )


def plot_results(y_true, y_probs, class_names, prefix="val"):
    # 1. Classification Report
    y_pred = np.argmax(y_probs, axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\nClassification Report ({prefix}):\n{report}")
    with open(f"{prefix}_report.txt", "w") as f:
        f.write(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    with open(f"{prefix}_confusion_matrix.txt", "w") as f:
        print(cm, file=f)

    # 3. ROC Curve
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({prefix})")
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_roc.png")
    plt.close()


if __name__ == "__main__":
    TRAIN_DIR = "data/train_data"
    TEST_DIR = "data/test_data"
    CSV_FILE = "data/train_labels.csv"

    BATCH_SIZE = 16
    N_FOLDS = 5
    EPOCHS = 50
    MODEL_NAME = "vit_base_patch16_224"

    # --- Transforms ---
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Loading Dataset...")
    full_dataset = SubtypeDataset(
        img_dir=TRAIN_DIR,
        train_labels_path=CSV_FILE,
        transform=transform,
        mode="train",
    )

    class_names = [
        full_dataset.idx_to_label[i] for i in range(len(full_dataset.idx_to_label))
    ]
    print(f"Classes: {class_names}")

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_val_probs = []
    all_val_targets = []

    best_loss = float("inf")
    best_model_path = ""

    print(f"Starting {N_FOLDS}-Fold CV...")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\n=== FOLD {fold + 1}/{N_FOLDS} ===")

        dm = KFoldDataModule(full_dataset, train_ids, val_ids, batch_size=BATCH_SIZE)

        model = LitClassifier(
            num_classes=len(class_names), learning_rate=1e-4, model_name=MODEL_NAME
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename=f"fold_{fold + 1}_best",
            save_top_k=1,
            mode="min",
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode="min",
        )

        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=CSVLogger("logs", name=f"fold_{fold + 1}"),
            enable_progress_bar=True,
        )

        trainer.fit(model, dm)

        print(f"Best model for Fold {fold + 1}: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score < best_loss:
            best_loss = checkpoint_callback.best_model_score
            best_model_path = checkpoint_callback.best_model_path

        preds = trainer.predict(
            model, dataloaders=dm.predict_dataloader(), ckpt_path="best"
        )

        for batch_probs, batch_targets in preds:
            all_val_probs.append(batch_probs.cpu().numpy())
            all_val_targets.append(batch_targets.cpu().numpy())

    print("\nGenerating Cross-Validation Report...")
    all_val_probs = np.concatenate(all_val_probs)
    all_val_targets = np.concatenate(all_val_targets)

    plot_results(all_val_targets, all_val_probs, class_names, prefix="cv_aggregated")

    print(f"\nRunning Inference on Test Set using Best Model: {best_model_path}...")

    test_dataset = SubtypeDataset(
        img_dir=TEST_DIR,
        train_labels_path=None,
        transform=transform,
        mode="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_model = LitClassifier.load_from_checkpoint(best_model_path)
    trainer = L.Trainer(accelerator="auto", devices="auto", logger=False)

    test_preds = trainer.predict(best_model, dataloaders=test_loader)

    final_ids = []
    final_probs = []

    for batch_probs, batch_filenames in test_preds:
        final_probs.append(batch_probs.cpu().numpy())
        final_ids.extend(batch_filenames)

    final_probs = np.concatenate(final_probs, axis=0)
    final_pred_indices = np.argmax(final_probs, axis=1)
    final_pred_labels = [full_dataset.idx_to_label[i] for i in final_pred_indices]

    submission_df = pd.DataFrame(
        {"sample_index": final_ids, "label": final_pred_labels}
    )
    submission_df.to_csv("submission.csv", index=False)
    print("Inference Done! Saved to submission.csv")
    print(submission_df.head())
