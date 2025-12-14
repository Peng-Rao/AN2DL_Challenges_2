import albumentations as A
import lightning as L
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from models import PathologyModel
from PathologyDataModule import PathologyDataModule

L.seed_everything(42)


class Config:
    # Data paths
    DATA_DIR = "./data"
    TRAIN_DATA_DIR = "./data/train_data"
    TEST_DATA_DIR = "./data/test_data"
    TRAIN_LABELS_PATH = "./data/train_labels.csv"
    # Class labels
    CLASSES = ["Luminal A", "Luminal B", "HER2(+)", "Triple negative"]
    NUM_CLASSES = 4


if __name__ == "__main__":
    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            # A.HueSaturationValue(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    datamodule = PathologyDataModule(
        train_data_dir=Config.TRAIN_DATA_DIR,
        test_data_dir=Config.TEST_DATA_DIR,
        train_labels_path=Config.TRAIN_LABELS_PATH,
        trash_list_path="./data/trash_list.txt",
        use_mask=True,
        use_patches=True,
        patch_size=224,
        num_patches=16,
        img_size=224,
        batch_size=1,
        min_annotation_pixels=1,
        train_transform=train_transform,
        val_transform=val_transform,
        classes=Config.CLASSES,
        num_workers=0,
    )
    model = PathologyModel(
        model_name="convnext_tiny",
        use_patches=True,
        patch_aggregation="clam",
        learning_rate=2e-4,
        weight_decay=1e-2,
        dropout_rate=0.3,
        label_smoothing=0.1,
        # freeze_backbone_epochs=5,
        optimizer_name="adamw",
        mixup_alpha=0.1,
    )

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                monitor="val/acc",
                mode="max",
                filename="{epoch:02d}-{val/acc:.4f}",
                save_weights_only=False,
            ),
            EarlyStopping(monitor="val/acc", patience=10, mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        accumulate_grad_batches=1,
        # gradient_clip_val=0.5,
        precision="16-mixed",
        log_every_n_steps=5,
        devices="auto",
        logger=None,
        overfit_batches=1,
    )
    trainer.fit(model=model, datamodule=datamodule, weights_only=False)
