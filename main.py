import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from datetime import datetime
import dotenv

from src.datasets import get_cifar10_dataloaders
from src.transforms import get_train_transforms, get_test_transforms
from src.models import model_factory
from src.trainer import Trainer

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to setup and run training."""

    # Load the env so we can log to wandb (in k8s we will set the api-key with secrets).
    dotenv.load_dotenv()

    print("Configuration:", flush=True)
    for key, value in cfg.items():
        print(f"  {key}: {value}", flush=True)

    # Set device
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # Get transforms
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    # Get dataloaders
    print("\nLoading CIFAR10 dataset...", flush=True)
    train_dataloader, test_dataloader = get_cifar10_dataloaders(
        data_dir=cfg["dataset_conf"]["data_dir"],
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        batch_size=cfg["dataset_conf"]["batch_size"],
        num_workers=cfg["dataset_conf"]["num_workers"]
    )

    print(f"Training samples: {len(train_dataloader.dataset)}") # type: ignore stops the typechecker from moaning. 
    print(f"Test samples: {len(test_dataloader.dataset)}") # type: ignore 

    print(f"\nCreating model: {cfg["model_conf"]['model_name']}")
    model = model_factory(
        **cfg["model_conf"]
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["trainer_conf"]["learning_rate"],
        weight_decay=cfg["trainer_conf"]["weight_decay"]
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["trainer_conf"]["epochs"]
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler, # type: ignore again moan moan moan.
        device=device,
        save_path=cfg["dataset_conf"]["save_path"]
    )

    history = trainer.train(epochs=cfg["trainer_conf"]["epochs"])

    print("\nTraining complete!")

    # Save final results summary
    results_dir = Path(cfg["dataset_conf"]["save_path"]).parent
    results = {
        "timestamp": datetime.now().isoformat(),
        "best_test_accuracy": trainer.best_acc,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_test_accuracy": history["test_acc"][-1],
        "total_epochs": cfg["trainer_conf"]["epochs"],
        "config": OmegaConf.to_container(cfg, resolve=True)
    }

    results_path = results_dir / "results_summary.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results summary saved to {results_path}")


if __name__ == "__main__":
    main()
