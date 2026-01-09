import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import json
from pathlib import Path
import wandb

class Trainer:
    """
    Trainer class for managing model training and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device | None = None,
        save_path: str = "best_model.pth"
    ):
        """
        Initialize the Trainer.

        Args:
            model: PyTorch model to train
            train_dataloader: Training dataloader
            test_dataloader: Test/validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            save_path: Path to save best model checkpoint
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device("cpu")
        self.save_path = save_path

        # Create save directory if it doesn't exist
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        self.best_acc = 0.0
        self.run = wandb.init(
            entity="camwheeler135-university-of-edinburgh",
            project="k8s_tutorial",
            config={
                "model": self.model,
                "criterion": self.criterion,
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "dataset": "CIFAR10"                
            }
        )

    def train_epoch(self) -> tuple[float, float]:
        """
        Train the model for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_dataloader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def evaluate(self) -> tuple[float, float]:
        """
        Evaluate the model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.test_dataloader, desc="Evaluating")

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        epoch_loss = running_loss / len(self.test_dataloader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            torch.save(checkpoint, self.save_path)
            print(f"Model saved to {self.save_path} with accuracy: {self.best_acc:.2f}%")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.history = checkpoint.get('history', {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        })

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get('epoch', 0)

    def train(self, epochs: int, start_epoch: int = 0):
        """
        Full training loop.

        Args:
            epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming training)
        """
        print(f"\nStarting training for {epochs} epochs...")

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)

            # Send the data to wandb
            self.run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
                })

            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint(epoch + 1, is_best=True)

        print(f"\nTraining complete! Best accuracy: {self.best_acc:.2f}%")

        # Save training history to JSON file
        history_path = Path(self.save_path).parent / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")

        # Training is done, we need to clean up the wandb stuff. 
        self.run.finish()
        return self.history

    def get_history(self) -> Dict[str, list]:
        """
        Get training history.

        Returns:
            Dictionary containing training history
        """
        return self.history
