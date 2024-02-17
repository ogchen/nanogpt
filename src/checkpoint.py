import os
import torch


class CheckpointManager:
    def __init__(self, model, optimizer, statefile):
        self.model = model
        self.optimizer = optimizer
        self.statefile = statefile
        self.epoch = 0

    def save(self):
        torch.save(
            {
                "iteration": self.epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            self.statefile,
        )

    def load(self):
        if not os.path.isfile(self.statefile):
            return
        checkpoint = torch.load(self.statefile)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epoch = checkpoint["iteration"]
        print(f"Loaded state file from {self.statefile}, iteration: {self.epoch}")
