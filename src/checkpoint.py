import os
import torch


class CheckpointManager:
    def __init__(self, model, optimizer, statefile):
        self.model = model
        self.optimizer = optimizer
        self.statefile = statefile
        self.epoch = 0
        self.runid = ""

    def save(self):
        torch.save(
            {
                "iteration": self.epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "runid": self.runid,
            },
            self.statefile,
        )

    def load(self):
        if not os.path.isfile(self.statefile):
            return False
        checkpoint = torch.load(self.statefile)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epoch = checkpoint["iteration"]
        self.runid = checkpoint["runid"]
        print(f"Loaded state file from {self.statefile}, iteration: {self.epoch}")
        return True

    def log_loss(self, loss, epoch, flush=False):
        self.losses.append({"loss": loss, "epoch": epoch})
        self.writer.add_scalars("loss", loss, epoch)
        if flush:
            self.writer.flush()

    def log_output(self, output, epoch, flush=False):
        self.outputs.append({"output": output, "epoch": epoch})
        self.writer.add_text("output", output, epoch)
        if flush:
            self.writer.flush()
