import wandb

class Logger:

    def __init__(self, config):
        self.config = config

        wandb.init(project="video-generation", name=config["logging"]["run_name"], config=config)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def get_wandb(self):
        return wandb

