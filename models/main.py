from trainer import Trainer
from config_maker import get_config

def run_training():
    config = get_config()
    trainer = Trainer(config)
    trainer.train()

def run_testing():
    config = get_config()
    trainer = Trainer(config)
    trainer.test()

if __name__ == '__main__':
    run_training()
    run_testing()
