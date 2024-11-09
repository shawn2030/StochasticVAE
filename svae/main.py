import lightning as lit
from lightning.pytorch.loggers import MLFlowLogger
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from training_config import (
    EPOCHS,
    BATCH_SIZE,
    LATENT_DIM,
    MLFLOW_TRACKING_URI,
    DATA_ROOT,
    PLAN,
    PLAN_DECODER,
    LEARNING_RATE,
    USER_INPUT_LOGVAR
)
from stochastic_vae import Stochastic_VAE
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_density_network import Stochastic_Density_NN
from pathlib import Path


def main():
    ################
    ## Data setup ##
    ################

    train_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #################
    ## Model setup ##
    #################

    svae = Stochastic_VAE(
        Stochastic_Recognition_NN(input_dim=784, z_dim=LATENT_DIM, user_input_logvar=USER_INPUT_LOGVAR),
        Stochastic_Density_NN(input_dim=784, z_dim=LATENT_DIM),
    )

    #####################
    ## Lightning setup ##
    #####################

    logger = MLFlowLogger(
        experiment_name="LitSVAE",
        tracking_uri=MLFLOW_TRACKING_URI,
        log_model=True,
    )
    trainer = lit.Trainer(
        logger=logger,
        max_epochs=EPOCHS,
        default_root_dir=logger.root_dir,  # TODO - double check that logger.root_dir is right
    )

    ############
    ## Run it ##
    ############

    logger.log_hyperparams(
        {
            "PLAN": PLAN,
            "PLAN_DECODER": PLAN_DECODER,
            "LATENT_DIM": LATENT_DIM,
            "LEARNING_RATE": LEARNING_RATE,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "USER_INPUT_LOGVAR": USER_INPUT_LOGVAR
        }
    )
    trainer.fit(model=svae, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
