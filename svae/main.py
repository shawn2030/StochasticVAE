import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import lightning as lit
import mlflow
import torch
import torchvision.datasets as datasets
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from stochastic_density_network import Stochastic_Density_NN
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_vae import Stochastic_VAE


torch.set_float32_matmul_precision("high")
ENCODER_PLAN = [500, 300, 200, 100, 50]
DECODER_PLAN = [50, 100, 300, 500]
MLFLOW_TRACKING_URI = "/data/projects/SVAE/mlruns/"
MLFLOW_EXPERIMENT = "LitSVAE_RDL"
DATA_ROOT = "/data/datasets/"


def main(
    latent_dim: int = 20,
    lambda_: float = 2.0,
    number_of_nearest_neighbors: int = 4,
    n_forward_pass: int = 8,
    learning_rate: float = 1e-5,
    epochs: int = 200,
    batch_size: int = 250,
    user_input_logvar: float = -10,
    ablate_entropy: bool = False,
    ablate_fim: bool = False,
    load_decoder_from_run: str = None,
):
    ################
    ## Data setup ##
    ################

    train_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=True, transform=transforms.ToTensor()
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(123456)
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    #################
    ## Model setup ##
    #################

    svae = Stochastic_VAE(
        Stochastic_Recognition_NN(
            input_dim=784,
            latent_dim=latent_dim,
            user_input_logvar=user_input_logvar,
            plan=ENCODER_PLAN,
        ),
        Stochastic_Density_NN(
            input_dim=784,
            latent_dim=latent_dim,
            plan=DECODER_PLAN,
        ),
        lambda_=lambda_,
        lr=learning_rate,
        k_neighbor=number_of_nearest_neighbors,
        n_forward=n_forward_pass,
        ablate_entropy=ablate_entropy,
        ablate_fim=ablate_fim,
    )
    #####################
    ## Lightning setup ##
    #####################

    logger = MLFlowLogger(
        experiment_name=MLFLOW_EXPERIMENT,
        tracking_uri=MLFLOW_TRACKING_URI,
        log_model=True,
    )
    trainer = lit.Trainer(
        logger=logger,
        max_epochs=epochs,
        default_root_dir=logger.root_dir,  # TODO - double check that logger.root_dir is right
    )

    ############
    ## Run it ##
    ############

    logger.log_hyperparams(
        {
            "encoder_plan": ENCODER_PLAN,
            "decoder_plan": DECODER_PLAN,
            "latent_dim": latent_dim,
            "lambda_": lambda_,
            "number_of_nearest_neighbors": number_of_nearest_neighbors,
            "n_forward_pass": n_forward_pass,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "user_input_logvar": user_input_logvar,
            "ablate_entropy": ablate_entropy,
            "ablate_fim": ablate_fim,
            "decoder_source": load_decoder_from_run,
        }
    )

    # If specified, load decoder weights from a checkpoint and freeze it
    if load_decoder_from_run:
        # TODO - programmatically load the *best* checkpoint from the run instead of hardcoding
        #  the path
        checkpoint = torch.load(
            mlflow.artifacts.download_artifacts(
                run_id=load_decoder_from_run,
                artifact_path="model/checkpoints/epoch=199-step=48000/epoch=199-step=48000.ckpt",
            )
        )

        decoder_state_dict = {
            k.split(".", 1)[1]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("decoder")
        }

        svae.decoder.load_state_dict(decoder_state_dict)
        for param in svae.decoder.parameters():
            param.requires_grad = False

    # Do training
    trainer.fit(model=svae, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Do testing (including inference-goodness)
    trainer.test(model=svae, dataloaders=test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", type= int, default= 20),
    parser.add_argument("--lambda", dest="lambda_", type= float, default= 2.0),
    parser.add_argument("--number_of_nearest_neighbors", type= int, default= 4),
    parser.add_argument("--n_forward_pass", type= int, default= 8),
    parser.add_argument("--learning_rate", type= float, default= 1e-5),
    parser.add_argument("--epochs", type= int, default= 200),
    parser.add_argument("--batch_size", type= int, default= 250),
    parser.add_argument("--user_input_logvar", type= float, default= -10),
    parser.add_argument("--ablate_entropy", type= bool, default= False),
    parser.add_argument("--ablate_fim", type= bool, default= False),
    parser.add_argument("--load_decoder_from_run", type= str, default= None),
    args = parser.parse_args()

    # Only let lightning 'see' one GPU, but can be overridden by setting the environment variable
    # CUDA_VISIBLE_DEVICES from outside the script.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

    main(**vars(args))
