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
    USER_INPUT_LOGVAR,
    LAMBDA,
    N_FORWARD_PASS,
    NUMBER_OF_NEAREST_NEIGHBORS
)
from stochastic_vae import Stochastic_VAE
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_density_network import Stochastic_Density_NN
from pathlib import Path
from argparse import ArgumentParser
import torch
import mlflow



def main():
    ################
    ## Data setup ##
    ################

    train_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    #################
    ## Model setup ##
    #################

    ablate_entropy = args.ablate_entropy    # Ablation Study part 1: Ablate Entropy 
    ablate_fim = args.ablate_fim            # Ablation Study part 2: Ablate Fischer Information Matrix

    svae = Stochastic_VAE(
        Stochastic_Recognition_NN(input_dim=784, z_dim=LATENT_DIM, user_input_logvar=USER_INPUT_LOGVAR),
        Stochastic_Density_NN(input_dim=784, z_dim=LATENT_DIM),
        k_neighbor=NUMBER_OF_NEAREST_NEIGHBORS,
        n_forward=N_FORWARD_PASS,
        ablate_entropy=ablate_entropy,
        ablate_fim=ablate_fim
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
            "USER_INPUT_LOGVAR": USER_INPUT_LOGVAR,
            "LAMBDA" : LAMBDA,
            "N_FORWARD_PASS" : N_FORWARD_PASS,
            "NUMBER_OF_NEAREST_NEIGHBORS" : NUMBER_OF_NEAREST_NEIGHBORS
        }
    )

    if args.train:
        trainer.fit(model=svae, train_dataloaders=train_loader, val_dataloaders=val_loader)


    # ablation study Part 2 --- freeze decoder
    if args.freeze_decoder:
        checkpoint = torch.load("603393962448548868/bc47b5faee3e4618aa8232ae44fb7980/checkpoints/epoch=999-step=469000.ckpt", map_location="cpu")

        # init_params = {k: p.detach().clone() for k, p in svae.named_parameters()}

        for name, param in svae.named_parameters():
            if 'decoder' in name:
                param.data = checkpoint["state_dict"][name]
                param.requires_grad = False

        trainer.fit(model=svae, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if args.test:      
        # TODO - change this checkppint loading to the best model dynamically if possible
        checkpoint = torch.load("603393962448548868/86612901fb1f45d1bc694a796628f854/checkpoints/epoch=999-step=469000.ckpt")
        test_svae = Stochastic_VAE(Stochastic_Recognition_NN(input_dim=784, z_dim=LATENT_DIM, user_input_logvar=USER_INPUT_LOGVAR),
                                Stochastic_Density_NN(input_dim=784, z_dim=LATENT_DIM),
                                k_neighbor=NUMBER_OF_NEAREST_NEIGHBORS,
                                n_forward=N_FORWARD_PASS)
        test_svae.load_state_dict(checkpoint["state_dict"])

        trainer.test(model=test_svae, dataloaders=val_loader)

    if args.infer_entropy_gap:
        # pretrained on Lambda = 1.1 trained with a frozen decoder
        # checkpoint = torch.load("603393962448548868/2db153191f05418bb4e5295945b085a3/checkpoints/epoch=999-step=469000.ckpt")

        # pretrained on Lambda = 1.3 trained with a frozen decoder
        # checkpoint = torch.load("603393962448548868/8653831daef54d948533d50a77029c4a/checkpoints/epoch=999-step=469000.ckpt")

        # pretrained on Lambda = 1.5 trained with a frozen decoder
        # checkpoint = torch.load("603393962448548868/dfb6769bfc4541adbb6a4fea6f77ec17/checkpoints/epoch=999-step=469000.ckpt")

        # pretrained on Lambda = 2 trained with a frozen decoder
        # checkpoint = torch.load("603393962448548868/d74ef05ac03a4aa493813ced22a7ae63/checkpoints/epoch=999-step=469000.ckpt")

        # pretrained on Lambda = 5 trained with a frozen decoder
        # checkpoint = torch.load("603393962448548868/3a4edbb246f54352b1e38a92de2b7848/checkpoints/epoch=999-step=469000.ckpt")

        # pretrained on Lambda = 10 trained with a frozen decoder
        # checkpoint = torch.load("603393962448548868/f24fee43e7df4b5caf4cf19077bfae2b/checkpoints/epoch=999-step=469000.ckpt")

        # pretrained on Lambda = 100 trained with a frozen decoder
        checkpoint = torch.load("603393962448548868/33995e5dfa3a493ab7675265f76e9e6f/checkpoints/epoch=999-step=469000.ckpt")

        mlflow.set_tags({
                        "stage": "testing_inference",
                        "data": "validation",
                        "author": "Shounak Desai",
                        "model": "Stochastic VAE",
                        "lambda": LAMBDA,
                        })
                        
        test_svae = Stochastic_VAE(Stochastic_Recognition_NN(input_dim=784, z_dim=LATENT_DIM, user_input_logvar=USER_INPUT_LOGVAR),
                                Stochastic_Density_NN(input_dim=784, z_dim=LATENT_DIM),
                                k_neighbor=NUMBER_OF_NEAREST_NEIGHBORS,
                                n_forward=N_FORWARD_PASS)
        test_svae.load_state_dict(checkpoint["state_dict"])
        trainer.test(model=test_svae, dataloaders=val_loader)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", default=False)
    parser.add_argument("--test", default=False)
    parser.add_argument("--freeze_decoder", default=False)
    parser.add_argument("--ablate_entropy", default=False)
    parser.add_argument("--ablate_fim", default=False)
    parser.add_argument("--infer_entropy_gap", default=False)
    args = parser.parse_args()
    main()
