import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vae import VAE
from training_config import NUM_EPOCHS, BATCH_SIZE, LATENT_DIM, LR_RATE, DEVICE, DATA_ROOT
from recognition_net import RecognitionModel
from densitynet import DensityNet
import mlflow
from pathlib import Path
from argparse import ArgumentParser

from stochastic_density_network import Stochastic_Density_NN



mlflow.set_experiment("LitSVAE_inference")
mlflow.set_tags({
                        "stage": "training inference",
                        "data": "validation",
                        "author": "Shounak Desai",
                        "model": "VAE",
                        })
                        


def train_model(train_loader, model, optimizer, device, num_epochs=NUM_EPOCHS):
    model.to(device)
    model.train()
    losses = []
    steps = 0
    encoder_num_layers = model.get_encoder_num_layers()

    mlflow.log_param("Learning Rate", LR_RATE)
    mlflow.log_param("Epochs", NUM_EPOCHS)
    mlflow.log_param("Batch Size", BATCH_SIZE)
    mlflow.log_param("Latent Space size", LATENT_DIM)
    mlflow.log_param("Number of layers", encoder_num_layers)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            steps += 1
            data = data.view(-1, 784).to(device)

            optimizer.zero_grad()
            loss, _, kl_term, reconstruction_loss = model.loss(data)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mlflow.log_metric("KL Divergence term", kl_term, step=steps)
            mlflow.log_metric("Reconstruction log likelihood", reconstruction_loss, step=steps)

            for name, param in model.named_parameters():
                if "encoder" in name:
                    if "weight" in name:
                        mlflow.log_metric(f"{name}_mean", param.data.mean().item(), step=steps)
                        mlflow.log_metric(f"{name}_std", param.data.std().item(), step=steps)
                    if "bias" in name:
                        mlflow.log_metric(f"{name}_mean", param.data.mean().item(), step=steps)
                        mlflow.log_metric(f"{name}_std", param.data.std().item(), step=steps)

        mlflow.log_metric("ELBO", -total_loss, step=epoch)

    return model


def test_VAE(test_loader, vae, device, num_examples=3):
    vae = vae.to("cpu")
    vae.eval()
    images = []
    label = 0
    images = []
    label = 0
    for x, y in test_loader:
        if label == 10:
            break

        if y == label:
            images.append(x)
            label += 1

    for d in range(10):
        with torch.no_grad():
            z = vae.encoder(images[d].view(-1, 784).to(device))
            z = z.view((-1, LATENT_DIM))
            output = vae.decoder(z) 
            # entropy_gap, secondmoment = vae.entropy_gap(images[d].view(1, 784))
            output = output.view(-1, 1, 28, 28)
            # save_image(output, f"vae_mlflow/output/generated_{d}_ex_{i}.png")


def calculate_entropy_gap(test_loader, vae, device):
    vae.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x = x.view(x.size(0), -1)  # Flatten
            elbo, _, _, _=  vae.loss(x)
            mlflow.log_metric("Test ELBO", elbo)



def main():
    train_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    vae = VAE(RecognitionModel(LATENT_DIM), Stochastic_Density_NN(input_dim=784, z_dim=LATENT_DIM))
    vae.parameters()  # [e.fc1, e.fc21, e.fc22, d.fc3, d.fc4, d.logvar]

    optim_vae = torch.optim.Adam(vae.parameters(), lr=LR_RATE)

    if args.train:
        if mlflow.active_run():
            mlflow.end_run()
        with mlflow.start_run(run_name="VAE") as run:
            vae = train_model(train_loader, vae, optim_vae, DEVICE, NUM_EPOCHS)
            
    if args.test:
        with mlflow.start_run(run_name="VAE") as run:
            test_VAE(val_loader, vae, DEVICE, num_examples=2)


    if args.inference_entropy:
        if mlflow.active_run():
            mlflow.end_run()
        checkpoint = torch.load("603393962448548868/bc47b5faee3e4618aa8232ae44fb7980/checkpoints/epoch=999-step=469000.ckpt", map_location="cpu")

        for name, param in vae.named_parameters():
            if 'decoder' in name:
                param.data = checkpoint["state_dict"][name]
                param.requires_grad = False
        
        with mlflow.start_run(run_name="VAE") as run:
            vae = train_model(train_loader, vae, optim_vae, DEVICE, NUM_EPOCHS)
            calculate_entropy_gap(val_loader, vae, DEVICE)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", default=False)
    parser.add_argument("--test", default=False)
    parser.add_argument("--inference_entropy", default=False)
    args = parser.parse_args()
    main()
