import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from training_config import EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE, LATENT_DIM, PLAN
from stochastic_vae import Stochastic_VAE
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_density_network import Stochastic_Density_NN
import mlflow


def train(train_loader, model, optimizer, device, num_epochs):
    model.to(device)
    model.train()
    losses = []
    steps = 0
    encoder_num_layers = len(PLAN)

    mlflow.log_param("Learning Rate", LEARNING_RATE)
    mlflow.log_param("Epochs", EPOCHS)
    mlflow.log_param("Batch Size", BATCH_SIZE)
    mlflow.log_param("Latent Space size", LATENT_DIM)
    mlflow.log_param("Number of layers", encoder_num_layers)

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            steps += 1
            data = data.view(-1, 784).to(device)

            optimizer.zero_grad()
            loss, kl_term, reconstruction_loss = model.loss(data)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mlflow.log_metric("KL Divergence term", kl_term, step=steps)
            mlflow.log_metric("Reconstruction log likelihood", reconstruction_loss, step=steps)

            for name, param in model.named_parameters():
                if 'encoder' in name:
                    if 'weight' in name:
                        mlflow.log_metric(f"{name}_mean", param.data.mean().item(), step=steps)
                        mlflow.log_metric(f"{name}_std", param.data.std().item(), step=steps)
                    if 'bias' in name:
                        mlflow.log_metric(f"{name}_mean", param.data.mean().item(), step=steps)
                        mlflow.log_metric(f"{name}_std", param.data.std().item(), step=steps)

        mlflow.log_metric("ELBO", -total_loss, step=epoch)

    return model



def test():
    pass

def main():
    dataset = datasets.MNIST(root='dataset/', train=True, transform = transforms.Compose([
                                transforms.ToTensor()
                                # transforms.Normalize((0.5,), (0.5,))
                            ]), download=True)
    train_loader = DataLoader(dataset=dataset, batch_size= BATCH_SIZE, shuffle=True)

    svae = Stochastic_VAE(Stochastic_Recognition_NN(input_dim=784, z_dim=LATENT_DIM), 
                          Stochastic_Density_NN(input_dim=784,z_dim=LATENT_DIM))
    optim_svae = torch.optim.Adam(svae.parameters(), lr=LEARNING_RATE)
    run_name = 'SVAE'

    with mlflow.start_run(run_name=run_name) as run:
        svae = train(train_loader, svae, optim_svae, DEVICE, EPOCHS)


if __name__ == "__main__":
    main()