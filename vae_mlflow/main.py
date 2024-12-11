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
from training_config import NUM_EPOCHS, BATCH_SIZE, LATENT_DIM, LR_RATE, DEVICE
from recognition_net import RecognitionModel
from densitynet import DensityNet
import mlflow

mlflow.set_experiment("VAE")


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
            mu, sigma = vae.encoder(images[d].view(1, 784))

            for i in range(num_examples):
                z = vae.reparameterize(mu, sigma)
                output = vae.decoder(z)
                output = output.view(-1, 1, 28, 28)
                save_image(output, f"vae_mlflow/output/generated_{d}_ex_{i}.png")


def main():
    dataset = datasets.MNIST(
        root="dataset/",
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor()
                # transforms.Normalize((0.5,), (0.5,))
            ]
        ),
        download=True,
    )
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=64, pin_memory=True)

    vae = VAE(RecognitionModel(LATENT_DIM), DensityNet(LATENT_DIM))
    vae.parameters()  # [e.fc1, e.fc21, e.fc22, d.fc3, d.fc4, d.logvar]

    optim_vae = torch.optim.Adam(vae.parameters(), lr=LR_RATE)
    run_name = "VAE"
    with mlflow.start_run(run_name=run_name) as run:

        vae = train_model(train_loader, vae, optim_vae, DEVICE, NUM_EPOCHS)

    test_VAE(dataset, vae, DEVICE, num_examples=2)


if __name__ == "__main__":
    main()
