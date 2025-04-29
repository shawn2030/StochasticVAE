from pathlib import Path

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms, io
from torchvision.utils import make_grid

from main import ENCODER_PLAN, DECODER_PLAN, MLFLOW_TRACKING_URI, DATA_ROOT
from stochastic_density_network import Stochastic_Density_NN
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_vae import Stochastic_VAE

params = {
    "latent_dim": 20,
    "user_input_logvar": -np.inf,
    "lambda_": 2.0,
    "learning_rate": 1e-3,
    "number_of_nearest_neighbors": 4,
    "n_forward_pass": 8,
    "ablate_entropy": False,
    "ablate_fim": False,
}

svae = Stochastic_VAE(
        Stochastic_Recognition_NN(
            input_dim=784,
            latent_dim=params["latent_dim"],
            user_input_logvar=params["user_input_logvar"],
            plan=ENCODER_PLAN,
        ),
        Stochastic_Density_NN(
            input_dim=784,
            latent_dim=params["latent_dim"],
            plan=DECODER_PLAN,
        ),
        lambda_=params["lambda_"],
        lr=params["learning_rate"],
        k_neighbor=params["number_of_nearest_neighbors"],
        n_forward=params["n_forward_pass"],
        ablate_entropy=params["ablate_entropy"],
        ablate_fim=params["ablate_fim"],
    )


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
reference_run = mlflow.get_run("37abd9dfafa647ecbdf484d76a04f169")

#%%

checkpoint = torch.load(
    mlflow.artifacts.download_artifacts(
        run_id=reference_run.info.run_id,
        artifact_path="model/checkpoints/epoch=99-step=20000/epoch=99-step=20000.ckpt",
    ), map_location=torch.device("cuda")
)
svae.load_state_dict(checkpoint["state_dict"])

test_dataset = datasets.MNIST(
    root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=100,
    num_workers=0,
    shuffle=True,
)

# Make some reconstructions to sanity-check the model was loaded
svae.eval().to("cuda")
x, y = next(iter(test_loader))
x, y = x.to("cuda"), y.to("cuda")
# with torch.no_grad():
#     (x, y) = next(iter(test_loader))
#     stuff = svae.loss(x.to("cuda"))
#     x_hat = stuff["reconstruction"].cpu()
#
#     x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=10)
#     recon_grid = make_grid(x_hat.view(-1, 1, 28, 28), nrow=10)
#
#     fig, ax = plt.subplots(1, 2, figsize=(8, 4))
#     ax[0].imshow(x_grid.permute(1, 2, 0).numpy())
#     ax[0].set_title("Original")
#     ax[0].axis("off")
#     ax[1].imshow(recon_grid.permute(1, 2, 0).numpy())
#     ax[1].set_title("Reconstruction")
#     ax[1].axis("off")
#     plt.show()

#%% Try different logvar values, measure output entropy for each one.

logvars = np.linspace(-15, -2, 20)
entropies = np.zeros(len(logvars))
with torch.no_grad():
    for i, logvar in enumerate(logvars):
        svae.deterministic = np.isinf(logvar)
        for name, p in svae.named_parameters():
            if "encoder" in name and "logvar" in name:
                p.data.fill_(logvar)

        stuff = svae.loss(x.to("cuda"))
        entropies[i] = stuff["entropy_term"].item()

#%%

plt.figure()
plt.plot(logvars, entropies)
plt.xlabel("logvar of weights/biases")
plt.ylabel("differential entropy of SNN output (up to constants)")
plt.show()