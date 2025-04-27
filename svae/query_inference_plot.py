from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from nn_lib.utils import search_runs_by_params
from torchvision import datasets, transforms

from main import DATA_ROOT

client = MlflowClient()


def barplot_with_custom_errors(data, x, y, yerr, **kwargs):
    data_low = data.copy()
    data_hi = data.copy()
    data_low[y] = data[y] - data[yerr]
    data_hi[y] = data[y] + data[yerr]
    data_combo = pd.concat([data_low, data, data_hi], axis=0).reset_index()

    def calculate_errors(low_mid_hi):
        return np.min(low_mid_hi), np.max(low_mid_hi)

    return sns.barplot(data_combo, x=x, y=y, errorbar=calculate_errors, **kwargs)


def plot_inference_goodness(plot_df):
    ds = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
    )

    plot_df["goodness_standard_error"] = np.sqrt(
        (plot_df["metrics.goodness_moment2"] - (plot_df["metrics.goodness_moment1"] ** 2)) / len(ds)
    )
    plot_df["goodness_relative"] = plot_df["metrics.goodness_moment1"] - plot_df[plot_df["params.lambda_"] == "inf"]["metrics.goodness_moment1"].values[0]
    plot_df["params.lambda_"] = plot_df["params.lambda_"].astype(float)
    plot_df["params.user_input_logvar_f"] = plot_df["params.user_input_logvar"].astype(float)
    plot_df = plot_df.sort_values(["params.user_input_logvar_f", "params.lambda_"])

    plt.figure(figsize=(8, 6))
    barplot_with_custom_errors(
        data=plot_df,
        x="params.lambda_",
        y="goodness_relative",
        yerr="goodness_standard_error",
        hue="params.user_input_logvar",
    )
    plt.xlabel("Lambda")
    plt.ylabel("$KL(m(z|x)||p(z|x)) + C$")
    plt.title("Qualitity of Inference")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/lambda_v_inference_goodness_mcse.png", dpi=300)


def main():
    mlflow.set_tracking_uri("/data/projects/SVAE/mlruns")
    runs_df = search_runs_by_params(
        experiment_name="LitSVAE_RDL",
        params={
            "decoder_source": "37abd9dfafa647ecbdf484d76a04f169",
            # "user_input_logvar": "-10.0",
        },
        finished_only=True,
    )

    plot_inference_goodness(runs_df)


if __name__ == "__main__":
    main()
