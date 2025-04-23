from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from training_config import BATCH_SIZE, EPOCHS, MLFLOW_TRACKING_URI
import mlflow
import seaborn as sns

client = MlflowClient()



def calculate_inference_goodness(run):
    return np.sqrt((run['metrics.entropy gap second moment'] - (run['metrics.entropy gap--m_p']**2) ) / (79 * BATCH_SIZE))


def plot_inference_goodness(df):
    plot_data = []
    # print(df.columns)
    # exit(0)
    for index, row in df.iterrows():
        try:
            lam = float(row['params.LAMBDA'])
            ig = calculate_inference_goodness(row)
            entropy_gap = row['metrics.entropy gap--m_p']
            run_dict = {
                        'lambda': lam, 
                        'entropy gap': entropy_gap,
                        'inference goodness': ig
                        }
            plot_data.append(run_dict)
        except Exception as e:
            print(f"Skipping run {row['run_id']} due to error: {e}")
            continue

    plot_df = pd.DataFrame(plot_data)
    print(plot_df)
    plot_df = plot_df.sort_values(by='lambda')

    plt.figure(figsize=(8, 6))
    sns.barplot(x=plot_df["lambda"], y=plot_df["entropy gap"], data=plot_df, palette="Blues_d")
    plt.errorbar(
        x=np.arange(len(plot_df['lambda'])),
        y=plot_df['entropy gap'],
        yerr=plot_df['inference goodness'],
        fmt='none',
        capsize=5,
        ecolor='black',
        elinewidth=1,
    )

    plt.xlabel("Lambda")
    plt.ylabel("Inference Goodness with MCSE")
    plt.title("Inference Goodness vs Lambda (SVAE)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    # plt.xticks(rotation=45)
    plt.ylim(-1600, -1400)
    plt.tight_layout()
    plt.savefig('svae/plots/lambda_v_inference_goodness_mcse.png')
    plt.show()


def main():
    experiment_name = "LitSVAE_inference"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    # tag_filter = "tags.stage = 'testing inference--entropy gap'"
    # runs_df = mlflow.search_runs(experiment_ids=experiment_id, filter_string=tag_filter)
    runs_df = mlflow.search_runs(experiment_ids=experiment_id)
    # print(runs_df['metrics.Test ELBO'])

    plot_inference_goodness(runs_df)

if __name__ == "__main__":
    main()
