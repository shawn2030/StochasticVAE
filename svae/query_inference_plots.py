from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

client = MlflowClient()

TEST_RUN_IDS = [
    "7ccff42562844d16a6d1b39a187add87",
    "3a4edbb246f54352b1e38a92de2b7848",
   "8813bbff603a40e4b81db0f1fab32565",
   "8653831daef54d948533d50a77029c4a",
   "fba78ae29a0c4c73be7b43250b3b81fe",
   "e47135a4f2f64277bed9c769249461bd",
   "8146197b392e4ee49ae37c44881e9241"
]

def logdata_2_df():

    # Fetch each run and convert to a DataFrame
    run_data = []
    for run_id in TEST_RUN_IDS:
        run = client.get_run(run_id)
        run_dict = {
            "run_id": run.info.run_id,
            **run.data.params,
            **run.data.metrics,
            **run.data.tags
        }
        run_data.append(run_dict)

    # Create DataFrame of selected runs
    runs_df = pd.DataFrame(run_data)

    # print(runs_df['entropy gap second moment'])
    return runs_df

def calculate_inference_goodness(run):
    return math.sqrt((run['entropy gap second moment'] - (run['entropy gap--m_p']**2) )/ 60000)


def plot_inference_goodness(df):
    # Store means and MCSEs per lambda
    lambda_to_rows = {}

    for index, row in df.iterrows():
        lam = row['LAMBDA']
        if lam not in lambda_to_rows:
            lambda_to_rows[lam] = []
        lambda_to_rows[lam].append(row)

    lambdas = []
    means = []
    mcse = []

    for lam, rows in lambda_to_rows.items():
        entropy_gaps = [r['entropy gap--m_p'] for r in rows]
        second_moments = [r['entropy gap second moment'] for r in rows]

        mean_gap = np.mean(entropy_gaps)
        second_moment_mean = np.mean(second_moments)
        mcse_gap = np.sqrt((second_moment_mean - mean_gap**2) / 60000)

        lambdas.append(lam)
        means.append(mean_gap)
        mcse.append(mcse_gap)

    # Sort by lambda
    sorted_data = sorted(zip(lambdas, means, mcse), key=lambda x: float(x[0]))
    lambdas, means, mcse = zip(*sorted_data)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar([str(l) for l in lambdas], means, yerr=[3 * m for m in mcse], capsize=5)
    plt.ylabel('Goodness of Inference')
    plt.xlabel('Lambda')
    plt.title('Entropy Gap with Monte Carlo Standard Error (MCSE)')
    plt.ylim(bottom=1400)
    plt.grid(True, axis='y', linestyle='--', alpha=0.1)
    plt.tight_layout()
    plt.savefig('svae/plots/entropy_gap_plot.png')
    plt.show()



def main():
    df = logdata_2_df()
    plot_inference_goodness(df)

if __name__ == "__main__":
    main()
