# %% Time on Job
import stan
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

def time_on_job(n_jobs: int) -> np.array:
    """
    Simulate the time spent on a job for a van with a given job
    Parameters
    ----------
    n_jobs : int
        Number of jobs.
    Returns
    -------
    time_on_job : np.array
        Time spent on a job for a van.
    """
    job_types = np.random.choice(
            np.arange(3),
            size=n_jobs,
            p=[0.3, 0.3, 0.4]
        )
    job_rates = [1.0, 0.2, 3.0]
    rates = [job_rates[job_type] for job_type in job_types]
    return pd.DataFrame({
        "job_type": job_types,
        "job_time": [np.random.exponential(scale=rate, size=1)[0] for rate in rates]
    })

if __name__ == "__main__":
    with open("time_on_job.stan", "r") as f:
        stan_code = f.read()

    n = 100
    job_times = time_on_job(1.0, n)
    stan_data = {
        "N": n,
        "job_times": job_times
    }

    posterior = stan.build(stan_code, data=stan_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    df = fit.to_frame()  

    print(df.head())

    # Plot traceplots
    plt.figure()
    axes = az.plot_trace(fit, var_names=["lambda"])
    plt.savefig("lambda_traceplots.png")
