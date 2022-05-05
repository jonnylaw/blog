import stan
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


def experiment(n, p1, p2):
    sample1 = np.random.binomial(n=1, p=p1, size=n)
    sample2 = np.random.binomial(n=1, p=p2, size=n)
    return sample1, sample2


if __name__ == "__main__":
    with open("binomial_ab_test.stan", "r") as f:
        stan_code = f.read()

    y1, y2 = experiment(n=100, p1=0.05, p2=0.055)
    stan_data = {
        "N": 100,
        "y1": y1.sum(),
        "y2": y2.sum(),
    }

    posterior = stan.build(stan_code, data=stan_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas

    print(df.head())

    # Plot traceplots
    plt.figure()
    axes = az.plot_trace(fit, var_names=["theta1", "theta2", "difference"])
    plt.savefig("binomial_ab_test_traceplots.png")

    # Plot the differences
    plt.figure()
    az.plot_posterior(fit, var_names=["difference"])
    plt.savefig("binomial_ab_test_differences.png")