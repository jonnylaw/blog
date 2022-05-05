# %% Linear regression with SVI

# %% Simulate a regression model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.plot(X, y, ".")

# %%
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


def model(x, y=None):
    b0 = numpyro.sample("b0", dist.Normal(0, 1))
    beta = numpyro.sample("b1", dist.Normal(0, 1))

    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))

    with numpyro.plate("data", x.shape[0]):
        y_hat = b0 + beta * x.T
        numpyro.sample("Y", dist.Normal(loc=y_hat, scale=sigma), obs=y)


# %% HMC
from numpyro.infer import MCMC, NUTS
import jax

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000, num_chains=4)
rng = jax.random.PRNGKey(0)
mcmc.run(rng, x=x_train, y=y_train, extra_fields=("potential_energy",))

# %%
mcmc.print_summary()
# %%
from numpyro.infer import Predictive

params = mcmc.get_samples()
predictive = Predictive(model, posterior_samples=params)
posterior_samples = predictive(jax.random.PRNGKey(1), jnp.sort(x_test, axis=0))

obs = posterior_samples["Y"].reshape(params['b0'].shape[0], -1)

# %%
posterior_mean = jnp.mean(obs, axis=0)
upper_posterior = jnp.percentile(obs, 97.5, axis=0)
lower_posterior = jnp.percentile(obs, 2.5, axis=0)

sorted_x = jnp.sort(x_test, axis=0).squeeze()
plt.plot(x_test, y_test, ".")
plt.plot(sorted_x, posterior_mean, "-")
plt.fill_between(sorted_x, lower_posterior, upper_posterior, alpha=0.5)

# %%
