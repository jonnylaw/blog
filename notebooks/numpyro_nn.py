# %% Simulate a regression model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_data(xs):
    a = 5
    b = 3
    mean = lambda x: a * np.sin(x + b)
    data = [(x, np.random.normal(loc=mean(x), scale=1, size=1))
            for x in xs]
    x, y = zip(*data)
    return np.array(x), np.array(y)


xs = np.random.uniform(0, 10, size=100)
x, y = generate_data(xs)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

plt.scatter(x, y)

# %%
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.example_libraries.stax as stax

def dense_model():
    return stax.serial(
        stax.Dense(32),
        stax.Relu,
        stax.Dropout(0.1),
        stax.Dense(32),
        stax.Relu,
        stax.Dropout(0.1),
        stax.Dense(32),
        stax.Relu,
        stax.Dropout(0.1),
        stax.Dense(32),
        stax.Relu,
        stax.Dropout(0.1),
        stax.Dense(1)
    )

# %% Get the shape of the parameters
import jax

init, apply = dense_model()
rng = jax.random.PRNGKey(0)
output_shape, init_params = init(rng, (1,))

for tuple in init_params:
    if not tuple:
        print("()")
    for t in tuple:
        print(t.shape)

# %% 
from functools import reduce
def prior_module(layer, input_dim, output_dim):
    weight = numpyro.sample(f"w{layer}", dist.Normal(jnp.zeros((input_dim, output_dim)), jnp.ones((input_dim, output_dim))))
    bias = numpyro.sample(f"b{layer}", dist.Normal(jnp.zeros((output_dim,)), jnp.ones((output_dim,))))

    return [(weight, bias), (), ()]

def prior():
    input_layer = prior_module(0, 1, 32)
    hidden_layers = [prior_module(i, 32, 32) for i in range(1, 4)]
    output_weights = numpyro.sample(f"w4", dist.Normal(jnp.zeros((32, 1)), jnp.ones((32, 1))))
    output_bias = numpyro.sample(f"b4", dist.Normal(jnp.zeros((1,)), jnp.ones((1,))))

    return input_layer + reduce(lambda a, b: a + b, hidden_layers) + [(output_weights, output_bias)]

# %% Sample from the prior
from numpyro import handlers

prior_params = handlers.seed(prior, rng_seed=0)()

for tuple in prior_params:
    if not tuple:
        print("()")
    for t in tuple:
        print(t.shape)

# %% Apply the model using the prior parameters

apply(prior_params, x.reshape((100, 1)), rng=rng)

# %%
def model(x, y=None):
    # Sample prior
    params = prior()

    # Spec the convolutional layers
    _, apply = dense_model()  # We use our prior function instead of the initializer
    mu = apply(params, x, rng=rng)

    sigma = numpyro.sample("sigma", dist.HalfCauchy(0.5))

    # Specify the observation distribution
    with numpyro.plate("data", x.shape[0]):
        numpyro.sample("Y", dist.Normal(loc=mu, scale=1.0).to_event(1), obs=y)

# %% HMC
from numpyro.infer import MCMC, NUTS
import jax

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
rng = jax.random.PRNGKey(0)
mcmc.run(rng, x=x.reshape((100, 1)), y=y, extra_fields=("potential_energy",))

# %% posterior diagnostics
mcmc.print_summary()

# %% Sample from the posterior predictive distribution
from numpyro.infer import Predictive

x_plot = np.arange(0, 10, 0.1)
predictive = Predictive(model, posterior_samples=mcmc.get_samples())
posterior_samples = predictive(jax.random.PRNGKey(1), x_plot.reshape((100, 1)))

obs = posterior_samples["Y"]

# %%
posterior_mean = jnp.mean(obs, axis=0)
upper_posterior = jnp.percentile(obs, q = 1 - (1 - 0.89) / 2 , axis=0)
lower_posterior = jnp.percentile(obs, q = (1 - 0.89) / 2, axis=0)

posterior_mean[:10], upper_posterior[:10], lower_posterior[:10]

# %% Posterior predictions for x_test
import seaborn as sns
from seaborn import FacetGrid
import pandas as pd

predictive = Predictive(model, posterior_samples=mcmc.get_samples())
posterior_samples = predictive(jax.random.PRNGKey(1), x_test.reshape((20, 1)))

samples_df = pd.DataFrame(posterior_samples["Y"].squeeze()).T
samples_df['y_value'] = y_test
samples_df.set_index('y_value', inplace=True)

# Pivot the dataframe to x_test, y_pred
samples_df = samples_df.stack()
samples_df = pd.DataFrame(samples_df, columns=['y_pred'], index=samples_df.index)

# Convert the index to a column
samples_df.reset_index(inplace=True)

# Convert y_pred to a fucking number
samples_df['y_pred'] = samples_df['y_pred'].astype(float)

FacetGrid(data=samples_df, col = 'y_value', col_wrap=5, sharex=False).\
    map_dataframe(sns.kdeplot, x = "y_pred")

# %% Plot one example
sns.kdeplot(samples_df[samples_df['observation'] == samples_df['observation'][0]]['y_pred'], label='observation 0')
# %%

plt.plot(x_test, y_test, ".")
plt.plot(x_plot, posterior_mean.squeeze(), "-")
plt.fill_between(x_plot, lower_posterior.squeeze(), upper_posterior.squeeze(), alpha=0.5)

# %%
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer import SVI, Trace_ELBO

# model = p(theta) p(y | theta, x), guide = q(theta|phi)
optimizer = numpyro.optim.Adam(step_size=0.0005)
svi = SVI(model, AutoDiagonalNormal(model), optimizer, loss=Trace_ELBO())
rng = jax.random.PRNGKey(0)

# Use svi_run
svi_result = svi.run(rng, 10000, x_train.reshape((80, 1)), y_train)

# %% Learning curve
plt.plot(svi_result.losses)

# %%
params = svi_result.params
n_samples = 100
predictive = Predictive(model, params=params, num_samples=n_samples)
samples = predictive(jax.random.PRNGKey(1), x_plot.reshape((100, 1)))

# %% Plot the posterior predictive distribution
posterior_mean = jnp.mean(samples["Y"], axis=0)
upper_posterior = jnp.percentile(samples["Y"], 0.75, axis=0)
lower_posterior = jnp.percentile(samples["Y"], 0.25, axis=0)

sorted_x = jnp.sort(x_test, axis=0).squeeze()
plt.plot(x_test, y_test, ".")
plt.plot(x_plot, posterior_mean, "-")
# plt.fill_between(x_plot, lower_posterior.squeeze(), upper_posterior.squeeze(), alpha=0.5)


# %%
