# %%
import jax.numpy as jnp
import jax
import numpy as np
from itertools import accumulate
from scipy import stats
import matplotlib.pyplot as plt

# Define a partially observed predator-prey model. 
# There are two populations, one prey, x_1 and one predator x_2. We only observe the prey.

# %% 
def step_lotka_volterra(p, x0, dt):
    x1 = x0[0]
    x2 = x0[1]
    d = p[1] * x1 * x2
    a = np.array([p[0] * x1 - d, d - p[2] * x2])
    b = np.matrix([[p[0] * x1 + d, -d], 
                   [-d, d + p[2] * x2]])
    b = b + b.T / 2
    return stats.multivariate_normal(x0 + a * dt, b * dt)


def observation(x1, p):
    return stats.norm(x1, p[3])


def simulate(step_fun, obs, x0, n):
    xs = np.empty((2, n))
    ys = np.empty(n)
    xs[:,0] = x0
    ys[0] = obs(x0[0])
    for i in range(n - 1):
        xs[:,i+1] = step_fun(xs[:,i])
        ys[i+1] = obs(xs[0, i])
    return (xs, ys)


p = [0.5, 0.0025, 0.3, 1]
dt = 0.2
n = 100
simulation = simulate(
    lambda x: step_lotka_volterra(p, x, dt).rvs(1), 
    lambda x: observation(x, p).rvs(1), 
    np.array([100, 100]), 
    n)

ts = np.linspace(0, 1.0, num = 100)
plt.plot(ts, simulation[0][0], 'r-', ts, simulation[0][1], 'b-')
plt.show()

# %%
# Particle filter
def multinomial_resample(x, weights):
    """Perform multinomial resampling"""
    n = len(x)
    p = x[0].shape
    max_weight = np.max(weights)
    w1 = np.exp(weights - max_weight)
    probs = w1 / np.sum(w1)
    mn = stats.multinomial(n = n, p = probs)
    selected = [np.tile(x[i], total) for (_, i), total in np.ndenumerate(mn.rvs(1))]
    reshaped = np.reshape(np.concatenate(selected), (p[0], n))
    return reshaped.T


def step_pf_vectorised(x0, y, resample, obs_lik, step_fn):
    """A single Step of the particle filter where the step
    function and observation likelihood are assumed to be vectorised"""
    x1 = step_fn(x0) # vectorised step function
    log_weights = obs_lik(x1, y) # vectorised likelihood
    return resample(x1, log_weights)


def step_pf(x0, y, resample, obs_lik, step_fn):
    """A single Step of the particle filter where the step
    function and observation likelihood are not vectorised"""
    x1 = [step_fn(x) for x in x0] 
    log_weights = [obs_lik(x, y) for x in x0]
    return resample(x1, log_weights)


def pf(ys, x0, obs_lik, step_fn):
    """Functional particle filter which works in python 3.8+.
    The initial argument was not added until Python 3.8"""
    f = lambda x, y: step_pf_vectorised(x, y, multinomial_resample, obs_lik, step_fn)
    return np.array(list(accumulate(ys, f, initial=x0)))


def particle_filter(ys, x0, obs_lik, step_fn):
    n, p = x0.shape
    x = np.empty((n, p, len(ys) + 1))
    x[:, :, 0] = x0
    for i, y in enumerate(ys):
        x[:, :, i + 1] = step_pf_vectorised(x[:, :, i], y, multinomial_resample, obs_lik, step_fn)
    return x

ys = simulation[1]
key = jax.random.PRNGKey(0)
x0 = jax.random.multivariate_normal(key, np.array([100, 100]), np.diag([10, 10]), shape=(1000,))

# vectorised observation function
obs = lambda x, y: stats.norm(x[:,0], p[3]).logpdf(y)

# Define a step function which can run on the GPU in parallel
def jnp_step_lotka_volterra(p, x0, dt):
    x1 = x0[0]
    x2 = x0[1]
    d = p[1] * x1 * x2
    a = jnp.array([p[0] * x1 - d, d - p[2] * x2])
    b = jnp.array([[p[0] * x1 + d, -d], 
                   [-d, d + p[2] * x2]])
    b = b + b.T / 2
    return jax.random.multivariate_normal(key, x0 + a * dt, b * dt)

# Partially apply and use vmap this uses jit to compile the function
# and runs on a GPU where available
jit_step = jax.jit(lambda x: jnp_step_lotka_volterra(p, x, 0.2))
step_fn = jax.vmap(jit_step, in_axes=0, out_axes=0)
filtered = particle_filter(ys[:50], x0, obs, step_fn)

# Summarise the filtered results
filtered_mean = np.mean(filtered, axis=0)
lower = np.quantile(filtered, 0.25, axis=0)
upper = np.quantile(filtered, 0.75, axis=0)

# %%
# Plot the filtered results
time = np.linspace(0, 2, num = 21)
# plt.plot(filtered_df["time"], filtered_df["filtered_prey"])
plt.plot(time, simulation[1][:21], "r-")
plt.plot(time, filtered_mean[1, :], "g-")
plt.fill_between(time, lower[1, :], upper[1, :], color="g", alpha=0.2)
plt.show()

# %% Marginal likelihood
from functools import reduce

def step_pf_vectorised(x0, ll, y, resample, obs_lik, step_fn):
    """A single Step of the particle filter where the step
    function and observation likelihood are assumed to be vectorised"""
    x1 = step_fn(x0) # vectorised step function
    log_weights = obs_lik(x1, y) # vectorised likelihood
    max_weight = np.max(log_weights)
    w1 = np.exp(log_weights - max_weight) # max weight size is 1
    ll = ll + max_weight + np.log(np.mean(w1)) # update the marginal likelihood
    return (resample(x1, log_weights), ll)


def pfmlik(ys, x0, obs_lik, step_fn):
    """Functional particle filter which returns the pseudo-marginal log-likelihood"""
    f = lambda x, y: step_pf_vectorised(x[0], x[1], y, multinomial_resample, obs_lik, step_fn)
    return reduce(f, ys, (x0, 0.0))[1]


x0 = jax.random.multivariate_normal(key, np.array([100, 100]), np.diag([10, 10]), shape=(100,))
pfmlik(ys[:20], x0, obs, step_fn)

# step_pf_vectorised(x0, 0.0, simulation[1][0], multinomial_resample, obs, step_fn)

# %%
def pmmh(init, proposal, m, ll):
    p = np.empty((len(init), m))
    p[:,0] = init
    prev_ll = -1e99
    for i in range(m):
        prop_p = proposal(p[:, i])
        prop_ll = ll(prop_p)
        a = prop_ll - prev_ll
        u = stats.uniform().rvs(1)
        if a < np.log(u):
            p[:, i + 1] = prop_p
            prev_ll = prop_ll
        else:
            p[:, i + 1] = p[i, :]

    return p

def ll(p): 
    obs = lambda x, y: stats.norm(x[:,0], p[3]).logpdf(y)
    jit_step = jax.jit(lambda x: jnp_step_lotka_volterra(p, x, 0.2))
    step_fn = jax.vmap(jit_step, in_axes=0, out_axes=0)
    x0 = jax.random.multivariate_normal(key, np.array([100, 100]), np.diag([10, 10]), shape=(100,))
    return pfmlik(ys, x0, obs, step_fn)

def proposal(p):
    return np.exp(jax.random.multivariate_normal(key, np.log(np.array(p)), np.diag([0.1, 0.1, 0.1, 0.1])))

iters = pmmh(p, proposal, 100, ll)

# %%
