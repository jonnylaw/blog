# %% Frequentist A/B Test
# Simulate some data
import numpy as np

def experiment(n, p1, p2):
    sample1 = np.random.binomial(n=n, p=p1, size=1)
    sample2 = np.random.binomial(n=n, p=p2, size=1)
    return sample1, sample2

# %% Fisher exact test
from scipy.stats import fisher_exact

n = 5000
y1, y2 = experiment(n=n, p1=0.05, p2=0.07)
odds_ratio, p_value = fisher_exact(table=np.array([[y1, n - y1],[y2, n - y2]]).squeeze(), alternative="less")

p_value

# %% Sample size and significance
import matplotlib.pyplot as plt
sample_sizes = np.geomspace(100, 10000, 50)
def fisher_exact_significance(sample_sizes):
    p_values = []
    for n in sample_sizes:
        for _ in range(reps):
            y1, y2 = experiment(n=n, p1=0.05, p2=0.07)
            _, p_value = fisher_exact(table=np.array([[y1, n - y1],[y2, n - y2]]).squeeze(), alternative="less")
            p_values.append(p_value)
    
    return pd.DataFrame([sample_sizes, p_values])

sample_size, p_values = fisher_exact_significance(sample_sizes)

plt.plot(sample_size, p_values)

# %% Chi Square Test
from statsmodels.stats.proportion import proportions_chisquare

test_stat, p_value, contingency_table = proportions_chisquare(np.concatenate([y1, y2]), np.array([1000, 1000]))
test_stat, p_value, contingency_table
# %% 
def chi_squared_test(reps, alpha=0.05):
    """Perform a chi-squared test for the difference between two proportions
    with a given number of repetitions.

    Count the number of times the difference between the two proportions is significant
    at the alpha level. This is the statistical power of the test.
    """
    p_values = []
    for _ in range(reps):
        y1, y2 = experiment(n=1000, p1=0.05, p2=0.07)
        test_stat, p_value, contingency_table = proportions_chisquare(np.concatenate([y1, y2]), np.array([1000, 1000]))
        p_values.append(p_value)

    return sum([p_value < alpha for p_value in p_values]) / reps

chi_squared_test(10000)

# %% Statistical power
from statsmodels.stats.power import GofChisquarePower
chisq = GofChisquarePower()
calc_power = chisq.solve_power(effect_size=0.02, alpha=0.05, power=None, nobs=1000)
calc_power

# %% Sample size calculation

# Determine the amount of samples we need with alpha = 0.05, 1 - beta = 0.8
# probability of type I error is 0.05 - the probability of rejecting the null when it is true
# probability of type II error is 0.2 - the probability of accepting the null when it is false

# We first define our effect size, which is the difference between 
# the latent probability of the two groups
# We can calculate the number of samples required detect the 
# effect size at the desired power with alpha = 0.05.
import matplotlib.pyplot as plt
import statsmodels.stats.power
import pandas as pd
from plotnine import *

effect_size = np.geomspace(0.01, 0.1, 20)
chi = statsmodels.stats.power.GofChisquarePower()
sample_size = [chi.solve_power(effect_size=d, nobs=None, alpha=0.05, power=0.8) for d in effect_size]

df = pd.DataFrame({"effect_size": effect_size, "sample_size": sample_size})

ggplot(aes(y = sample_size, x = effect_size)) + \
  geom_line() +\
  geom_point() +\
  labs(y = "Sample Size", x = "Effect Size", title = "Required sample size to detect effect size") + \
  theme_minimal()
  
# %% Chi-squared test of proportions
import statsmodels.stats.power

chisq = statsmodels.stats.power.GofChisquarePower()
sample_size = chisq.solve_power(effect_size=0.05, alpha=0.05, power=0.8, nobs=None)
sample_size

# %% Bayesian A/B Test, with a uniform prior beta(1, 1)
import matplotlib.pyplot as plt
import scipy.stats.distributions as stats
n = 10
y1, y2 = experiment(n=n, p1=0.05, p2=0.06)

alpha, beta = 1, 1
# Perform exact inference
theta1 = stats.beta(a=alpha + y1.sum(), b=beta + n - y1.sum())
theta2 = stats.beta(a=alpha + y2.sum(), b=beta + n - y2.sum())

x = np.arange(0, 1, 0.001)
plt.plot(x, theta1.pdf(x), color = 'r', label = 'theta1')
plt.plot(x, theta2.pdf(x), color = 'b', label = 'theta2')
plt.show()

# %%
import scipy.stats.distributions as dist

alphas = np.arange(1, 100, 1)
betas = np.arange(1, 100, 12)
variances, alpha, beta = zip(*[(dist.beta.var(alpha, beta), alpha, beta) for alpha in alphas for beta in betas])
df = pd.DataFrame({
  'variance': variances,
  'alpha': alpha,
  'beta': beta
})

ggplot(df, aes(x = "alpha", y = "variance")) +\
    geom_line() +\
    facet_wrap("~beta", scales="free_y") + \
    theme(subplots_adjust={'wspace': 0.3})

# %% We can calculate the posterior probability of the difference between theta1 and theta2
# The most straightforward way is to sample from the posterior distribution
from plotnine import *
import pandas as pd

difference = theta2.rvs(1000) - theta1.rvs(1000)

# Region of practical equivalence. 
# We could define a ROPE of say 0.01 and then plot that 
ggplot(aes(x = "difference"), data = pd.DataFrame(difference)) + \
    geom_density() + \
    geom_vline(xintercept = 0.0) + \
    geom_vline(xintercept = 0.01) + \
    geom_vline(xintercept = -0.01)

# %% Bayesian power calculation with exact inference
# Calculate the probability that the difference is greater than 0.01

def ab_test(n_sample, n_sims, theta1, theta2, rope=0.01):
    """Calculate the probability that theta2 is larger than theta1
    with a given sample size and a given rope."""
    y1, y2 = experiment(n=n_sample, p1=theta1, p2=theta2)

    alpha, beta = 1, 1
    # Perform exact inference for the latent conversion with uniform prior
    theta1 = stats.beta(a=alpha + y1.sum(), b=beta + n - y1.sum())
    theta2 = stats.beta(a=alpha + y2.sum(), b=beta + n - y2.sum())

    difference = theta2.rvs(n_sims) - theta1.rvs(n_sims)
    pr_a_best = sum(difference < - rope) / n_sims
    pr_equiv = sum((difference > - rope) & (difference < rope)) / n_sims
    pr_b_best = sum(difference > rope) / n_sims
    return pr_a_best, pr_equiv, pr_b_best

sample_size = np.arange(10, 1000, 20)
a, none, b = zip(*[ab_test(n, 1000, 0.05, 0.06) for n in sample_size])

plt.plot(sample_size, a, label='a wins')
plt.plot(sample_size, none, label='equivalent')
plt.plot(sample_size, b, label='b wins')

# %%
def simulation_power_calc(sample_sizes, trials, theta1, theta2, rope=0.01):
    """ Calculate the power for a range of sample sizes"""
    return pd.DataFrame({
        'theta1': theta1,
        'theta2': theta2,
        'sample_size': sample_sizes,
        'power':[sum([ab_test(n, theta1, theta2) > 0.95 for _ in range(trials)]) / trials for n in sample_sizes]
    })

results = simulation_power_calc(np.arange(100, 10000, 1000), 100, 0.04, 0.06)

plt.plot(results.sample_size, results.power)
# %% 
