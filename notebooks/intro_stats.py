# %% Hypothesis Testing
# H0: No difference between means
# H1: There is a difference between means
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import smacof

# Simulate from two Normally distributed populations
def simulate_two_pops(mu1, mu2, n):
    y1 = np.random.normal(loc=mu1, scale=1, size=n)
    y2 = np.random.normal(loc=mu2, scale=1, size=n)
    return y1, y2

# %% plot the two samples
from plotnine import *
import pandas as pd

y1, y2 = simulate_two_pops(mu1=0, mu2=1, n=1000)
df = pd.DataFrame(
    {
        "y1": y1, 
        "y2": y2
    })

melt_df = pd.melt(df, var_name="x", value_name="y")
# %%
ggplot(melt_df, aes(x = "y", colour = "x")) + \
    geom_freqpoly(binwidth=0.1)

# %% Perform a t-test for different values of n
# The p-value represents the probability of getting a result as extreme as the
# observed one under the null hypothesis.
from statsmodels.stats.weightstats import ttest_ind

def experiment(mu1, mu2, n):
    y1, y2 = simulate_two_pops(mu1=mu1, mu2=mu2, n=n)
    return ttest_ind(y1, y2)

ns = np.geomspace(10, 10000, 10)
results = [(int(n), experiment(0.0, 0.1, int(n))) for n in ns for _ in range(10)]

results
# %%
experiment_results = pd.DataFrame(
    {
        "n": [int(n) for n, _ in results],
        "p-value": [r[1] for _, r in results]
    }
)

experiment_results

# %%

ggplot(experiment_results, aes(x = 1, y = "p-value")) + \
    geom_boxplot() + \
    facet_wrap("~n")

# %% Statistical Power
# Statistical Power is the probability of the test correctly rejecting the null
# Avoiding a type II error (incorrectly rejecting the null hypothesis)
# We can replicate the experiment under the same conditions multiple times and calcualte
# the proportion of times we reject the null hypothesis for different sample sizes.

def power_experiment(sample_size, replicates, alpha=0.05):
    def ttest():
        y1, y2 = simulate_two_pops(mu1=0, mu2=0.5, n=sample_size)
        return ttest_ind(y1, y2)[1]
    return sum([ttest() < alpha for _ in range(replicates)]) / replicates

ns = np.geomspace(5, 100, 10)
results = [power_experiment(int(n), 1000) for n in ns]

results
# %%
experiment_results = pd.DataFrame(
    {
        "n": [int(n) for n in ns],
        "power": [r for r in results]
    })

ggplot(experiment_results, aes(x = "n", y = "power")) + \
    geom_point() + \
    geom_hline(yintercept=0.8, linetype="dashed") + \
    labs(title = "Power of the experiment at different sample sizes")

# %% ANOVA (Analysis of variance)
# Anova is a special type of linear regression with categorical variables.
# ANOVA provides a statistical test of whether two or more population means are equal, 
# and therefore generalizes the t-test beyond two means.
import statsmodels.api as sm
from patsy import dmatrices

y1 = np.random.normal(loc=0, scale=1, size=100)
y2 = np.random.normal(loc=-1, scale=1, size=100)
y3 = np.random.normal(loc=1, scale=1, size=100)

anova_df = pd.DataFrame({
    "y1": y1,
    "y2": y2,
    "y3": y3
})

anova_df = pd.melt(anova_df, value_vars=["y1", "y2", "y3"], var_name="x", value_name="y")

anova_df 

# %% Use the patsy formula to create a design matrix
y, X = dmatrices("y ~ 1 + x", data=anova_df)
model = sm.OLS(y, X, data=melt_df)

res = model.fit()

results_summary = res.summary()
results_summary

# %% plot the results
from plotnine import *

# Convert the results to a dataframe
results_as_html = results_summary.tables[1].as_html()
summary_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
summary_df.reset_index(inplace=True)
summary_df

# %% Compute confidence intervals of estimates
summary_df['upper'] = summary_df['coef'] + 1.96 * summary_df['std err']
summary_df['lower'] = summary_df['coef'] - 1.96 * summary_df['std err']

# %%
ggplot(summary_df) + \
    geom_point(aes(x = "index", y = "coef")) + \
    geom_errorbar(aes(ymin = "upper", ymax = "lower", x = "index"))
# %% We can directly use the formula interface
import statsmodels.formula.api as smf
model
# %%
