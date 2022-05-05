data {
    int N;
    real y[N];
}
parameters {
    real<lower=0> lambda;
}
model {
    lambda ~ gamma(1,1);
    y ~ normal(0,1/sqrt(lambda));
}