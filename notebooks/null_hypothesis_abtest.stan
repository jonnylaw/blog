data {
    int<lower=0> N;
    int<lower=0> ya;
    int<lower=0> yb;
}

parameters {
    real<lower=0, upper=1> theta;
}

model {
    theta ~ beta(1, 1);
    
    ya ~ binomial(N, theta);
    yb ~ binomial(N, theta);
}