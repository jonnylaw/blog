data {
    int<lower=0> N;
    int<lower=0> ya;
    int<lower=0> yb;
}

parameters {
    real<lower=0, upper=1> theta_a;
    real<lower=0, upper=1 - theta_a> theta_delta; // positive delta which creates a "larger" theta_b relative to theta_a
}

model {
    real theta_b= theta_a + theta_delta;
    
    theta_a ~ beta(1, 1);
    theta_delta ~ uniform(0, 1 - theta_a);
    
    ya ~ binomial(N, theta_a);
    yb ~ binomial(N, theta_b);
}