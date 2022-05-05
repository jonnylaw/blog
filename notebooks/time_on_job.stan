data {
    int<lower=0> N;
    real<lower=0> job_times[N];
}

parameters {
    real<lower=0> lambda;
}

model {
    lambda ~ gamma(3, 3);
    
    job_times ~ exponential(lambda);
}