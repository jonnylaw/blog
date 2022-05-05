data {
    int<lower=0> N;
    int<lower=0> y1;
    int<lower=0> y2;
}

parameters {
    real<lower=0, upper=1> theta1;
    real<lower=0, upper=1> theta2;
}

model {
    theta1 ~ beta(1,1);
    theta2 ~ beta(1,1);
    
    y1 ~ binomial(N, theta1);
    y2 ~ binomial(N, theta2);
}

generated quantities {
    real difference;
    difference = theta1 - theta2;
}