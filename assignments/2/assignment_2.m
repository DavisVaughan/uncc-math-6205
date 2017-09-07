%%% Reproducible seed
rng(123);

%%% Parameters
M       = mod(800808804, 25); % Assignment #
N       = 10000;              % Sim number
r       = 0.02 + 0.002 * M;   % Risk free rate
sigma   = 0.25 + 0.005 * M;   % Volatility
T       = 0.5;                % Length
dt      = 1/12;               % Time step
t_total = T / dt;             % Total time steps
s_0     = 100;                % Stock price time 0

%%% 1.1 a - Exact simulation

% Black Scholes stock price - exact
% s_t = s_0 * exp((r - 1/2 * sigma ^ 2) t + sigma * W_t)

% ~N(0,1) random matrix
z = [zeros(N, 1) randn(N, t_total)];

% dW matrix
dW = sqrt(dt) * z;

% rowwise cumulative sum of dW to get W at each time step
W = cumsum(dW, 2);

% 10k sim stock prices for 6 steps each
s_t = s_0 * exp((r-1/2 * sigma^2) * (0:dt:T) + sigma * W);


%%% 1.1 b - Euler scheme

% Black Scholes Euler scheme
% s_(t + dt) = s_t + (r * s_t * dt) + (sigma * s_t * {W_(t + dt) - W_t})
% s_(t + dt) = s_t + (r * s_t * dt) + (sigma * s_t * sqrt(dt) * Z)

% Set up s_t matrix to fill. Initialize to s_0
s_t_euler = nan(N, t_total + 1);
s_t_euler(:, 1) = s_0;

% Fill the columns (time steps). 1 row per path
for(i = 1:t_total)
    s_t_euler(:, i+1) =  s_t_euler(:, i) + r * s_t_euler(:, i) * dt + sigma * s_t_euler(:, i) * sqrt(dt) .* z(:, i+1);
end

%%% 1.1 Extra requirements

% Mean squared discretization error
mse = mean((s_t - s_t_euler) .^2);
mse(:, 7)

% Histogram of lognormal s_T for case a
s_T = s_t(:, t_total + 1);
histogram(s_T)

% Plot first 10 paths of each with same seed
% Cyan is exact
% Magenta is Euler discretization
plot(0:dt:T, s_t(1:10,  :),      'c', ...
     0:dt:T, s_t_euler(1:10, :), 'm')

% Discussion:
% The error is definitely noticeable in the plot. It looks like the farther
% away we get from time 0, the more discretization error is introduced. 
% Decreasing dt would likely help this.


%%% 1.2 Price based on 1.1a compared to Black Scholes

% At the money, so K = 100
k = 100;

% European Call option prices at time 0
c_0 = exp(-r * T) * max(0, s_T - k);

% Average call option price
c_hat_0 = mean(c_0)

% Black scholes european call option price
d_1 = (log(s_0 / k) + (r + sigma^2 / 2) * T ) / (sigma * sqrt(T));
d_2 = d_1 - sigma * sqrt(T);

bs_c_0 = s_0 * normcdf(d_1) - k * exp(-r * T) * normcdf(d_2)

% Out of curiousity, price from 1.1b
c_0_euler = exp(-r * T) * max(0, s_t_euler(:, t_total + 1) - k);
c_hat_0_euler = mean(c_0_euler)
