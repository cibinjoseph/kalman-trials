## Kalman filter on a position-velocity system
# Implements one using a vanilla kalman filter and one using KalmanFilters.jl
using Plots
using Random
using LinearAlgebra
using KalmanFilters

# Seed to get reproducible results
Random.seed!(1234)

# Position state
function get_position(t)
    return 14.4*sin(0.5*t)
end

# This is for verification at the end
function get_velocity(t)
    return 14.4*0.5*cos.(0.5*t)
end

n = 400
t = LinRange(0, 20, n)
dt = t[2]-t[1]

# Initialize state and covariance
xhat_prev = [0.0; 14.4*0.5]
P_prev = [5.0 0; 0 5]
Q = [0 0; 0 3]
A = [1 dt; 0 1]
H = [1 0]
R = [15]
noise = randn(length(t))

# Vectors for recording
x_record = zeros(2, n)
xhat_pred = zeros(2, 1)
P_pred = zeros(2, 2)
K_gain = zeros(2, 1)
zk = zeros(1, n)
xhat_estimate = similar(xhat_pred)
P_estimate = similar(P_pred)

# Implementation using Vanilla Kalman
# for i in 1:n
#     # Predict state and error covariance
#     xhat_pred .= A*xhat_prev
#     P_pred .= A*P_prev*A' + Q
#
#     # Compute Kalman gain
#     K_gain .= P_pred*H'*(H*P_pred*H' + R)^(-1)
#
#     # Compute state estimate
#     zk[:, i] .= get_position(t[i]) + noise[i]
#     xhat_estimate .= xhat_pred + K_gain*(zk[:, i] - H*xhat_pred)
#     x_record[:, i] .= xhat_estimate
#
#     # Compute estimate error covariance
#     P_estimate .= P_pred - K_gain*H*P_pred
#
#     # Update values for next timestep
#     xhat_prev .= xhat_estimate
#     P_prev .= P_estimate
# end

# Implementation using KalmanFilters.jl package
i = 1
mu = measurement_update(xhat_prev, P_prev, [get_position(t[i])+noise[i]], H, R)
for i = 1:n
    tu = time_update(get_state(mu), get_covariance(mu), A, Q)
    global mu = measurement_update(get_state(tu), get_covariance(tu), [get_position(t[i])+noise[i]], H, R)
    zk[:, i] .= get_position(t[i]) + noise[i]
    x_record[:, i] .= mu.state
end

# Plot results
plot(t, zk', label="Measured position")
plot!(t, x_record[1, :], label="Estimated position")
plot!(t, x_record[2, :], label="Estimated velocity")
plot!(t, get_velocity(t), label="Actual velocity")
