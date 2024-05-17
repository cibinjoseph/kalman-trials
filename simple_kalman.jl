## A simple kalman filter on a 1-d system
using Plots
using Random

Random.seed!(1234)

function get_V(t)
    # return 14.4*sin(0.5*t) + randn()
    return 14.4 + 4*randn()
end

n = 400
t = LinRange(0, 20, n)
z = get_V.(t)

# Initialize state and covariance
xhat_prev = 14.0
P_prev = 6.0
Q = 0
A = 1
H = 1
R = 4
At = A'
Ht = H'

# Vectors for recording
x_record = zeros(n)

for i in 1:n
    # Predict state and error covariance
    xhat_pred = A*xhat_prev
    P_pred = A*P_prev*At + Q

    # Compute Kalman gain
    K_gain = P_pred*Ht*(H*P_pred*Ht + R)^(-1)

    # Compute state estimate
    zk = get_V(t[i])
    xhat_estimate = xhat_pred + K_gain*(zk - H*xhat_pred)
    x_record[i] = xhat_estimate

    # Compute error covariance
    P_estimate = P_pred - K_gain*H*P_pred

    # Update values for next timestep
    global xhat_prev = xhat_estimate
    global P_prev = P_estimate
end

plot(t, z)
plot!(t, x_record)
