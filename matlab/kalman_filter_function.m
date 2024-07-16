function [filtered_values,estimate_errors] = kalmanFilter(data,process_noise,measurement_noise,estimated_error,initial_value)

%   Initialize variables
n_iterations = size(data, 1);
estimates = zeros(n_iterations, 1);
errors = zeros(n_iterations, 1);

% initial guess
estimate = initial_value;
error_estimate = estimated_error;

for i = 1:n_iterations
    % kalman gain
    kalman_gain = error_estimate / (error_estimate + measurement_noise);

    % update estimate
    estimate = estimate + kalman_gain * (data(i) - estimate);

    % update error estimate
    error_estimate = (1 - kalman_gain) * error_estimate + process_noise;

    estimates(i) = estimate;
    errors(i) = error_estimate;
end

filtered_values = estimates;
estimate_errors = errors;

end

