data = csvread('radarDataMatlab.csv');

% part 2 - perforam kalman filtering
process_noise = 0.01; % process noise variance
measurement_noise = 1.0; % measrument noise variance
estimated_error = 1.0; % initial estimate of error
initial_value = 0;

% measure execution time
tic; 
[filtered_values, estimated_errors] = kalman_filter_function(data, process_noise, measurement_noise, estimated_error, initial_value);
elapsedTime = toc;

disp(['Time elapsed for kalman filtering: ', num2str(elapsedTime), ' seconds']);