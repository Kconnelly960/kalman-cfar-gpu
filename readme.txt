README

This project contains all of the source code for the kalman filtering and CFAR algorithms, created by Kevin Connelly and Daniel Bower 

Setup

First create the data using the Matlab code in the ./Matlab folder
Run the generate_data_single_target.m file to create the data
To do that we created a Matlab session at the following website where we access the files and run them
https://ood.hpc.arizona.edu/

The file will be generated in your base directiory at /home/u8/{id}, copy these files into the project folder location.
They will then need to be copied into the KF folder and the combined folder. You can perform these operations with


Cp /home/u8/{id}/generate_data_single_target.m.csv  /home/u8/{id}/{path to project directory}/KF
Cp /home/u8/{id}/generate_data_single_target.m.csv  /home/u8/{id}/{path to project directory}/Combined



Once that is done to run the you code you simply need to navigate back to the project directory and run 'srun run_project.slurm' 
this will run multiple versions of the kalman filter code and the combined version of the code, each will be run 30 times.

Be sure to give the slurm file read write and execute privileges 

Once run in the KF folder there are three subfolders, each ending in _result which will output kernel results and a performance_metrics.txt 
which contains the run times for each run.

In the Combined folder you will find the same structure, with a combined_results folder.
