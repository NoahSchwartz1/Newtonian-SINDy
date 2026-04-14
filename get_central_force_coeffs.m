function [outputArg1,outputArg2] = get_central_force_coeffs(trajectories,options)
%GET_CENTRAL_FORCE_COEFFS Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    trajectories
    options.maxOrder = 4
    
end

arguments (Output)
    outputArg1
    outputArg2
end

outputArg1 = inputArg1;
outputArg2 = inputArg2;
end