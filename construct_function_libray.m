function library = construct_function_libray(trajectories,options)
%CONSTRUCT_FUNCTION_LIBRAY Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    trajectories
    options.maxOrder = 4 % sets the max order of the central force polynomial expansion
end

arguments (Output)
    library
end

N = length(trajectories);
T = length(trajectories(1).t);

% We now compute orders of central force for each timestep and each 

end