function trajectories = get_newtonian_trajectories(x_init,v_init,masses,options)
%GET_NEWTONIAN_TRAJECTORIES Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    x_init
    v_init
    masses
    options.tmax = 10
    options.uniformGridSpacing = true
    options.tResolution = 0.01
end

arguments (Output)
    trajectories
end

N = length(masses);

% Builds a function that we can pass into an ODE solver
% The given function accepts a vector x=[positions,velocities] as input,
% and returns a vector xdot=[velocities,accelerations] as output
f = @(t,x) ([...
    reshape(x(3*N+1:end),3*N,1);...
    reshape(get_newtonian_acceleration(...
        reshape(x(1:3*N),[3,N]),masses),3*N,1)...
    ]);

if options.uniformGridSpacing
    tVals = 0:options.tResolution:options.tmax;
    [t,x] = ode15s(f,tVals,[x_init(:),v_init(:)]);
else
    [t,x] = ode15s(f,[0,options.tmax],[x_init(:),v_init(:)]);
end

% Build out the trajectories struct in a useful format
for i=1:N
    trajectories(i).t = t;
    trajectories(i).x = zeros(6,length(trajectories(i).t));
    trajectories(i).x(1:3,:) = x(:,(3*i)-2:(3*i))';
    trajectories(i).x(4:6,:) = x(:,(3*(i+N))-2:(3*(i+N)))';
end