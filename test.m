for i=1:10
    fprintf("%d\n",i)
end

traj = get_newtonian_trajectories(...
    [0,0,0; 1 1 1; 2 0 2]',...
    [0,0,0; 0 0 0; 1 1 0]',...
    [1,1,2]);
disp(traj)

figure; hold on; grid on;
for i=1:length(traj)
    plot3(traj(i).x(1,:),traj(i).x(2,:),traj(i).x(3,:));
end