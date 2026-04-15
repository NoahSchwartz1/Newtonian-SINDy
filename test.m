for i=1:10
    fprintf("%d\n",i)
end

% traj = get_newtonian_trajectories(...
%     [0,0,0; 1 1 1; 2 0 2]',...
%     [0,0,0; 0 0 0; 1 1 0]',...
%     [1,1,2]);
% disp(traj)
% 
% figure; hold on; grid on;
% for i=1:length(traj)
%     plot3(traj(i).x(1,:),traj(i).x(2,:),traj(i).x(3,:));
% end


traj = get_newtonian_trajectories(...
    [0,0,0; 1 1 1; 2 0 2; 5 3 2; -2 9 4; 3 1 2; 5 5 5]',...
    [0,0,0; 0 0 0; 1 1 0; 0 0 0 ; 0 0 0; 1 -2 3; 5 0 0]',...
    [1,1,3,2.2,12,3,1]);
disp(traj)

figure; hold on; grid on;
for i=1:length(traj)
    plot3(traj(i).x(1,:),traj(i).x(2,:),traj(i).x(3,:));
end

save("traj.mat","traj");


%%

traj_3body = get_newtonian_trajectories(...
    [0,0,0; 1 1 1; 2 0 2]',...
    [0,0,0; 0 0 0; 1 1 0]',...
    [1,1,3]);
figure; hold on; grid on;
for i=1:length(traj)
    plot3(traj(i).x(1,:),traj(i).x(2,:),traj(i).x(3,:));
end
save("3body.mat","traj_3body")