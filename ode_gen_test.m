x = [0,0,0,1,1,1,0.25,0,0,0,0,0];
newtonian_gravity(0,x)

% y = ode23s(@newtonian_gravity,[1,10],x)
% figure; hold on; grid on;
% plot3(y.y(1,:),y.y(2,:),y.y(3,:),Color='r');
% plot3(y.y(4,:),y.y(5,:),y.y(6,:),Color='b');

x = [0,0,0,1,1,1,2,2,0,0.25,0,0,0,0,0,0,1,0];
y = ode23s(@newtonian_gravity,[1,10],x)
figure; hold on; grid on;
plot3(y.y(1,:),y.y(2,:),y.y(3,:),Color='r');
plot3(y.y(4,:),y.y(5,:),y.y(6,:),Color='b');
plot3(y.y(7,:),y.y(8,:),y.y(9,:),Color='g');



function output = newtonian_gravity(t,x)
    % assume x is an input vector where the first 3N entries are the
    % positions of particles ordered by (x1,y1,z1,x2,y2,z2), and the second
    % 3N entries are particle velocities ordered by (vx1,vy1,vz1,vx2,...)
    % and so on

    % return the vector corresponding to (v,a), where the first 3N entries
    % are the velocities and the second 3N entries are the accelerations as
    % given by Newton's gravitational rule 1/r^2 pairwise


    output = zeros(size(x));

    N = length(x) / 6;
    disp(N)

    positions = x(1:3*N);
    velocities = x((3*N+1): end)

    output(1:3*N) = velocities;
    positions_reshaped = reshape(positions,[3,N]);
    accelerations_reshaped = zeros(3,N)

    for i=1:N
        for j=1:N
            if i~=j
                % compute the gravitational acceleration on i by j
                r_vec = positions_reshaped(1:3,j) - positions_reshaped(1:3,i)
                r_mag = norm(r_vec);
                a_ij = r_vec ./ r_mag^3;
                accelerations_reshaped(1:3,i) = accelerations_reshaped(1:3,i) + a_ij;
            end
        end
    end

    output(1+3*N:end) = accelerations_reshaped(:);

end