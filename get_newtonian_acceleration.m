function accelerations = get_newtonian_acceleration(positions,masses)
%GET_NEWTONIAN_ACCELERATIONS Summary of this function goes here
%   Detailed explanation goes here

arguments (Input)
    positions
    masses
end

arguments (Output)
    accelerations
end

N = length(masses);
assert(all(size(positions)==[3,N]));
accelerations = zeros(3,N);

% For each pair of particles, compute the acceleration on particle i as a
% consequence of the gravity of particle j
for i=1:N
    for j=1:N
        if i~=j
            r_vec = positions(:,j) - positions(:,i);
            r_mag = norm(r_vec);
            a_ij = masses(j) * r_vec ./ (r_mag^3);
            accelerations(:,i) = accelerations(:,i) + a_ij;
        end
    end
end

end