import numpy as np
import pysindy as ps
import scipy as sp

def run_centralforce_sindy(trajectory_filename):
    # Unpack the dataset into a convenient format
    nTraj,t,x = unpack_matlab_data(trajectory_filename)

    # print(t.shape)

    # Builds the "control"
    u = build_u_matrix(x,0)

    for i in range(nTraj):
        run_sindy(x,t,i)
        run_sindy_merge_xyz(x,t,i)

# Unpacks MATLAB data from a .mat file
def unpack_matlab_data(trajectory_filename):
    # Load in the .mat file
    data = sp.io.loadmat(trajectory_filename)

    # Check that only one variable is in the .mat file
    variableNames = list(data.keys())

    # Unpack the data into a more convenient format
    nTraj = len(data[variableNames[-1]][0])
    t = data[variableNames[-1]][0][0][0][:].flatten()
    x = data[variableNames[-1]][0][0][1][0:3,:]
    for i in range(1,nTraj):
        x = np.concatenate([x,data[variableNames[-1]][0][i][1][0:3,:]],axis=0)
    x = x.T

    # Return the unpacked data
    return nTraj,t,x

# Builds the "control" matrix (r_ij, xhat_ij, yhat_ij, zhat_ij) for a single i
def build_u_matrix(x,body_idx):

    u = np.array([]).reshape(x.shape[0],0)
    nTraj = round(x.shape[1]/3)
    
    # 
    for i in range(nTraj):
        # No self-interaction
        if i==body_idx:
            continue

        # Computes the delta, the magnitude of the delta, then normalize the delta
        delta = x[:,3*body_idx:3*(body_idx+1)]-x[:,3*i:3*(i+1)]
        r = np.linalg.norm(delta,axis=1)[:,np.newaxis]
        r_inv = np.pow(r,-1)
        hat = delta * r_inv

        u = np.concatenate([u,r_inv,hat],axis=1)
    
    return u


def run_sindy(x,t,body_idx):
    # Build a library involving vector hats and inverse powers of r
    poly_lib = ps.PolynomialLibrary(
        degree=2,
        include_bias=False)
    vectorHat_lib = ps.IdentityLibrary()
    tensorProd_lib = ps.TensoredLibrary(
        [poly_lib,vectorHat_lib],
        [[0],[1]]
    )

    # Build the differentiation model
    diff = ps.differentiation.FiniteDifference(
        order=2,
        d=2,
        axis=0)

    # Build the u-matrix
    u = build_u_matrix(x,body_idx)
    
    # Build a list of libraries to pass into the generalized library
    nTraj = round(x.shape[1]/3)
    libList = [tensorProd_lib]*(nTraj-1)

    # Iterate over xhat, yhat, zhat
    for i in range(3):

        # Build the inputs to create the generalized library
        idx_list = []
        for j in range(nTraj-1):
            idx_list.append([1+(4*j),2+i+(4*j)])
        # print(idx_list)

        # Build the appropriate generalized library
        lib = ps.GeneralizedLibrary(
            libList,
            None,
            idx_list
        )

        # Build the SINDy model
        model = ps.SINDy(
            feature_library=lib,
            optimizer=ps.optimizers.STLSQ(alpha=0.05),
            differentiation_method=diff,
        )

        # Fit the model
        # print((3*body_idx)+i)
        model.fit(
            x[:,(3*body_idx)+i],
            t=t,
            u=u
        )
        model.print()


# Builds a u-matrix of the form (r_ij, [x_ij,y_iij,z_ij])
# so that we can simultaneously run SINDy on all three vector
# components simultaneously
def build_merged_u_matrix(u):
    nTraj = round(u.shape[1]/4)
    u_merged = np.array([]).reshape(u.shape[0]*3,0)

    for i in range(nTraj):
        u_merged = np.concatenate([u_merged,np.tile(u[:,i*4,np.newaxis],[3,1])],axis=1)
        col = np.concatenate([u[:,(i*4)+1,np.newaxis],u[:,(i*4)+2,np.newaxis],u[:,(i*4)+3,np.newaxis]],axis=0)
        u_merged = np.concatenate([u_merged,col],axis=1)

    return u_merged

# Manually computes (and merges) the second derivative for the xhat,
# yhat, and zhat components of a particular body
def compute_merged_second_derivative(x,t,body_idx):
    dt = abs(t[1]-t[0])
    x_body = x[:,3*body_idx:3*(body_idx+1)]
    v_body = np.gradient(x_body,dt,axis=0,edge_order=2)
    a_body = np.gradient(v_body,dt,axis=0,edge_order=2)
    merged_a = np.ravel(a_body,'F')

    return merged_a


# Since we expect the same coefficients on the xyz everything,
# we can concatenate the data, the u-matrix, and the manually
# computed second derivative and run SINDy on all of them simulataneously
# *** REQUIRES UNIFORMLY SPACED DATA ***
def run_sindy_merge_xyz(x,t,body_idx):
    
    # Merge the xyz components into a single vector
    x_merged = np.ravel(x[:,3*body_idx:3*(body_idx+1)],'F')
    a_merged = compute_merged_second_derivative(x,t,body_idx)
    u = build_u_matrix(x,body_idx)
    u_merged = build_merged_u_matrix(u)

    # Build a library involving vector hats and inverse powers of r
    poly_lib = ps.PolynomialLibrary(
        degree=2,
        include_bias=False)
    vectorHat_lib = ps.IdentityLibrary()
    tensorProd_lib = ps.TensoredLibrary(
        [poly_lib,vectorHat_lib],
        [[0],[1]]
    )
    
    # Build the generalized library to plug into the model
    nTraj = round(x.shape[1]/3)
    libList = [tensorProd_lib]*(nTraj-1)
    idx_list = []
    for i in range(nTraj-1):
        idx_list.append([(2*i)+1,(2*i)+2])
    lib = ps.GeneralizedLibrary(
        libList,
        None,
        idx_list
    )
    
    # Create and fit the model
    model = ps.SINDy(
        feature_library=lib,
        optimizer=ps.optimizers.STLSQ(alpha=0.05),
        differentiation_method=ps.differentiation.FiniteDifference(),
    )
    model.fit(
        x_merged,
        t=abs(t[1]-t[0]),
        u=u_merged,
        x_dot=a_merged
    )
    model.print()

    
