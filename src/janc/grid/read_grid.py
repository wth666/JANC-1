import jax.numpy as jnp
import scipy.io as sio
import CGNS.MAP
import CGNS.PAT.cgnsutils as CGU

J = 1.0
dxi_dx = 1.0
deta_dx = 0.0
dxi_dy = 0.0
deta_dy = 1.0
nx_L = -1.0
ny_L = 0.0
nx_R = 1.0
ny_R = 0.0
nx_B = 0.0
ny_B = -1.0
nx_U = 0.0
ny_U = 1.0
dxi = 1.0
deta = 1.0

def read_CGNSt(file_path):
    global J,dxi_dx,deta_dx,dxi_dy,deta_dy,nx_U,ny_U,dxi,deta
    data = sio.loadmat(file_path)
    J = data['J'][None,:,:]
    dxi_dx = data['kx'][None,:,:]
    dxi_dy = data['ky'][None,:,:]
    deta_dx = data['vx'][None,:,:]
    deta_dy = data['vy'][None,:,:]
    nx_U = data['n'][:,0][None,3:-3,None]
    ny_U = data['n'][:,1][None,3:-3,None]
    dxi = 0.0095
    deta = 0.0095



def compute_metrics(X, Y):
    Ni, Nj = X.shape
    dxi = 1.0 / (Ni - 1)
    deta = 1.0 / (Nj - 1)
    # 中心差分（内部）
    dx_dxi = (X[2:,:] - X[:-2,:]) / (2*dxi)
    dx_dxi = jnp.concatenate([(X[1:2,:]-X[0:1,:])/dxi,dx_dxi,(X[-1:,:]-X[-2:-1,:])/dxi],axis=0)

    dx_deta = (X[:,2:] - X[:,:-2]) / (2*deta)
    dx_deta = jnp.concatenate([(X[:,1:2]-X[:,0:1])/deta,dx_deta,(X[:,-1:]-X[:,-2:-1])/deta],axis=1)

    dy_dxi = (Y[2:,:] - Y[:-2,:]) / (2*dxi)
    dy_dxi = jnp.concatenate([(Y[1:2,:]-Y[0:1,:])/dxi,dy_dxi,(Y[-1:,:]-Y[-2:-1,:])/dxi],axis=0)

    dy_deta = (Y[:,2:] - Y[:,:-2]) / (2*deta)
    dy_deta = jnp.concatenate([(Y[:,1:2]-Y[:,0:1])/deta,dy_deta,(Y[:,-1:]-Y[:,-2:-1])/deta],axis=1)
    
    #边界法向量计算
    #left_boundary
    theta_L = jnp.atan(dy_deta[0,:]/dx_deta[0,:]) + jnp.pi/2
    #right_boundary
    theta_R = jnp.atan(dy_deta[-1,:]/dx_deta[-1,:]) - jnp.pi/2
    #bottom_boundary
    theta_B = jnp.atan(dy_dxi[:,0]/dx_dxi[:,0]) + jnp.pi/2
    #up_boundary
    theta_U = jnp.atan(dy_dxi[:,-1]/dx_dxi[:,-1]) - jnp.pi/2
    
    nx_L = jnp.cos(theta_L)[None,None,3:-3]
    ny_L = jnp.sin(theta_L)[None,None,3:-3]
    
    nx_R = jnp.cos(theta_R)[None,None,3:-3]
    ny_R = jnp.sin(theta_R)[None,None,3:-3]
    
    nx_B = jnp.cos(theta_B)[None,3:-3,None]
    ny_B = jnp.sin(theta_B)[None,3:-3,None]
    
    nx_U = jnp.cos(theta_U)[None,3:-3,None]
    ny_U = jnp.sin(theta_U)[None,3:-3,None]
    
    
    
    # Jacobian（内部节点）
    J = dx_dxi * dy_deta - dx_deta * dy_dxi
    dxi_dx = dy_deta/J
    deta_dx = -dy_dxi/J
    dxi_dy = -dx_deta/J
    deta_dy = dx_dxi/J
    
    #J = jnp.pad(J,pad_width=(2,2),mode='edge')[None,:,:]
    #dxi_dx = jnp.pad(dxi_dx,pad_width=(2,2),mode='edge')[None,:,:]
    #deta_dx = jnp.pad(deta_dx,pad_width=(2,2),mode='edge')[None,:,:]
    #dxi_dy = jnp.pad(dxi_dy,pad_width=(2,2),mode='edge')[None,:,:]
    #deta_dy = jnp.pad(deta_dy,pad_width=(2,2),mode='edge')[None,:,:]
    
    return J[None,:,:], dxi_dx[None,:,:], deta_dx[None,:,:], dxi_dy[None,:,:], deta_dy[None,:,:], dxi, deta,nx_L,ny_L,nx_R,ny_R,nx_U,ny_U,nx_B,ny_B


def read_CGNS(file_path=None):
    global J, dxi_dx, deta_dx, dxi_dy, deta_dy, dxi, deta, nx_L,ny_L,nx_R,ny_R,nx_U,ny_U,nx_B,ny_B
    if file_path is not None:
        tree, links, paths = CGNS.MAP.load(file_path)
        gc_path = "/Base/dom-1/GridCoordinates"
        # 拿到那个节点（节点列表）
        gridcoord = CGU.getNodeByPath(tree, gc_path)
        if gridcoord is None:
            raise KeyError(f"路径 {gc_path} 找不到节点")
        
        # 然后分别取子节点 CoordinateX 和 CoordinateY
        node_x = CGU.getNodeByPath(gridcoord, "CoordinateX")
        node_y = CGU.getNodeByPath(gridcoord, "CoordinateY")
        if node_x is None or node_y is None:
            raise KeyError("在该 GridCoordinates 组下未找到 CoordinateX/CoordinateY")
        
        X = jnp.array(node_x[1])
        Y = jnp.array(node_y[1])
        J, dxi_dx, deta_dx, dxi_dy, deta_dy, dxi, deta, nx_L,ny_L,nx_R,ny_R,nx_U,ny_U,nx_B,ny_B = compute_metrics(X, Y)
