import jax.numpy as jnp
from ...model import thermo_model as thermo

#current versions mainly use AUSMPW+ scheme
def split_M_and_P(M):
	indicator = (jnp.abs(M)<=1)
	Mplus = jnp.where(indicator,1/4*(M+1)**2,1/2*(M+jnp.abs(M)))
	Mminus = jnp.where(indicator,-1/4*(M-1)**2,1/2*(M-jnp.abs(M)))
	Pplus = jnp.where(indicator,1/4*(M+1)**2*(2-M),1/2*(1+jnp.sign(M)))
	Pminus = jnp.where(indicator,1/4*(M-1)**2*(2+M),1/2*(1-jnp.sign(M)))
	return Mplus,Mminus,Pplus,Pminus

def fLR(M_L,M_R,p_L,p_R,p_s):
    indicator_L = jnp.abs(M_L)<1.0
    f_L = jnp.where(indicator_L,(p_L/p_s-1),jnp.zeros_like(p_L))
    indicator_R = jnp.abs(M_R)<1.0
    f_R = jnp.where(indicator_R,(p_R/p_s-1),jnp.zeros_like(p_R))
    return f_L,f_R

def w_pL_pR(p_L,p_R):
	mask = jnp.minimum(p_L/p_R,p_R/p_L)
	return 1 - mask**3
	
def inteface_mach_number(q_L_x,q_R_x,q_L_y,q_R_y,h_L_x,h_R_x,h_L_y,h_R_y,gamma_L_x,gamma_R_x,gamma_L_y,gamma_R_y):
    rho_L = q_L_x[0:1]
    u_L,v_L = q_L_x[1:2],q_L_x[2:3]
    V_square = (u_L)**2
    p_L_x = q_L_x[3:4]
    gamma_L = gamma_L_x#p_L_x/rhoe + 1 #rhoE shold mius kinetic!!!
    H_L = h_L_x + 0.5*V_square
    
    rho_R = q_R_x[0:1]
    u_R,v_R = q_R_x[1:2],q_R_x[2:3]
    V_square = (u_R)**2
    p_R_x = q_R_x[3:4]
    gamma_R = gamma_R_x#p_L_x/rhoe + 1 #rhoE shold mius kinetic!!!
    H_R = h_R_x + 0.5*V_square
    
    Hnormal = 0.5*(H_L + H_R)
    gamma = 0.5*(gamma_L+gamma_R)
    cs_x = jnp.sqrt(2*(gamma-1)/(gamma+1)*Hnormal)
    indicator_x = (1/2*(u_L+u_R)>0)
    dom_L = jnp.maximum(jnp.abs(u_L),cs_x)
    dom_R = jnp.maximum(jnp.abs(u_R),cs_x)
    c_L_x = jnp.sqrt(gamma_L*p_L_x/rho_L)
    c_R_x = jnp.sqrt(gamma_R*p_R_x/rho_R)
    c_interface_x = 0.5*(c_L_x+c_R_x)
    M_L_x = u_L/c_interface_x
    M_R_x = u_R/c_interface_x
    #cs_x = jnp.sqrt(jnp.abs((2*Hnormal*(gamma_L-1)/(gamma_R+1)*(rho_L/gamma_L-rho_R/gamma_R*(gamma_R-1)/(gamma_L-1)+1e-10)/(rho_L/gamma_R-rho_R/gamma_L*(gamma_L+1)/(gamma_R+1)+1e-10))))
    #cs_x = jnp.sqrt(2*Hnormal*((gamma_L-1)/gamma_L*rho_L-(gamma_R-1)/gamma_R*rho_R + 1e-10)/((gamma_R+1)/gamma_R*rho_L-(gamma_L+1)/gamma_L*rho_R + 1e-10))
    rho_L = q_L_y[0:1]
    u_L,v_L = q_L_y[1:2],q_L_y[2:3]
    V_square = (v_L)**2
    p_L_y = q_L_y[3:4]
    gamma_L = gamma_L_y#p_L_x/rhoe + 1 #rhoE shold mius kinetic!!!
    H_L = h_L_y + 0.5*V_square
    
    rho_R = q_R_y[0:1]
    u_R,v_R = q_R_y[1:2],q_R_y[2:3]
    V_square = (v_R)**2
    p_R_y = q_R_y[3:4]
    gamma_R = gamma_R_x#p_L_x/rhoe + 1 #rhoE shold mius kinetic!!!
    H_R = h_R_y + 0.5*V_square
    
    Hnormal = 0.5*(H_L + H_R)
    gamma = 0.5*(gamma_L+gamma_R)
    cs_y = jnp.sqrt(2*(gamma-1)/(gamma+1)*Hnormal)
    indicator_y = (1/2*(v_L+v_R)>0)
    dom_L = jnp.maximum(jnp.abs(v_L),cs_y)
    dom_R = jnp.maximum(jnp.abs(v_R),cs_y)
    c_L_y = jnp.sqrt(gamma_L*p_L_y/rho_L)
    c_R_y = jnp.sqrt(gamma_R*p_R_y/rho_R)
    c_interface_y = 0.5*(c_L_y+c_R_y)
    M_L_y = v_L/c_interface_y
    M_R_y = v_R/c_interface_y
	
    return M_L_x,M_R_x,M_L_y,M_R_y,c_interface_x,c_interface_y

def riemann_flux(q_L_x,q_R_x,q_L_y,q_R_y):
	#interpolate cell interface values
    rho_L_x,rho_R_x,rho_L_y,rho_R_y =  q_L_x[0:1],q_R_x[0:1],q_L_y[0:1],q_R_y[0:1] 
    p_L_x,p_R_x,p_L_y,p_R_y = q_L_x[3:4],q_R_x[3:4],q_L_y[3:4],q_R_y[3:4]
    Y_L_x,Y_R_x,Y_L_y,Y_R_y = q_L_x[4:],q_R_x[4:],q_L_y[4:],q_R_y[4:]
    R_L_x,R_R_x,R_L_y,R_R_y = thermo.get_R(q_L_x[4:]),thermo.get_R(q_R_x[4:]),thermo.get_R(q_L_y[4:]),thermo.get_R(q_R_y[4:])
    T_L_x,T_R_x,T_L_y,T_R_y = p_L_x/(R_L_x*rho_L_x),p_R_x/(R_R_x*rho_R_x),p_L_y/(R_L_y*rho_L_y),p_R_y/(R_R_y*rho_R_y)
    _,gamma_L_x,h_L_x,_,_ = thermo.get_thermo(T_L_x,Y_L_x)
    _,gamma_R_x,h_R_x,_,_ = thermo.get_thermo(T_R_x,Y_R_x)
    _,gamma_L_y,h_L_y,_,_ = thermo.get_thermo(T_L_y,Y_L_y)
    _,gamma_R_y,h_R_y,_,_ = thermo.get_thermo(T_R_y,Y_R_y)
    M_L_x,M_R_x,M_L_y,M_R_y,c_interface_x,c_interface_y = inteface_mach_number(q_L_x,q_R_x,q_L_y,q_R_y,h_L_x,h_R_x,h_L_y,h_R_y,gamma_L_x,gamma_R_x,gamma_L_y,gamma_R_y)
    phi_L_x = jnp.concatenate([rho_L_x,rho_L_x*q_L_x[1:3],
                             rho_L_x*h_L_x+0.5*rho_L_x*(q_L_x[1:2]**2+q_L_x[2:3]**2),rho_L_x*Y_L_x],axis=0)
    phi_R_x = jnp.concatenate([rho_R_x,rho_R_x*q_R_x[1:3],
                             rho_R_x*h_R_x+0.5*rho_R_x*(q_R_x[1:2]**2+q_R_x[2:3]**2),rho_R_x*Y_R_x],axis=0)
    phi_L_y = jnp.concatenate([rho_L_y,rho_L_y*q_L_y[1:3],
                             rho_L_y*h_L_y+0.5*rho_L_y*(q_L_y[1:2]**2+q_L_y[2:3]**2),rho_L_y*Y_L_y],axis=0)
    phi_R_y = jnp.concatenate([rho_R_y,rho_R_y*q_R_y[1:3],
                             rho_R_y*h_R_y+0.5*rho_R_y*(q_R_y[1:2]**2+q_R_y[2:3]**2),rho_R_y*Y_R_y],axis=0)	
	#get x-flux
    M_L_plus,_,P_L_plus,_ = split_M_and_P(M_L_x)
    _,M_R_minus,_,P_R_minus = split_M_and_P(M_R_x)
    p_s = P_L_plus*p_L_x + P_R_minus*p_R_x
    w = w_pL_pR(p_L_x,p_R_x)
    fL,fR = fLR(M_L_x,M_R_x,p_L_x,p_R_x,p_s)
    m_half = M_L_plus + M_R_minus
    indicator = m_half>=0
    M_L_plus_av = jnp.where(indicator,M_L_plus+M_R_minus*((1-w)*(1+fR)-fL),M_L_plus*w*(1+fL))
    M_R_minus_av = jnp.where(indicator,M_R_minus*w*(1+fR),M_R_minus+M_L_plus*((1-w)*(1+fL)-fR))
    P_L = jnp.zeros_like(phi_L_x).at[1:2].set(p_L_x)
    P_R = jnp.zeros_like(phi_R_x).at[1:2].set(p_R_x)
    F_interface = M_L_plus_av*c_interface_x*phi_L_x + M_R_minus_av*c_interface_x*phi_R_x + (P_L_plus*P_L + P_R_minus*P_R)

	
	#get y-flux
    M_L_plus,_,P_L_plus,_ = split_M_and_P(M_L_y)
    _,M_R_minus,_,P_R_minus = split_M_and_P(M_R_y)
    p_s = P_L_plus*p_L_y + P_R_minus*p_R_y
    w = w_pL_pR(p_L_y,p_R_y)
   	#fL,fR = fLR(p1_L_y,p1_R_y,p_L_y,p_R_y,p2_L_y,p2_R_y,p_s)
    fL,fR = fLR(M_L_y,M_R_y,p_L_y,p_R_y,p_s)
    m_half = M_L_plus + M_R_minus
    indicator = m_half>=0
    M_L_plus_av = jnp.where(indicator,M_L_plus+M_R_minus*((1-w)*(1+fR)-fL),M_L_plus*w*(1+fL))
    M_R_minus_av = jnp.where(indicator,M_R_minus*w*(1+fR),M_R_minus+M_L_plus*((1-w)*(1+fL)-fR))
    P_L = jnp.zeros_like(phi_L_y).at[2:3].set(p_L_y)
    P_R = jnp.zeros_like(phi_R_y).at[2:3].set(p_R_y)
    G_interface = M_L_plus_av*c_interface_y*phi_L_y + M_R_minus_av*c_interface_y*phi_R_y + (P_L_plus*P_L + P_R_minus*P_R)
		
    return F_interface,G_interface

