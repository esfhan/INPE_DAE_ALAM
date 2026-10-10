from pylab import *
from numpy import *
from scipy import *
from scipy.signal import *
from scipy.ndimage import *
#from scipy.stats import skewnorm
from scipy.special import erf

#=============================================================================#
def ndf(x,mu,sig):
    y=(x-mu)/sig
    return exp(-(y)**2./2)

def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)

#===
    
def step_f(i1,i2,ndata):
    f=zeros((ndata))
    f[i1:i2]=1.
    return f

#===Simulation Volume
re=6.371e+03
dx_km=10.;dy_km=dx_km
y=arange(80,880+dy_km,dy_km);ny=len(y)
dy=dy_km*1.e+03;dx=dx_km*1.e+03
wl=1.5e+03
x=arange(-int(2.5*wl),int(2.5*wl)+dx_km,dx_km);nx=len(x)

r=y+re;re0=re+300
dr=5;dphi=arctan(5*dr/re0);dtheta=5*dphi
wl_f=arctan(wl/re0)
fhi=arange(-1*wl_f,1*wl_f,dphi)
theta=arange(-2*dtheta,2.*dtheta+dtheta,dtheta)
[r3,fhi3,theta3]=meshgrid(r,fhi,theta)
h1=1+0*r3;h2=r3*cos(theta3);h3=r3
dy_km=h1*dr;dx_km=h2*dphi;dz_km=h3*dtheta
dy=dy_km*1.e+03;dx=dx_km*1.e+03;dz=dz_km*1.e+03
y=r;x=fhi;z=theta
nx=len(fhi);ny=len(r);nz=len(theta)
    
dt=10.

[y2,x2,z2]=meshgrid(y-re,x,z)
b_o=8.e+15         #Tm^3
delta=sqrt(1.+3.*(sin(theta3))**2.)
r3m=r3*1.e+03
b_r=-2*b_o*sin(theta3)/r3m**3.;b_theta=b_o*cos(theta3)/r3m**3.
b=b_o*delta/r3m**3.
ai=arccos(b_theta/b);

#===Ambient Ionosphere

np=1.e+12;yp=350;rp=y2/yp
a=2.;b=-2.5*3.#3.95,2.05,1.45
no_f=2.*np/(exp(a*(rp-1.))+exp(b*(rp-1.)))
al=3.;bl=-2.
rl=0.35;
no_l=1.e-03*np/(exp(al*(rp-rl))+exp(bl*(rp-rl)))
no_y=no_f+no_l#+no_top
n_o=no_y

gr_n=gradient(log(no_y))[1]/gradient(y2)[1]
sc_h=1./gr_n
i_pk=abs(y-yp+50.).argmin()
sc_mean=sc_h[:i_pk].mean()

#===Growth rate====

gr_x=0;gr_y=-9.8
eo_p=-2.e-03;eo_x=eo_p;eo_y=0;eo_z=0   #V/m
wo_x=0;wo_y=0

b_o=30.e-06    #Tesla
q_c=1.6e-19;m_i=1.67e-27;z_i=16.
gyro_i=q_c*b_o/(z_i*m_i)

nu_in=1.e+03*exp(-(y2-80.)/28.)  
nu_in[nu_in<=1.e-03]=1.e-03   #

nu_inert=0.25*abs(gr_y/(nu_in*sc_h*1.e+03))#
#nu_inert=abs(6.e+01/(sc_mean*1.e+03))
kappa=gyro_i/(nu_in+nu_inert)
mu_p=kappa/(b_o*(1.+kappa**2.))
mu_h=kappa**2./(b_o*(1.+kappa**2.))
mu_o=mu_p*(1.+kappa**2.)

nu_e=1.e+01*nu_in;gyro_e=16*1.837e+03*gyro_i
kappa_e=gyro_e/(nu_e+nu_inert)
mu_pe=kappa_e/(b_o*(1.+kappa_e**2.))
mu_he=kappa_e**2./(b_o*(1.+kappa_e**2.))
mu_oe=mu_pe*(1.+kappa_e**2.)

mu_11=mu_p;mu_12=mu_h;mu_13=mu_h
mu_21=-mu_h;mu_22=mu_p;mu_23=mu_h
mu_31=-mu_13;mu_32=mu_23;mu_33=mu_p

mue_11=mu_pe;mue_12=mu_he;mue_13=mu_he
mue_21=-mu_he;mue_22=mu_pe;mue_23=mu_he
mue_31=-mue_13;mue_32=mue_23;mue_33=mu_pe

i_o=1
ai=abs(ai)/1.
mu_11=mu_p;mu_12=mu_11*kappa*cos(ai);mu_13=mu_11*kappa*sin(ai)
mu_21=-mu_12;mu_22=mu_11*(1.+i_o*kappa**2.*sin(ai)**2.);
mu_23=i_o*mu_11*kappa**2.*cos(ai)*sin(ai)#-mu_11*kappa**2.*cos(ai)*sin(ai)   
mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+i_o*kappa**2.*cos(ai)**2.)

mue_11=mu_pe;mue_12=mue_11*kappa_e*cos(ai);mue_13=mue_11*kappa_e*sin(ai)
mue_21=-mue_12;mue_22=mue_11*(1.+i_o*kappa_e**2.*sin(ai)**2.);
mue_23=i_o*mue_11*kappa_e**2.*cos(ai)*sin(0*ai)#-mue_11*kappa_e**2.*cos(ai)*sin(ai)   
mue_31=-mue_13;mue_32=mue_23;mue_33=mue_11*(1.+i_o*kappa_e**2.*cos(ai)**2.)

vo_ex=mu_11*eo_x+mu_12*eo_y+mu_13*eo_z
vo_ey=mu_21*eo_x+mu_22*eo_y+mu_23*eo_z
vo_ez=mu_31*eo_x+mu_32*eo_y+mu_33*eo_z

#===Continuity equation 

n_0=n_o
n=zeros((nx,ny,nz));phi=zeros((nx,ny,nz));phi_o=phi;

it=0;dt=1.;t=0
dv_xo=0;dv_yo=0;vxo=vo_ex;vyo=vo_ey;
dv_x=0;dv_y=0*vo_ey;
var_evol=[];
n_evol=[]
phi_evol=[]
n0_evol=[]
v0_evol=[]

while abs(dv_y).max()< 60*vo_ey.max():
    
    x_shift=-0*arctan(400.*1.e-03/re0)*t*60.
    xo=x2.mean()-arctan(0./re0)*dx_km
    e1=-skew(x2,xo-x_shift,2*wl_f,-1.)/4.
    e2=skew(x2,xo-x_shift,wl_f/2.5,1.)/1.5
    e3=skew(x2,xo+arctan(25/re0)*dx_km-0*x_shift,wl_f/10,1.)/6.
    eoy=skew(y2,500,wl/2.,-3.)
    eox=skew(x2,0.0*dphi,wl_f/4.,0.)#*skew(y2,400,wl/10.,0.)
    eo_prvd=eo_p*(e1+e2)*eoy/1.
    wk_agw=2.*pi/arctan(600./re0)
    a1=50./100.;a2=5./100.
    eo_agw=eo_p*(a1*sin(wk_agw*x2)+0*a2*cos(wk_agw*x2))
    t_o=4000.;sigma_t=2000.;af=5.;bf=-2.;t_e=t*60.
    ele_t=4./(exp(af*(t_e/t_o-1.))+exp(bf*(t_e/t_o-1.))) #0.75
    ele_amb=-1.5e-03*ele_t
    eo_x=ele_amb+0*eo_prvd+eo_agw*(eox-eox[0,:,:])
    eo_y=0*eo_prvd#*skew(y2,500,wl/10.,0.)
    eo_z=0
    
    vo_ex=mu_11*eo_x+mu_12*eo_y+mu_13*eo_z
    vo_ey=mu_21*eo_x+mu_22*eo_y+mu_23*eo_z
    vo_ez=mu_31*eo_x+mu_32*eo_y+mu_33*eo_z

    fx=b_o*(eo_x/b_o+wo_y+gr_x/gyro_i)
    fy=b_o*(0*eo_y/b_o-wo_x+gr_y/gyro_i)
    fz=b_o*(0*eo_y/b_o-wo_x+gr_y/gyro_i)
    vo_x=mu_11*fx+mu_12*fy+mu_13*fz
    vo_y=mu_21*fx+mu_22*fy+mu_23*fz
    vo_z=mu_31*fx+mu_32*fy+mu_33*fz
    
    if t==0:
        vx=vo_ex;vy=vo_ey;vz=vo_ey
    fx_o=gradient(n_o*vx)[0]/2.
    fy_o=gradient(n_o*vy)[1]/2.
    fz_o=gradient(n_o*vz)[2]/2.
        
    ep_x=gradient(-phi)[0]/(2.*dx)
    ep_y=gradient(-phi)[1]/(2.*dy)
    ep_z=gradient(-phi)[2]/(2.*dz)
    
    dv_x=mu_11*ep_x+mu_12*ep_y+mu_13*ep_z
    dv_y=mu_21*ep_x+mu_22*ep_y+mu_23*ep_z
    dv_z=mu_31*ep_x+mu_32*ep_y+mu_33*ep_z
    
    vx=vo_ex+dv_x
    vy=vo_ey+dv_y
    vz=vo_ez+dv_z
    
    # if t<=60.:
    #     vx=dv_x;vy=dv_y
    
    vnx=dx/dt;vny=dy/dt;vnz=dz/dt
    
    fx_0=gradient(n_0*vo_ex)[0]/2.
    fy_0=gradient(n_0*vo_ey)[1]/2.
    fz_0=gradient(n_0*vo_ez)[2]/2.
    
    n_0=n_0-(fx_0/vnx+fy_0/vny+fz_0/vnz)   
    
    n=n_o
    for iter in range (11):
        fx_g=gradient(n*vx)[0]/2.
        fy_g=gradient(n*vy)[1]/2.
        fz_g=gradient(n*vz)[2]/2.
        fx=(fx_o+fx_g)/2.
        fy=(fy_o+fy_g)/2.
        fz=(fz_o+fz_g)/2.
        n=n_o-(fx/vnx+fy/vny+fz/vnz)
#        n[:,0]=n[:,1];n[:,-1]=n[:,-2];
        grad_n=gradient(n)[1]
        n[:,0,:]=n[:,1,:]-grad_n[:,2,:]/2.
        n[:,-1,:]=n[:,-2,:]+grad_n[:,-3,:]/2.
        # n[-1,:,:]=n[0,:,:]

#%%Potential Equation
    
    e_gx=b_o*gr_x/gyro_i;e_gy=b_o*gr_y/gyro_i;e_gz=0
    j_gx=q_c*n_o*(mu_11*e_gx+mu_12*e_gy+mu_12*e_gz)
    j_gy=q_c*n_o*(mu_21*e_gx+mu_22*e_gy+mu_23*e_gz)
    j_gz=q_c*n_o*(mu_31*e_gx+mu_32*e_gy+mu_33*e_gz)
    
    s11=q_c*n_o*(mu_11-mue_11);s12=q_c*n_o*(mu_12-mue_12);s13=q_c*n_o*(mu_13-mue_13);
    s21=q_c*n_o*(mu_21-mue_21);s22=q_c*n_o*(mu_22-mue_22);s23=q_c*n_o*(mu_23-mue_23);
    s31=q_c*n_o*(mu_31-mue_31);s32=q_c*n_o*(mu_32-mue_32);s33=q_c*n_o*(mu_33-mue_33);
    
    # s11=s_11;s12=s_12*cos(ai);s13=s_13*sin(ai)
    # s21=-s12;s22=s_11*cos(ai)**2.+s_33*sin(ai)**2.;s23=(s_11-s_33)*sin(ai)*cos(ai)
    # s31=-s13;s32=s23;s33=s_33*cos(ai)**2.+s_11*sin(ai)**2
       
    
    j_ex=s11*(eo_x+wo_y*b_o)+s12*(eo_y-wo_x*b_o)+s13*eo_z
    j_ey=s21*(eo_x+wo_y*b_o)+s22*(eo_y-wo_x*b_o)+s23*eo_z
    j_ez=s31*(eo_x+wo_y*b_o)+s32*(eo_y-wo_x*b_o)+s33*eo_z
    
    sg=gradient(j_gx)[0]/(2.*dx)+0*gradient(j_gy)[1]/(2.*dy)+0*gradient(j_gz)[2]/(2.*dz)
    se=gradient(j_ex)[0]/(2.*dx)+0*gradient(j_ey)[1]/(2.*dy)+0*gradient(j_ez)[2]/(2.*dz)
    s=(1./s11)*(sg+se)

#%%    Poisson equation
    dzm=1.e+00*dz
    dym=1.e+00*dy
    w=1.
        
    for iter in range (11):
        
        
        sxo=(roll(phi,1,axis=0)+roll(phi,-1,axis=0))/(2.*dx**2.)
        syo=(s22/s11)*(roll(phi,1,axis=1)+roll(phi,-1,axis=1))/(2.*dym**2.)
        szo=(s33/s11)*(roll(phi,1,axis=2)+roll(phi,-1,axis=2))/(2.*dzm**2.)
        
        sxg=(1./s11)*(gradient(s11)[0]/(2.*dx)+gradient(s21)[1]/(2.*dym)+
                      gradient(s31)[2]/(2.*dzm))*gradient(phi)[0]/(2.*dx)
        
        syg=(1./s11)*(gradient(s12)[0]/(2.*dx)+gradient(s22)[1]/(2.*dym)+
                      gradient(s32)[2]/(2.*dzm))*gradient(phi)[1]/(2.*dym)
        
        szg=(1./s11)*(gradient(s13)[0]/(2.*dx)+gradient(s23)[1]/(2.*dym)+
                      gradient(s33)[2]/(2.*dzm))*gradient(phi)[2]/(2.*dzm)
        
        smap=(1/s11)*(s12*gradient(gradient(phi)[0])[1]/(2.*dx*dym)
                      +s13*gradient(gradient(phi)[0])[2]/(2.*dx*dzm)
                      +s21*gradient(gradient(phi)[1])[0]/(2.*dym*dx)
                      +s23*gradient(gradient(phi)[1])[2]/(2.*dym*dzm)
                      +s31*gradient(gradient(phi)[2])[0]/(2.*dzm*dx)
                      +s32*gradient(gradient(phi)[2])[1]/(2.*dzm*dym))
        c_phi=(1./dx**2.)+(s22/s11)*(1./dym**2)+(s33/s11)*(1./dzm**2.)
        phi=(1./c_phi)*(sxo+syo+szo+sxg+syg+szg+0*smap-s)
        grad_n=gradient(phi)[1]
        phi[:,0,:]=phi[:,1,:]-grad_n[:,2,:]/2.
        phi[:,-1,:]=phi[:,-2,:]+grad_n[:,-3,:]/2.
        # phi[-1,:,:]=phi[0,:,:]
    
    dn=100*(n-n_0)/n_0;
    dn_n=abs(dn[:,20:]).max()#100.*(abs(n[nm,:]-n[nmx,:])/n[nmx,:]).max()

    var_evol.append((t,dn_n,dv_y.max(),vo_ey.max()))
    
    n_evol.append((n));
    phi_evol.append((phi));
    n0_evol.append((n_0));
    v0_evol.append((vo_ey));
    
    if abs(vy).max()<abs(vyo).max() and abs(dv_y).max()>5.*vo_ey.max():
        break
    
    n_o=n;phi_o=phi;vxo=vx;vyo=vy
    
    dt_s=0.1*dy.mean()/(vo_ey.max()+dv_y.max())
    dt=max(dt_s,0.01)
    
    t=t+dt/60.
    it=it+1
    
    print (dt_s,t,dv_y.max())
    
#    if dv_y.max()>3.e+03:
#        break
    ieq=abs(theta-0).argmin()   
    if remainder(it,10)==0:
        
        fig=figure(1,figsize=(12,12),facecolor='w',edgecolor='k')
        # ax=subplot(121)
        # pcolormesh(x,y,dn.T,cmap=cm.seismic,vmax=50,vmin=-50)
        ax=subplot(111)
        c_lb=linspace(n.min(),n.mean(),11)               
        contour(x2[:,:,ieq],y2[:,:,ieq],n[:,:,ieq],11,cmap=cm.gray)
        contour(x2[:,:,ieq],y2[:,:,ieq],phi[:,:,ieq],5,cmap=cm.seismic);
        pcolormesh(x2[:,:,ieq],y2[:,:,ieq],vo_y[:,:,ieq],cmap=cm.seismic,vmin=-50,vmax=50)
        plot(x,vo_ey[:,20,1],'g')
        #clabel(cs)
        title(str(round(t,2))+','+str(round(vo_y.max(),2))+', '\
              +str(round(dv_y.max(),2)))
#        draw()
        pause(0.01)
        clf()

fig=figure(1,figsize=(12,12),facecolor='w',edgecolor='k')          
# ax=subplot(121)         
# pcolormesh(x,y,dn.T,cmap=cm.seismic,vmax=50,vmin=-50)
ax=subplot(111)
c_lb=linspace(n.min(),n.mean(),11)               
contour(x2[:,:,ieq],y2[:,:,ieq],n[:,:,ieq],11,cmap=cm.gray)
contour(x2[:,:,ieq],y2[:,:,ieq],phi[:,:,ieq],5,cmap=cm.seismic);
pcolormesh(x2[:,:,ieq],y2[:,:,ieq],vo_y[:,:,ieq],cmap=cm.seismic,vmin=-50,vmax=50)
#clabel(cs)
title(str(round(t,2))+','+str(round(vo_y.max(),2))+', '\
      +str(round(dv_y.max(),2)))
        

var_evol=array(var_evol)
n=array(n_evol)
phi=array(phi_evol)
n_0=array(n0_evol)
v_0=array(v0_evol)
wl=1500
save(str(int(wl))+'km_'+'evol3d_dec'+'.npy',var_evol)
save('den3d_dec.npy',n)
save('phi3d_dec.npy',phi)
save('den03d_dec.npy',n_0)
save('v03d_dec.npy',v_0)


fig=figure(2,figsize=(12,12),facecolor='w',edgecolor='k')
subplot(121)
plot(var_evol[:,0],var_evol[:,1],lw=3)
subplot(122)
plot(var_evol[:,0],var_evol[:,2],lw=3)


# fig=figure(3,figsize=(12,12),facecolor='w',edgecolor='k')
# pcolormesh(x,var_evol[:,0],vo_yt);
# show()

