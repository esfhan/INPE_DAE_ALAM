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
#===
    
def sum_gr(ndim,ndata,data):
    phi_2=0*data
    
    if ndim==0:
        for j in range (ndata):
            if j==0:
                phi_2[j,:]=(data[j+1,:]+data[j,:])/2.
            elif j==ndata-1:
                phi_2[j,:]=(data[j-1,:]+data[j,:])/2.
            else:
                phi_2[j,:]=data[j+1,:]+data[j-1,:]
    
    if ndim==1:
        for j in range (ndata):
            if j==0:
                phi_2[:,j]=(data[:,j+1]+data[:,j])/2.
            elif j==ndata-1:
                phi_2[:,j]=(data[:,j-1]+data[:,j])/2.
            else:
                phi_2[:,j]=data[:,j+1]+data[:,j-1]

    return phi_2#gradient(gradient(data)[ndim])[ndim]+2.*data

#===Simulation Plane

dx_km=20;dy_km=dx_km/2.
y=arange(150,750+dy_km,dy_km);ny=len(y)


dy=dy_km*1.e+03;dx=dx_km*1.e+03

#===Ambient Ionosphere

np=1.e+12;yp=400;r=y/yp
a=2.;b=-2.5*2.#3.95,2.05,1.45
no_f=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))
al=3.;bl=-2.
rl=0.35;
no_l=1.e-03*np/(exp(al*(r-rl))+exp(bl*(r-rl)))
no_y=no_f+no_l#+no_top

#no_y=1.e+12*skew(y,350,200,5)+5.e+11*skew(y,150,100,0)

gr_n=gradient(log(no_y))/gradient(y)
sc_h=1./gr_n
wl=1.5e+03
x=arange(-int(2.5*wl),int(8.5*wl)+dx_km,dx_km);nx=len(x)

n_o=empty((nx,ny))

for ix in range (nx):
    np=1.e+12;yp=400;r=y/yp
    a=2.;b=-2.5*2.*nx/(nx+ix)#3.95,2.05,1.45
    no_f=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))
    al=3.;bl=-2.
    rl=0.35;
    no_l=1.e-03*np/(exp(al*(r-rl))+exp(bl*(r-rl)))
    no_y=no_f+no_l#+no_top
    n_o[ix,:]=no_y#+no_y*no_x[ix]
    
#===
    
dt=10.

[y2,x2]=meshgrid(y,x)

#===Growth rate====

gr_x=0;gr_y=-9.8
eo_p=-2.e-03;eo_x=eo_p;eo_y=0    #V/m
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

nu_e=1.e+03*nu_in;gyro_e=1.837e+03*gyro_i
kappa_e=gyro_e/(nu_e+nu_inert)
mu_pe=kappa_e/(b_o*(1.+kappa_e**2.))
mu_he=kappa_e**2./(b_o*(1.+kappa_e**2.))
mu_oe=mu_pe*(1.+kappa_e**2.)

vo_ex=mu_p*eo_x+mu_h*eo_y
vo_ey=mu_p*eo_y-mu_h*eo_x

##
fx=b_o*(eo_x/b_o+wo_y+gr_x/gyro_i)
fy=b_o*(eo_y/b_o-wo_x+gr_y/gyro_i)

vo_x=mu_p*fx+mu_h*fy
vo_y=mu_p*fy-mu_h*fx

#===Continuity equation 

n_0=n_o
n=zeros((nx,ny));phi=zeros((nx,ny));phi_o=phi;

it=0;dt=100.;t=0
dv_xo=0;dv_yo=0;vxo=vo_ex;vyo=vo_ey;
dv_x=0;dv_y=0*vo_ey;
ep_x=0;ep_y=0
var_evol=[];
n_evol=[]
phi_evol=[]
n0_evol=[]
v0_evol=[]

while abs(dv_y).max()< 60*vo_ey.max():
    gr_n=1.e-03*gradient(log(n_o))[1]/gradient(y)
    nu_inert=0.5*abs((dv_y+gr_y/nu_in)*gr_n.mean())#
    kappa=gyro_i/(nu_in+nu_inert)
    mu_p=kappa/(b_o*(1.+kappa**2.))
    mu_h=kappa**2./(b_o*(1.+kappa**2.))
    mu_o=mu_p*(1.+kappa**2.)
    # t0=0+45.
    # tau=wl*1000./(400.*60.)
    # t_shift=0;
    # e1=-skew(t,t0-t_shift,tau,0.)/2.
    # e2=skew(t,t0-t_shift,tau/2.5,1.)
    # eo_t=(e1+e2);
    # figure (11)
    # plot(t,eo_t,'o')
    # draw()
    x_shift=-400.*t*60.*1.e-03
    xo=x2.mean()-250.*dx_km
    e1=-skew(x2,xo-x_shift,4*wl,-1.)/1.
    e2=skew(x2,xo-x_shift,wl/2,2.)/1.
    e3=skew(x2,xo+400*dx_km-x_shift,wl/2,2.)/2.
    eoy=skew(y2,400,wl/2.,-1.)
    eo_prvd=eo_p*(e1+e2+e3)#*eoy/1.
    wk_agw=2.*pi/900.
    a1=20./100.;a2=5./100.
    eoy=0+skew(y2,300,200.,-2.)
    a1=1.*a1*(eoy/eoy.max())
    imx=abs(a1-a1.max()).argmin()
    #a1[y2<y[imx]]=10./100.
   
    ydelta=2*(y2-y2.min())/y2.mean()
    v_agw=60*1.e-03
    omega=wk_agw*v_agw
    eo_agw=1*eo_p*a1*sin(wk_agw*(x2-(xo-x_shift))+0*ydelta*pi/2-0*omega*t*60)
   
    eo_x=eo_p+eo_prvd+eo_agw
    eo_x=eo_x/2.
    eo_y=0*eo_prvd#*skew(y2,500,wl/10.,0.)
    
    vo_ex=mu_p*eo_x+mu_h*eo_y
    vo_ey=mu_p*eo_y-mu_h*eo_x

    fx=b_o*(eo_x/b_o+wo_y+gr_x/gyro_i)
    fy=b_o*(0*eo_y/b_o-wo_x+gr_y/gyro_i)

    vo_x=mu_p*fx+mu_h*fy
    vo_y=mu_p*fy-mu_h*fx
    

    if t==0:
        vx=vo_ex;vy=vo_ey
    fx_o=gradient(n_o*vx)[0]/2.
    fy_o=gradient(n_o*vy)[1]/2.
    
    ep_xp=ep_x;ep_yp=ep_y
    
    ep_x=gradient(-phi)[0]/(2.*dx)
    ep_y=gradient(-phi)[1]/(2.*dy)
    
    dv_x=mu_p*ep_x+mu_h*ep_y
    dv_y=mu_p*ep_y-mu_h*ep_x
    
    
    vx=vo_ex+dv_x
    vy=vo_ey+dv_y
    
    
    vnx=dx/dt;vny=dy/dt
    
    fx_0=gradient(n_0*vo_ex)[0]/2.
    fy_0=gradient(n_0*vo_ey)[1]/2.
    
    n_0=n_0-(fx_0/vnx+fy_0/vny)   
    
    n=n_o
    for iter in range (31):
        fx_g=gradient(n*vx)[0]/2.
        fy_g=gradient(n*vy)[1]/2.
        fx=(fx_o+fx_g)/2.
        fy=(fy_o+fy_g)/2.
        n=n_o-(fx/vnx+fy/vny)
#        n[:,0]=n[:,1];n[:,-1]=n[:,-2];
        grad_n=gradient(n)[1]
        n[:,0]=n[:,1]-grad_n[:,2]/2.
        n[:,-1]=n[:,-2]+grad_n[:,-3]/2.
        
    n_bottom=n[:,0].mean()
    a=gradient(n)[0]/(dx*sum_gr(0,nx,n))
    b=gradient(n)[1]/(dy*sum_gr(1,ny,n))
    
    sc_ht=1.e-03/b
    r_ea=6.371e+03;lat=0.*pi/180.
    theta_m=arccos(sqrt(y2/(r_ea+y2)))
    print ('Theta=',degrees(theta_m))
    sc_o=1.e+03*cos(lat)*pi*(r_ea+y2)/sqrt(1.+3.*(sin(theta_m))**2.)#1.e+04*sc_mean
#    s_pl=(n_bottom/n)*(mu_o.max()/mu_p.max())*(sc_ht/sc_o)**2.
    #sigma_o_0=sigma_o[:,0].mean()
    
    sigma_o=(mu_o-mu_oe);sigma_p=(mu_p-mu_pe)
    s_pl=(sigma_o/sigma_p)*(sc_ht/sc_o)**2.
    l_per=1.e+00;l_par=1.e+04#/(1+1*abs(x2-(xo-x_shift))/x2.max())
    l_r=(l_per/l_par)**2.
    s_pl=(abs(n-n_o)/n_o)*(sigma_o/sigma_p)*l_r
    s_o=1./(1.+abs(s_pl))
    s_x=vo_x*a/mu_p
    s_y=vo_y*b/mu_p
    s=s_o*s_x
    
    e_gx=b_o*gr_x/gyro_i;e_gy=b_o*gr_y/gyro_i
    j_gx=q_c*n_o*(mu_p*e_gx+mu_h*e_gy)
    j_gy=q_c*n_o*(mu_p*e_gy-mu_h*e_gx)
    sigma_p=q_c*n_o*(mu_p-mu_pe)
    sigma_h=q_c*n_o*(mu_h-mu_he)
    j_ex=sigma_p*(eo_x+wo_y*b_o)+sigma_h*(eo_y-wo_x*b_o)
    j_ey=sigma_p*(eo_y-wo_x*b_o)-sigma_h*(eo_x+wo_y*b_o)
    # j_px=((n-n_o)*m_i/b_o**2.)*(ep_x-ep_xp)/dt
    # j_py=((n-n_o)*m_i/b_o**2.)*(ep_y-ep_yp)/dt
    dj_x=(sigma_p*ep_x+sigma_h*ep_y)*(n-n_o)/n_o
    dj_y=(sigma_p*ep_y-sigma_h*ep_x)*(n-n_o)/n_o
    j_px=(1./(nu_in+nu_inert))*dj_x/dt
    j_py=(1./(nu_in+nu_inert))*dj_y/dt
    sg=gradient(j_gx)[0]/(2.*dx)+gradient(j_gy)[1]/(2.*dy)
    se=gradient(j_ex)[0]/(2.*dx)+gradient(j_ey)[1]/(2.*dy)
    sp=gradient(j_px)[0]/(2.*dx)+gradient(j_py)[1]/(2.*dy)
    
    s=s_o*(1./sigma_p)*(sg+se+sp)
    
#===Poisson equation

    w=1.
        
    for iter in range (31):
        
        phi_x=a*gradient(phi)[0]/(2.*dx)
        phi_y=b*gradient(phi)[1]/(2.*dy)
        
        phi_2x=sum_gr(0,nx,phi)
        phi_2y=sum_gr(1,ny,phi)
        
        phi_x2=phi_2x/(2.*dx**2.)
        phi_y2=phi_2y/(2.*dy**2.)
        
        c=-(1./dx**2.)-(1./dy**2)
        phi=(1./c)*(s-phi_x-phi_y-phi_x2-phi_y2)
        
    
    dn=100*(n-n_0)/n_0;
    nm=int(nx/2.);nmx=nm+int(wl/(4.*dx_km))
    dn_n=abs(dn[:,20:]).max()#100.*(abs(n[nm,:]-n[nmx,:])/n[nmx,:]).max()

    var_evol.append((t,dn_n,dv_y.max(),vo_ey.max()))
    
    n_evol.append((n));
    phi_evol.append((phi));
    n0_evol.append((n_0));
    v0_evol.append((vo_ey));
    
    if abs(vy).max()<abs(vyo).max() and abs(dv_y).max()>5.*vo_ey.max():
        break
    
    n_o=n;phi_o=phi;vxo=vx;vyo=vy
    
    dt_s=0.1*dy/(vo_ey.max()+dv_y.max())
    dt=max(dt_s,0.01)
    
    t=t+dt/60.
    it=it+1
    
    print (dt_s,t,dv_y.max())
    
#    if dv_y.max()>3.e+03:
#        break
    
    if remainder(it,10)==0:
        
        fig=figure(1,figsize=(12,12),facecolor='w',edgecolor='k')
        # ax=subplot(121)
        # pcolormesh(x,y,dn.T,cmap=cm.seismic,vmax=50,vmin=-50)
        ax=subplot(111)
        c_lb=linspace(n.min(),n.mean(),11)               
        contour(x,y,n.T,11,cmap=cm.gray)
        #contour(x,y,phi.T,5,cmap=cm.seismic);
        pcolormesh(x,y,vo_y.T,cmap=cm.seismic,vmin=-50,vmax=50)
        plot(x,vo_ey[:,20],'g')
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
contour(x,y,n.T,11,cmap=cm.gray)
#contour(x,y,phi.T,5,cmap=cm.seismic);
pcolormesh(x,y,vo_y.T,cmap=cm.seismic,vmin=-100,vmax=100)
#clabel(cs)
title(str(round(t,2))+','+str(round(vo_y.max(),2))+', '\
      +str(round(dv_y.max(),2)))
        

var_evol=array(var_evol)
n=array(n_evol)
phi=array(phi_evol)
n_0=array(n0_evol)
v_0=array(v0_evol)
wl=1500
case=['dec_e1','jan_e2','dec_e3']
save(str(int(wl))+'km_evol3d_'+case[1]+'.npy',var_evol)
save('den3d_'+case[1]+'.npy',n)
save('phi3d_'+case[1]+'.npy',phi)
save('den03d_'+case[1]+'.npy',n_0)
save('v03d_'+case[1]+'.npy',v_0)


fig=figure(2,figsize=(12,12),facecolor='w',edgecolor='k')
subplot(121)
plot(var_evol[:,0],var_evol[:,1],lw=3)
subplot(122)
plot(var_evol[:,0],var_evol[:,2],lw=3)


fig=figure(3,figsize=(12,12),facecolor='w',edgecolor='k')
pcolormesh(x,var_evol[:,0],vo_yt);
show()

