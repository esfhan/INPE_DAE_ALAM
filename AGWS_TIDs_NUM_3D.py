#ESTE CÓDIGO RESOLVE AS EQUAÇÕES DADA POR KHERANI ET AL (2012, DOI: 10.1111/j.1365-246X.2012.05617.x)..
#(1) EQUAÇÃO DA ONDA PARA AMPLITUDE (W):
#d2w/dt2=(1/rho)grad(1.4 pn div.W)-((grad pn)/rho)div.(rho W)+(1/rho)grad(W.div)p+d((mu/rho)div.div (W)/dt-d(W.div W)/dt
#(2) EQUAÇÃO DA DENSIDADE (rho)
#d rho/dt+div.(rho W)=0
#(3) EQUAÇÃO DA ENERGIA OU PRESSÃO (pn)
#d pn/dt+div. (pn W)+(1.4-1)div. W=0
#ESTE CODIGO NUMERICO É COM ERRO DE SEGUNDA ORDEM EM SPAÇO..
#PORTANTO, EXISTE POSIBILIDADE DE MELHORAR...
#TEMBÉM, ESTE CÓDIGO EMPREGA METODO GAUSS-SEIDEL A RESOLVER A EQUAÇÃO DE
#MATRIZ. ESTE MÉTODO É SUBJETIVO..
#O CÓDIGO PODE REPRODUZ OBSEERVAÇÕES ATE 70%-80% QUALITATIVAMENTE..

#O CODIGO USA MKS UNIT E RESOLVE AS EQUAÇÕES EM O PLANO (X-Y) 
#QUE RERESENTA (LONGITUDE-ALTITUDE) OU (LATITUDE-ALTITUDE)
#ESTE CÓDIGO USA FORÇANTE NO SOLO (Y=0 KM) PARA EXCITAR AS ONDAS
#ESTE FORÇANTE VARIA NO TEMPO E NO X COMO GAUSSINAO
#FORÇANTE É DA CARATER MECANICAL ISTO É DE FORMA DE VENTO VERTICAL..
#O CÓDIGO FUNCIONA BEM COM dt=dy e dy<=dx<=2.*dy e 5km<=dy<=10 km
#OS CONTORNOS DAS LONGITUDES OU LATITUDES DEVERIA SER MAIS AFASTADA DE LOCALIZAÇÃO DA FORÇANTE PARA EVITAR AS REFELXÕES DAS ONDAS..
#O CONTORNO SUPERIOR (YMAX) DEVERIA SER IQUAL OU MAIOR DO 400 KM. 
   
from pylab import *
from numpy import *
#from nrlmsise_2000 import *
from scipy import *
from scipy.ndimage import *
from scipy.special import erf
from datetime import timedelta
from datetime import datetime
from nrlmsise00.dataset import msise_4d

close('all')
def d1_2(n,data):
    return repeat(data[newaxis,:],n,axis=0)
def d1_3(n,data):
    return repeat(data[newaxis,:,:],n,axis=0)

#%%
def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)

#%%
def div_f(f0,f1,f2):
    return gradient(f0)[0]/dx2+gradient(f1)[1]/dz2+gradient(f2)[2]/dy2 

#%%
def data_antes(ndim,ndata,data):
    data_n=0*data
    if ndim==2:
        data_n[:,:,1:-1]=data[:,:,0:-2]
        data_n[:,:,0]=data_n[:,:,1];data_n[:,:,-1]=data_n[:,:,-2];
    return data_n

#%%
def ambiente_atmos(y2):
    global rho_amb,tn_amb,r_g,nu_nn,lambda_c
    lat_ep=38;lon_ep=143
    # year,month,dom=2011,3,9
            
    # d0 = datetime.date(year,1,1)
    # d1 = datetime.date(year, month, dom)
    # delta = d1 - d0
    # doy=delta.days 
    # ut=17.55;lt=ut+lon_ep/15.
    
    # f107A,f107,ap=150,150,4
    
    # f=nrl_msis(doy,ut*3600.,lt,f107A,f107,ap,lat_ep,lon_ep,y)
    # tn_msis=f[1];
    # den_ox=f[2]*1.e+06;den_n=f[3]*1.e+06;den_o2=f[4]*1.e+06;den_n2=f[5]*1.e+06;
    # n_msis=den_ox+den_n+den_o2+den_n2;
    # rho_msis=f[6]*1.e+03
    # mean_mass=rho_msis/n_msis
    
    ds = msise_4d(datetime(2011, 3, 9, 17, 30, 10), y, lat_ep, lon_ep)
    tn_msis=ds['Talt'][0,:,0,0].values
    n_msis=1.e+06*(ds['O']+ds['O2']+ds['N2'])[0,:,0,0].values
    rho_msis=1.e+03*ds['rho'][0,:,0,0].values
    mean_mass=rho_msis/n_msis
    print (y,rho_msis)
    
    b_c=1.38e-23; 
    rg_msis=b_c/mean_mass
    pn_msis=rg_msis*rho_msis*tn_msis
    sn_msis=sqrt(1.4*pn_msis/rho_msis)
    
    nu_msis=pi*(7*5.6e-11)**2.*sn_msis*n_msis    
    visc_mu_1=3.563e-07*tn_msis**(0.71);
    visc_mu_2=1.3*pn_msis/nu_msis;
    lambda_msis=sn_msis**2./nu_msis                                                       #Conductividade termica
    
    rho2=d1_2(nz,rho_msis)
    tn2=d1_2(nz,tn_msis)
    rg2=d1_2(nz,rg_msis)
    nu_nn2=d1_2(nz,nu_msis)
    lambda_c2=d1_2(nz,lambda_msis)
    
    rho_amb=d1_3(nx,rho2)
    tn_amb=d1_3(nx,tn2)
    r_g=d1_3(nx,rg2)
    nu_nn=d1_3(nx,nu_nn2)
    lambda_c=d1_3(nx,lambda_c2)
    return
         
#%%
def dissip_coef():
#=============================================================================#
#d((mu/rho)div.div (W)/dt
#==============================================================================
    
    visc_mu=1.3*pn_amb/nu_nn;visc_ki=visc_mu/rho_amb                            #Viscocidade dinamica e kinamatica
    
    gr_flux=wx_o*gradient(rho_amb)[0]/dx2
    w_visc_x=(1.e+00/dx2**2.)*gr_flux*visc_mu/rho_amb**2.      
    gr_flux=wz_o*gradient(rho_amb)[1]/dz2
    w_visc_z=(1.e+00/dz2**2.)*gr_flux*visc_mu/rho_amb**2.
    gr_flux=wy_o*gradient(rho_amb)[2]/dy2
    w_visc_y=(1.e+00/dy2**2.)*gr_flux*visc_mu/rho_amb**2.
    
    w_visc_1=abs(w_visc_x)+abs(w_visc_z)+abs(w_visc_y)        #DUCTING EQUILIBRIUM                                      #viscocidade atraves d rho/dt
    w_visc_2=(1./dx2**2.+1./dz2**2.+1./dy2**2.)*visc_ki/dt #VISCOUS DAMPING                                 #viscocidade atraves d w/dt
    
    w_visc=exp(-0.5*dt**2.*(abs(w_visc_1)+abs(w_visc_2)))
    
#==============Saturação não linear===========================================#
#-d(W.div W)/dt
#=============================================================================#
    fac_nl=abs(wx_o).max()/(dx2*dt)
    w_nl=exp(-dt**2.*fac_nl)
    wx_damp=w_nl*w_visc    

    fac_nl=abs(wz_o).max()/(dz2*dt)
    w_nl=exp(-dt**2.*fac_nl)
    wz_damp=w_nl*w_visc                                                        #Damping Amplitude em x 
    
    fac_nl=(wy_o**2.).max()/(dy2*dt)
    w_nl=exp(-1*dt**2.*fac_nl)
    wy_damp=w_nl*w_visc                                                        #Damping amplitude em y 

    return (wx_damp,wz_damp,wy_damp)


def AGW_terms(i_axis,delta,pn,rho,div_w,div_flux,w_gr_pn):
#=======subroutina da primeira 4 termo da equação da ondas====================#
#w_1=(1/rho)grad(1.4 pn div.W)
#w_2=-((grad pn)/rho)div.(rho W)
#w_3=+(1/rho)grad(W.div)p
#=============================================================================#
    
    rho_m=data_antes(2,ny,rho)
    #rho_m=roll(rho,-2,axis=2)
    flx=1.4*pn*div_w
    grad=gradient(flx);grad_flux=grad[i_axis]/delta
    w_1=grad_flux/rho
    
    grad=gradient(pn);grad_pn=(grad[i_axis])/(2.*delta)    #IMPORTANT OF USING ABS
    
    w_2=-(grad_pn*div_flux)/rho_m**2.;      
    
    grad=gradient(w_gr_pn); 
    w_3=(1./rho)*(grad[i_axis])/(2.*delta)
    
    return (w_1,w_2,w_3)    

#%%

def AGW(wy_fonte):
    global wx,wy,wz
    
    wx=wx_o;wy=wy_o;wz=wz_o
    rho=rho_o;tn=tn_o;pn=pn_o
    
    for i_sor in range (11):
    
#%%======Equação das ondas AGWs=================================================#
#d2w/dt2=(1/rho)grad(1.4 pn div.W)-((grad pn)/rho)div.(rho W)
#+(1/rho)grad(W.div)p
#+d((mu/rho)div.div (W)/dt-d(W.div W)/dt
#
#w(t+dt,x,y)=w_damp*[2*w(t,x,y)-w(t-dt,x,y)+dt**2*(w_1+w_2+w_3)]
      
        pn=r_g*rho*tn;c_s=sqrt(1.4*pn/rho);  
    
        #===#
        
        wx_0=2.*wx_o-wx_m;
        wy_0=2.*wy_o-wy_m;
        wz_0=2.*wz_o-wz_m;
        
        #===#
        
        div_w=div_f(wx_o,wz_o,wy_o)       
        div_flux=div_f(rho_o*wx_o,rho_o*wz_o,rho_o*wy_o)
        
        gr_pn=gradient(pn_o)
        w_gr_pn=wx_o*gr_pn[0]/dx2+wy_o*gr_pn[1]/dz2+wy_o*gr_pn[2]/dy2
        
        #===#

        f_xo=AGW_terms(0,dx2,pn_o,rho_o,div_w,div_flux,w_gr_pn);        
        f_zo=AGW_terms(1,dz2,pn_o,rho_o,div_w,div_flux,w_gr_pn);
        f_yo=AGW_terms(2,dy2,pn_o,rho_o,div_w,div_flux,w_gr_pn);
        
        #===#
        
        div_w=div_f(wx,wz,wy)       
        div_flux=div_f(rho*wx,rho*wz,rho*wy)
        
        gr_pn=gradient(pn)
        w_gr_pn=wx*gr_pn[0]/dx2+wy*gr_pn[1]/dz2+wy*gr_pn[2]/dy2
        
        #===#

        f_xg=AGW_terms(0,dx2,pn,rho,div_w,div_flux,w_gr_pn);
        f_zg=AGW_terms(1,dz2,pn,rho,div_w,div_flux,w_gr_pn);
        f_yg=AGW_terms(2,dy2,pn,rho,div_w,div_flux,w_gr_pn);        
        
        f_x=f_xo+f_xg
        f_z=f_zo+f_zg
        f_y=f_yo+f_yg
        
        #===#
        
        f=dissip_coef()
        wx_damp=f[0];wz_damp=f[1];wy_damp=f[2]
        
        wx=sqrt(wx_damp)*(wx_0+dt**2.*(f_x[0]+f_x[1]+f_x[2])/2.)
        wz=sqrt(wz_damp)*(wz_0+dt**2.*(f_z[0]+f_z[1]+f_z[2])/2.)
        wy=wy_damp*(wy_0+dt**2.*(f_y[0]+f_y[1]+f_y[2])/2.-0*wy_fonte);
        
        wy[:,:,0]=wy_fonte[:,:,0]#
        #wy[:,:,0]=wy_o[:,:,0]                                                       #FORCING BOUNDARY CONDITION
 
#%%=======RESOLUÇÃO DA EQUAÇÃO DA DENSIDADE=====================================#
#rho(t+dt,x,y)=rho(t,x,y)-dt*c_diff*(div.flux(t+dt,x,y)+div.flux(t,x,y))/2.
        
        div_flux_o=div_f(rho_o*wx_o,rho_o*wz_o,rho_o*wy_o)
        div_flux=div_f(rho*wx,rho*wz,rho_o*wy)
        
        rho=rho_o-1*dt*(div_flux+div_flux_o)/2.
        
#%%=======RESOLUÇÃO DA EQUAÇÃO DA ENERGIA=====================================#
#tn(t+dt,x,y)=tn(t,x,y)-dt*c_diff*((div.flux(t+dt,x,y)+div.flux(t,x,y))/2.+(1.4-1)*tn*div.w)
        
        div_flux_o=div_f(tn_o*wx_o,tn_o*wz_o,tn_o*wy_o)        
        div_flux=div_f(tn*wx,tn*wz,tn*wy)        
        div_w=div_f(wx,wz,wy)
        
        tn=tn_o-1*dt*((div_flux+div_flux_o)/2.\
                        +(1.4-1.)*tn_o*div_w)
        
    return (wx,wz,wy,rho,tn,pn)

#%%
def ambiente_iono(iy0,y2):
    
    global no_y,n_o,nu_in,nu_0,gyro_i,b_o
    y2=y2[:,:,iy0:]
    np=1.e+12;yp=275.;r=y2/yp
    a=2;b=-5.
    n_o=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))                                    #Perfile (em altitude (y)) 
                                                                                #de densidade eletronica (/m³)
    sc_h=30.
    nu_in=1.e+03*exp(-(y2-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                #da colisão (nu_in)..
    b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
    q_c=1.6e-19;m_i=1.67e-27;z_i=16.
    gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions

    gyro_e=-gyro_i*1837.
    
    return 
#==============================================================================
def den_iono(n_o,vx,vy,vz,vnx,vny,vnz):
    fx_o=vx*gradient(n_o)[0]/2.        
    fz_o=vz*gradient(n_o)[1]/2.
    fy_o=vy*gradient(n_o)[2]/2.
    
    n=n_o
    for iter in range (11):                                                     #ITERATION LOOP PARA GAUSS-SEIDEL CONVERGENCE
        fx_g=vx*gradient(n)[0]/2.
        fz_g=vz*gradient(n)[1]/2.
        fy_g=vy*gradient(n)[2]/2.
        fx=(fx_o+fx_g)/2.                                                       #SEMI IMPLICIT CRANK-NICOLSON TIME INTEGRATION
        fy=(fy_o+fy_g)/2.
        fz=(fz_o+fz_g)/2.
        n=n_o-(fx/vnx+fy/vny+fz/vnz)                                                   #EQUAÇÃO NUMERICA DA DENSIDADE
#        n[:,0]=n[:,1];n[:,-1]=n[:,-2];                                          #condições contorno na ALTITUDE 
    return n
#==============================================================================
def vel(iy0,b_o,nu,gyro,wx,wy,wz):
    global mu_p                                                                 
    # |vx| | mu_p mu_h | |Ex|
    # |  |=|           | |  |
    # |vy| |-mu_h mu_p | |Ey|
    
    kappa=gyro/nu
    mu_p=kappa/(b_o*(1.+kappa**2.))                                             #PEDERSON MOBILITY
    mu_h=kappa**2./(b_o*(1.+kappa**2.))                                         #HALL MOBILITY
    mag_m=8.e+15         #Tm^3
    r_ea=6.371e+03
    
    theta=arctan2(z2[:,:,iy0:],y2[:,:,iy0:]+r_ea)
    delta=sqrt(1.+3.*(sin(theta))**2.)
    r3m=(r_ea+y2[:,:,iy0:])*1.e+03
    by=-2*mag_m*sin(theta)/r3m**3.;bz=mag_m*cos(theta)/r3m**3.
    bx=0
    b=mag_m*delta/r3m**3.
    ai=arccos(bz/b);
    
    ey=wz*bx-wx*bz
    ex=wy*bz-wz*by
    ez=wx*by-wy*bx
    
    i_o=1;
    mu_11=mu_p;mu_12=mu_11*kappa*cos(ai);mu_13=mu_11*kappa*sin(ai)
    mu_21=-mu_12;mu_22=mu_11*(1.+i_o*kappa**2.*sin(ai)**2.);
    mu_23=i_o*mu_11*kappa**2.*cos(ai)*sin(ai)-mu_11*kappa**2.*cos(ai)*sin(ai)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+i_o*kappa**2.*cos(ai)**2.)
    
    vx=mu_11*ex+mu_12*ey+mu_13*ez
    vy=mu_21*ex+mu_22*ey+mu_23*ez
    vz=mu_31*ex+mu_32*ey+mu_33*ez
    return (vx,vy,vz)
    
#%%=============================================================================#
#Plano (X-Y) de simulação representa a plano  em que 
#(+X,+Y) representam oeste OU norte e vertical para cima (altitude) 
#respectivamente.

dy_km=15;dx_km=2.*dy_km;dz_km=dx_km                                                            #A resoluções espaciais no kilometros
y_sc=[]
y_nw=0
while y_nw <420:
    print (y_nw)
    y_sc.append(y_nw)
    dy_nw=15
    if y_nw<=100:dy_nw=5
    # if y_nw>30 and y_nw<100:dy_nw=10
    y_nw=y_nw+dy_nw
y_sc=array(y_sc)
dy_km=gradient(y_sc)
print (dy_km)

y=y_sc#arange(0,400+dy_km,dy_km);
ny=len(y)                                          #A faixa de altitude
x=arange(-500,500+dx_km,dx_km);nx=len(x)
z=arange(-500,500+dz_km,dz_km);nz=len(z)

dy=dy_km*1.e+03;dx=0*x+dx_km*1.e+03;dz=0*z+dz_km*1.e+03                                                 #As resoluções espaciais no metros
[dz2,dx2,dy2]=meshgrid(dz,dx,dy)
[z2,x2,y2]=meshgrid(z,x,y)


iy0=abs(y-150).argmin()
f=ambiente_atmos(y2)
f=ambiente_iono(iy0,y2)

#%%==================================MAIN=======================================#
#(wx,wy) são amplitudes da AGWs na direções (x,y) 
#(rho_o,tn_o,pn_o) são densidade, temeratura e pressão atmosferica
#(wx_m,wy_m)=(wx(t-dt,x,y),wy(t-dt,x,y))
#(wx_o,wy_o)=(wx(t,x,y),wy(t,x,y)
#(rho_o,tn_o,pn_o)=(rho(t-dt,x,y),tn(t-dt,x,y),pn(t-dt,x,y))

global wx_m,wy_m,wx_o,wy_o,wz_m,wz_o                                                      
global rho_o,tn_o,pn_o,rho_amb,tn_amb,pn_amb
global t


wx_m=zeros((nx,nz,ny));wz_m=zeros((nx,nz,ny));wy_m=zeros((nx,nz,ny));
wx_o=0*wx_m;wy_o=0*wy_m;wz_o=0*wz_m

#============================INTIALIZATION====================================#
#rho_amb=rho_amb[0,0]*exp(-(y2)/20.)
pn_amb=r_g*rho_amb*tn_amb
rho_o=rho_amb;tn_o=tn_amb;pn_o=pn_amb

#=============================================================================#

data_sism= load('DATA_ILLAPEL/Dados_sismico.npy')
t_sism=3600*data_sism[0,290000:-4000:1200];#segundos
v_sism=data_sism[1,290000:-4000:1200]#m/s
v_sism=(v_sism/2.5+roll(v_sism,40))/1.
#v_sism[t_sism/3600.<22.91]=0
#v_sism[t_sism/3600.>22.95]=0
t_sism=t_sism-t_sism[0]
it=0;t=0;
t=t_sism[0];
dt=t_sism[1]-t_sism[0]

# sig_t=5*dt;t0=t_sism[0]+30*dt;omega=2.*pi/(2.*60.)
# v_sism=v_sism.max()*skew(t_sism, t0, sig_t, 10)
# v_sism=fac_red*v_sism*sin(omega*t_sism)
# v_sism=v_sism-v_sism[0]
vnx=dx2/dt;vny=dy2/dt;vnz=dz2/dt                                                         #VELOCIDADE NUMERICIAS

        
t_f=t_sism[-1]


sigma_x=2.*dx2*1.e-03;x_o=x2.mean()

time=[]
tn3=[];pn3=[];wx3=[];wy3=[];n3=[];vw3=[]
while t <=t_f/1.:    
    # f_x0=skew(x2,0,sigma_x/1.,0)[:,:,0]
    # f_z0=skew(z2,0,sigma_x/1.,0)[:,:,0]
    
    # f_x1=skew(x2,0+20*dx2*1.e-03,sigma_x/1.,0)[:,:,0]
    # f_z1=skew(z2,0+20*dz2*1.e-03,sigma_x/1.,0)[:,:,0]
    
    # f_lon=f_x0*f_z0#+f_x1*f_z1
    #f_lon=f_lon-f_lon[0]
    
    # wy_o[:,:,0]=v_sism[it]*f_lon
    # wy_o[:,:,0]=wy_o[:,:,0]-wy_o[0,0,0] 
    
    v_s=-0*500*1.e-03
    f_x0=skew(x2,0+it*dt*v_s,sigma_x/1.,0)
    f_z0=skew(z2,0+it*dt*v_s,sigma_x/1.,0)
    
    f_x1=skew(x2,0+10*dx2*1.e-03,sigma_x/1.,0)
    f_z1=skew(z2,0+10*dz2*1.e-03,sigma_x/1.,0)
    
    f_lon=f_x0*f_z0#+f_x1*f_z1
    f_lon=f_lon-f_lon[0]
    f_v=skew(y2,y2[0,0,3],sigma_x/10.,0);
    f_v=f_v-f_v[0,0,0]
    
    figure(41,(12,12))
    plot(f_v[0,0,:],y2[0,0,:])
    wy_fonte=v_sism[it]*f_lon#*f_v
    wy_fonte[y2>0]=0
    
    f=AGW(wy_fonte)
    wx=f[0];wz=f[1];wy=f[2];rho=f[3];tn=f[4];pn=f[5]

    f=vel(iy0,b_o,nu_in,gyro_i,wx[:,:,iy0:],wy[:,:,iy0:],wz[:,:,iy0:])
    vx=f[0];vy=f[1];vz=f[2]
    #vx=wx[:,:,iy0:];vy=wy[:,:,iy0:];vz=wz[:,:,iy0:]
    n=den_iono(n_o,vx,vy,vz,vnx[:,:,iy0:],vny[:,:,iy0:],vnz[:,:,iy0:])
       
#===========================ATUALIZAÇÃO EM TEMPO===================#
    # dn=abs(n-n_o)/n_o
    n_o=n
    wx_m=wx_o;wx_o=wx;
    wz_m=wz_o;wz_o=wz; 
    wy_m=wy_o;wy_o=wy;
    rho_o=rho;tn_o=tn;pn_o=pn
    
    i_xo=int((nx+1)/2);i_zo=int((nz+1)/2)
    time.append(t/60.)
    tn3.append(tn);pn3.append(pn);
    wx3.append(wx)
    wy3.append(wy)
    n3.append(n)
    # vw3.append(wy[i_xo,:]/v_sism.max())

#=========================================================================#
    
    print ('======================================================')
    print ("TIME=", timedelta(seconds=t),'TIME_STEP=',dt)
    print ('FORCING AMPLITUDE=',round(wy[:,:,0].max(),5))
    print ('AGWs Amplitudes=',round(wx[i_xo,i_zo,:].max(),2),round(wy[i_xo,i_zo,:].max(),2)) 
    #print ('Amplification ratio',abs(array(vw3).max()))
    #print ('TIDs Amplitudes %=',round(100*dn.max(),2)) 
    
    if wx[i_xo,i_zo,:].max()>5000. or wy[i_xo,i_zo,:].max()>5000.:
        break
    if rho.any()<0 or tn.any()<0:
        break
    
#    cs=sqrt(1.4*pn/rho);dt=int(0.5*dy/cs.max())
    t=t+dt
    it=it+1
    
    if remainder(it,5)==0:
        fig=figure(1,figsize=(12,12),facecolor='w',edgecolor='k') 
        subplot(121)
        semilogx(rho_amb[0,0,:],y2[0,0,:],'r')
        xlabel('Densidade, $kg m^{-3}$');ylabel('Altitude, km')
        subplot(122)
        plot(tn_amb[0,0,:],y2[0,0,:],'b')
        xlabel('Temperatura $^o$K');ylabel('Altitude, km')
        draw()
        
        fig=figure(2,figsize=(12,12),facecolor='w',edgecolor='k') 
        ext=[x2[0,0,0],x2[-1,0,0],y2[0,0,0],y2[0,0,-1]]
        subplot(223)
        vm=wx.max()
        im=imshow(wx[:,i_zo,:].T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
        im.set_interpolation('bilinear')
        axis('tight');
        title('Horizontal Amplitude $(W_x)$ of AGWs, m/s')
        subplot(224)
        vm=wy.max()
        im=imshow(wy[:,i_zo,:].T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
        im.set_interpolation('bilinear')
        axis('tight');
        title('Vertical Amplitude $(W_y)$ of AGWs, m/s')
        
        ext=[time[0]/60.,t/60.,y2[0,0,0],y2[0,0,-1]]
        subplot(221)
        plot(array(time),5.e+04*abs(array(wy3))[:,i_xo,i_zo,0],'-')
        data=array(wx3)[:,i_xo,i_zo,:]
        vm=data.max()
        im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
        im.set_interpolation('bilinear')
        axis('tight');
        title('Amplitude horizontal $(W_x)$, m/s as x=0 km')
        xlabel('Time, minutos');ylabel('Altitude, km')
        subplot(222)
        plot(array(time),5.e+04*abs(array(wy3))[:,i_xo,i_zo,0],'-')
        data=array(wy3)[:,i_xo,i_zo,:]
        vm=data.max()
        im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
        im.set_interpolation('bilinear')
        axis('tight');
        title('Amplitude vertical $(W_y)$, m/s as x=0 km')
        xlabel('Time, minutos');ylabel('Altitude, km')

        draw()
        
        fig=figure(4,figsize=(12,12),facecolor='w',edgecolor='k') 
        ext=[time[0]/60.,t/60.,x2[0,0,0],x2[-1,0,0]]
        subplot(211)
        plot(array(time),5.e+05*array(wy3)[:,i_xo,i_zo,0],'g-')
        data=(gradient(array(pn3)[:,:,i_zo,:])[0]/array(pn3)[:,:,i_zo,:]).sum(2)
        vm=data.max()/2.
        im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
        im.set_interpolation('bilinear')
        axis('tight');
        title('AID as x=0 km')
        xlabel('Time, minutos');ylabel('Epicentral Distance, km')
        
        subplot(212)
        data=gradient(array(n3))[0][:,:,i_zo,5:10].sum(2)
        plot(array(time),5.e+05*array(wy3)[:,i_xo,i_zo,0],'g-')
        vm=data.max()/2.
        im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
        im.set_interpolation('bilinear')
        axis('tight');
        title('TID as x=0 km')
        xlabel('Time, minutos');ylabel('Epicentral Distance, km')
        draw()
        
        fig=figure(5,figsize=(12,12),facecolor='w',edgecolor='k') 
        ext=[x2[0,0,0],x2[-1,0,0],z2[0,0,0],z2[0,-1,0]]
        data=gradient(array(n3))[0][it-1,:,:,:].sum(2)
        tec=dt*300*1.e+03*data/1.e+16
        vm=0.5
        im=imshow(tec.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
        im.set_interpolation('bilinear')
        axis('tight');
        
        pause(0.01)
        
show()


save('time_num.npy',array(time))
#save('lon.npy',x)
#save('dtn1.npy',array(dtn3))
#save('wx1.npy',array(wx3))
save('wy_num.npy',array(wy3))
save('n3_num.npy',array(n3))
#save('vw1.npy',array(vw3))