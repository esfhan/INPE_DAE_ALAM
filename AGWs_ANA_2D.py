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
#=============================================================================#
#sigma_t é espressura de pacote Gaussiano de tempo
#t_o é tempo em que forçante atinge amplitude maior e deve ser mair do 2*sigma_t
#t_f é tempo final de simulação e deve ser mais de 2*t_o
#=============================================================================#

#%%
#==================================MAIN====================================== 
#(wx,wy) são amplitudes da AGWs na direções (x,y) ou seja Longitudinal e transverse 
#(rho_o,tn_o,pn_o) são densidade, temeratura e pressão atmosferica
#(wx_m,wy_m)=(wx(t-dt,x,y),wy(t-dt,x,y))
#(wx_o,wy_o)=(wx(t,x,y),wy(t,x,y)
#(rho_o,tn_o,pn_o)=(rho(t-dt,x,y),tn(t-dt,x,y),pn(t-dt,x,y))

#%%
#==============================================================================
#=Plano (X-Y) de simulação representa a plano  em que 
#(+X,+Y) representam oeste OU norte e vertical para cima (altitude) 
#respectivamente.
   
from pylab import *
from numpy import *
from nrlmsise_2000 import *
from scipy import *
from scipy.ndimage import *
from scipy.special import erf
from scipy.integrate import trapz
from matplotlib.signal_alam import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rc("mathtext",fontset="cm")        #computer modern font 
matplotlib.rc("font",family="serif",size=12)


def d1_2(n,data):
    return repeat(data[newaxis,:],n,axis=0)

def mask_b():
    mask=1+zeros((nx,ny))
    rad_m=nx/4
    for j in range (0,nx):
        if j<rad_m or j>=int(nx/2)+rad_m:
            mask[j,:]=exp(-(0.5*j/rad_m)**2.)
#        if abs(k-iq_m)<0.8*iq_m and abs(j-iphi_m)<0.8*iphi_m:
#            mask_bound[:,j,k]=mask_q+mask_phi 
    return mask
#=============================================================================#
def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)

#=============================================================================#
def div_f(f0,f1):
    return gradient(f0)[0]/dx_m+gradient(f1)[1]/dy_m 

#%%
def sum_gr(ndim,ndata,data):
    data_n=0*data
    
    if ndim==0:
        for j in range (ndata):
            if j==0:
                data_n[j,:]=(data[j+1,:]+data[j,:])/2.
            elif j==ndata-1:
                data_n[j,:]=(data[j-1,:]+data[j,:])/2.
            else:
                data_n[j,:]=data[j+1,:]+data[j-1,:]
    
    if ndim==1:
        for j in range (ndata):
            if j==0:
                data_n[:,j]=(data[:,j+1]+data[:,j])/2.
            elif j==ndata-1:
                data_n[:,j]=(data[:,j-1]+data[:,j])/2.
            else:
                data_n[:,j]=data[:,j+1]+data[:,j-1]
    return data_n


#%%
def data_antes(dim,ndim,data):
    data_n=0*data
    if dim==1:
        data_n=0*data
        data_n[1:-1]=data[0:-2]
        data_n[0]=data_n[1];data_n[-1]=data_n[-2];
    else:
        if ndim==1:
            data_n[:,1:-1]=data[:,0:-2]
            data_n[:,0]=data_n[:,1];data_n[:,-1]=data_n[:,-2];
    return data_n

#%%
def data_proximo(dim,ndim,data):
    data_n=0*data
    if dim==1:
        data_n=0*data
        data_n[1:-1]=data[2:]
        data_n[0]=data_n[1];data_n[-1]=data_n[-2];
    else:
        if ndim==1:
            data_n[:,1:-1]=data[:,2:]
            data_n[:,0]=data_n[:,1];data_n[:,-1]=data_n[:,-2];
    return data_n

#%%
def ambiente_atmos(iw,x2,y2):
    global rho_amb,tn_amb,r_g,nu_nn,lambda_c
    global pn, sn
    if iw==0:
        lat_ep=35.7;lon_ep=-117.6
        year,month,dom=2019,7,4
                
        d0 = datetime.date(year,1,1)
        d1 = datetime.date(year, month, dom)
        delta = d1 - d0
        doy=delta.days 
        ut=17.55;lt=ut+lon_ep/15.
        
        f107A,f107,ap=150,150,4
        
        f=nrl_msis(doy,ut*3600.,lt,f107A,f107,ap,lat_ep,lon_ep,dy,y[0],ny)
        tn_msis=f[1];
        den_ox=f[2]*1.e+06;den_n=f[3]*1.e+06;den_o2=f[4]*1.e+06;den_n2=f[5]*1.e+06;
        n_msis=den_ox+den_n+den_o2+den_n2;
        rho_msis=f[6]*1.e+03
        mean_mass=rho_msis/n_msis
        
        b_c=1.38e-23; 
        rg_msis=b_c/mean_mass
        pn_msis=rg_msis*rho_msis*tn_msis
        sn_msis=sqrt(1.4*pn_msis/rho_msis)
        
        nu_msis=pi*(7*5.6e-11)**2.*sn_msis*n_msis    
        visc_mu_1=3.563e-07*tn_msis**(0.71);
        visc_mu_2=1.3*pn_msis/nu_msis;
        lambda_msis=sn_msis**2./nu_msis                                                       #Conductividade termica
        
        rho_amb=d1_2(nx,rho_msis)
        tn_amb=d1_2(nx,tn_msis)
        sn=d1_2(nx,sn_msis)
        r_g=d1_2(nx,rg_msis)
        nu_nn=d1_2(nx,nu_msis)
        lambda_c=d1_2(nx,lambda_msis)
        print (tn_amb.max())
    if iw==1:
        a=0.25e+25*32*1.6e-27*1.e-08#*1.e+06; 
        c=-4.5;d=-19.5;c=-3.5
        yr=130.;r_y=y2/yr-1
        rho_amb=a*(exp(c*r_y)+exp(d*r_y))                                             #Densidade de massa (kg/m³)
        
        a1=310;b1=200;c1=-1.;d1=0.5
        a2=220;b2=250;c2=-0.9;d2=6.
        a3=220;b3=135;c3=-0.2;d3=7.
        a4=120;b4=250;c4=-1.5;d4=18.
        a5=220;b5=220;c5=-0.25;d5=5.
        y_1=10.;y_2=65.;y_3=130.;y_4=100.;y_5=230.;
        
        r_y1=y2/y_1-1;r_y2=y2/y_2-1;r_y3=y2/y_3-1
        r_y4=y2/y_4-1;r_y5=y2/y_5-1
        
        first=a1-b1/(exp(c1*r_y1)+exp(d1*r_y1))
        second=-a2+b2/(exp(c2*r_y2)+exp(d2*r_y2))
        third= a3-b3/(exp(c3*r_y3)+exp(d3*r_y3))
        fourth=-a4+b4/(exp(c4*r_y4)+exp(d4*r_y4))
        fifth=a5-b5/(exp(c5*r_y5)+exp(d5*r_y5))
        sixth=-720+1.25*(first+second+0*fourth)+8.5*third+2.*fifth 
        tn_amb=0.5*sixth                                                              #Temperatura atmosferica (K)
    
        r_g=150.*(1.+sqrt(y2*1.e-03+5.)/5.);                                        #constante Boltzman/massa
        
        b_c=1.38e-23;                                                               #Constante Boltzmann
        mean_mass=b_c/r_g                                                           #massa atmosferica
        nn=rho_amb/mean_mass                                                          #Densidade numerica
        pn=r_g*rho_amb*tn_amb                                                           #Pressão atmosferica
        sn=sqrt(1.4*pn/rho_amb)                                                       #velocidade de som
        
        nu_nn=pi*(7*5.6e-11)**2.*sn*nn                                              #frequencia da colisão
        lambda_c=sn**2./nu_nn                                                       #Conductividade termica
        
    return
         
#%%
def dissip_coef():
#=============================================================================#
#d((mu/rho)div.div (W)/dt
#==============================================================================
    
    visc_mu=1.3*pn_amb/nu_nn;visc_ki=visc_mu/rho_amb                            #Viscocidade dinamica e kinamatica
    
    gr_flux=gradient(rho_amb*wx_o)[0]/dx_m
    w_visc_x=(1./dx_m**2.)*gr_flux*visc_mu/rho_amb**2.      
    gr_flux=gradient(rho_amb*wy_o)[1]/dy_m
    w_visc_y=(1./dy_m**2.)*gr_flux*visc_mu/rho_amb**2.
    
    w_visc_rho=abs(w_visc_x)+abs(w_visc_y)                                      #viscocidade atraves d rho/dt
    w_visc_w=(1./dx_m**2.+1./dy_m**2.)*visc_ki/dt                                 #viscocidade atraves d w/dt
      
    w_visc=exp(-0.5*dt**2.*(abs(w_visc_rho)+abs(w_visc_w)))
    
#==============Saturação não linear===========================================#
#-d(W.div W)/dt
#=============================================================================#
    fac_nl=abs(wx_o).max()/(dx_m*dt)
    w_nl=exp(-0.05*dt**2.*fac_nl)
    wx_damp=w_nl*w_visc                                                        #Damping Amplitude em x 
    
    fac_nl=abs(wy_o).max()**1./(dy_m*dt)
    w_nl=exp(-0.05*dt**2.*fac_nl)
    wy_damp=w_nl*w_visc                                                        #Damping amplitude em y 

    return (wx_damp,wy_damp)


def AGW_terms(i_axis,delta,pn,rho,div_w,div_flux,w_gr_pn):
#=======subroutina da primeira 4 termo da equação da ondas====================#
#w_1=(1/rho)grad(1.4 pn div.W)
#w_2=-((grad pn)/rho)div.(rho W)
#w_3=+(1/rho)grad(W.div)p
#=============================================================================#
    
    rho_m=data_antes(2,1,rho)
    flx=1.4*pn*div_w
    grad=gradient(flx);grad_flux=grad[i_axis]/delta
    w_1=grad_flux/rho
    
    grad=gradient(pn);grad_pn=abs(grad[i_axis])/delta    #IMPORTANT OF USING ABS
    
    w_2=-fac_amp*(grad_pn*div_flux)/rho_m**2.;        
    
    grad=gradient(w_gr_pn); 
    w_3=(fac_amp/rho_m)*(grad[i_axis])/delta
    
    return (w_1,w_2,w_3)    

#%%

def AGW():
    global wx,wy
    
    wx=wx_o;wy=wy_o;
#    rho_o=rho_amb;tn_o=tn_amb;pn_o=pn_amb
    rho=rho_o;tn=tn_o;pn=pn_o
    
    for i_sor in range (21):
    
#======Equação das ondas AGWs=================================================#
#d2w/dt2=(1/rho)grad(1.4 pn div.W)-((grad pn)/rho)div.(rho W)
#+(1/rho)grad(W.div)p
#+d((mu/rho)div.div (W)/dt-d(W.div W)/dt
#
#w(t+dt,x,y)=w_damp*[2*w(t,x,y)-w(t-dt,x,y)+dt**2*(w_1+w_2+w_3)]
#=============================================================================#        
        pn=r_g*rho*tn;c_s=sqrt(1.4*pn/rho);  
    
        #===#
        
        wx_0=2.*wx_o-wx_m;
        wy_0=2.*wy_o-wy_m;
        
        #===#
        
        div_w=div_f(wx_o,wy_o)       
        div_flux=div_f(rho_o*wx_o,rho_o*wy_o)
        
        gr_pn=gradient(pn_o)
        w_gr_pn=wx_o*gr_pn[0]/dx_m+wy_o*gr_pn[1]/dy_m
        
        #===#

        f_xo=AGW_terms(0,dx_m,pn_o,rho_o,div_w,div_flux,w_gr_pn);
        f_yo=AGW_terms(1,dy_m,pn_o,rho_o,div_w,div_flux,w_gr_pn);        
        
        #===#
        
        div_w=div_f(wx,wy)       
        div_flux=div_f(rho*wx,rho*wy)
        
        gr_pn=gradient(pn)
        w_gr_pn=wx*gr_pn[0]/dx_m+wy*gr_pn[1]/dy_m
        
        #===#

        f_xg=AGW_terms(0,dx_m,pn,rho,div_w,div_flux,w_gr_pn);
        f_yg=AGW_terms(1,dy_m,pn,rho,div_w,div_flux,w_gr_pn);        
        
        f_x=f_xo+f_xg
        f_y=f_yo+f_yg
        
        #===#
        
        f=dissip_coef()
        wx_damp=f[0];wy_damp=f[1]
        
        wx=wx_damp*(wx_0+dt**2.*(f_x[0]+f_x[1]+f_x[2])/2.); #
        wy=wy_damp*(wy_0+dt**2.*(f_y[0]+f_y[1]+f_y[2])/2.);
       
        wy[:,0]=wy_o[:,0]                                                       #FORCING BOUNDARY CONDITION
#        wx[:,0]=wx_o[:,0]
#        wx[nx-1,:]=-wx[0,:]
#        wy[nx-1,:]=wy[0,:]
#=======RESOLUÇÃO DA EQUAÇÃO DA DENSIDADE=====================================#
#rho(t+dt,x,y)=rho(t,x,y)-dt*c_diff*(div.flux(t+dt,x,y)+div.flux(t,x,y))/2.
#=============================================================================#
        f_m=fac_amp.mean()
        div_flux_o=div_f(rho_o*wx_o/f_m,rho_o*wy_o/f_m)
        div_flux=div_f(rho*wx/f_m,rho*wy/f_m)
        
        d2_rho=gradient(gradient(rho_o)[0])[0]/dx_m**2.\
                +gradient(gradient(rho_o)[1])[1]/dy_m**2.
        
        c_diff=exp(-0.*abs(lambda_c*d2_rho*t/rho_o))
        
        rho=rho_o-0.5*c_diff*dt*(div_flux+div_flux_o)/2.
        
#=======RESOLUÇÃO DA EQUAÇÃO DA ENERGIA=====================================#
#tn(t+dt,x,y)=tn(t,x,y)-dt*c_diff*((div.flux(t+dt,x,y)+div.flux(t,x,y))/2.+(1.4-1)*tn*div.w)
#=============================================================================#
        
        div_flux_o=div_f(tn_o*wx_o/f_m,tn_o*wy_o/f_m)        
        div_flux=div_f(tn*wx/f_m,tn*wy/f_m)        
        div_w=div_f(wx/f_m,wy/f_m)
        
        d2_tn=gradient(gradient(tn_o)[0])[0]/dx_m**2.\
                +gradient(gradient(tn_o)[1])[1]/dy_m**2.
        
        c_diff=exp(-0.*abs(lambda_c*d2_tn*t/tn_o))
        
        tn=tn_o-0.5*c_diff*dt*((div_flux+div_flux_o)/2.\
                        +(1.4-1.)*tn_o*div_w\
                        -0*lambda_c*d2_tn)
        
    return (wx,wy,rho,tn,pn)

#%%
def rho_tn(rho_o,tn_o):
    div_flux=div_f(rho_o*wx_ana,rho_o*wy_ana)
    rho=rho_o-0.0*dt*div_flux
    
    div_flux=div_f(tn_o*wx_ana,tn_o*wy_ana)        
    div_w=div_f(wx_ana,wy_ana)
    
    tn=tn_o-0.*dt*(div_flux+(1.-1.)*tn_o*div_w)
    
    return (rho,tn)

#%%ANALTICA
def AGW_ana():
    gma=1.4
    pn=r_g*rho_o*tn_o;c_s=sqrt(gma*pn/rho_o); 
    gr_pn=gradient(pn)[1];dy_m2=2.*dy_m
    zeta=(1./rho_o)*gr_pn/dy_m2
    k0=zeta/c_s**2.
    k0_antes=data_antes(2,1,k0)
    k0_proximo=data_proximo(2,1,k0)
#    mu=0.5*(k0)*y2_m#(k0)*y2*1.e+03#(k0_proximo+k0_antes)*dy_m#0.1*(k0-k0[0,0])*y2*1.e+03#trapz(k0,x=y2,dx=dy_m,axis=1)
    mu=0.5*(k0_proximo+k0_antes)*dy_m2/2.
    mu=cumsum(mu,1);mu=9.*mu/abs(mu).max()
    i_mx=abs(abs(mu[0,:])-abs(mu).max()).argmin()
    mu[:,i_mx:]=0*mu[:,i_mx:]+mu[0,i_mx]
    omega_c2=(gma**2.*k0*c_s)**2./4.
    omega_b2=((gma-1)*k0**2-0*(k0/c_s**2.)*gradient(c_s**2.)[1]/dy_m2)*c_s**2.
    omega_h2=wk_x**2.*c_s**2.
    omega_2=omega_b2+(wk_y**2.+k0**2.)*c_s**2.+omega_h2
    omega_mais=sqrt(omega_2+sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
    omega_menos=sqrt(omega_2-sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
#    omega_mais=sqrt(omega_2)
#    omega_menos=sqrt(omega_h2*omega_b2/omega_2)
#    if omega_menos.any()==0.:
#        omega_menos=omega_mais
    visc_mu=1.3*pn_amb/nu_nn;visc_ki=visc_mu/rho_amb
    nu_col=visc_ki*(-wk_x**2-wk_y**2.+k0**2.-gradient(k0)[1]/dy_m2)
    wx_mais=(omega_mais**2.+omega_h2-omega_2)/(wk_x*wk_y*c_s**2.)
    wx_menos=(omega_menos**2.+omega_h2-omega_2)/(wk_x*wk_y*c_s**2.)
    return (mu,omega_mais,omega_menos,nu_col,wx_mais,wx_menos,omega_b2,omega_c2)

#%%ANALTICA
def AGW_ana_x():
    gma=1.4
    pn=r_g*rho_o*tn_o*(1-0.2*cos(wk_x*x2_m));c_s=sqrt(gma*pn/rho_o);  
    gr_pn=gradient(pn)[0];dx_m2=2.*dx_m
    zeta=(1./rho_o)*gr_pn/dx_m2
    k0=zeta/c_s**2.
    omega_c2=(gma**2.*k0*c_s)**2./4.
    omega_b2=((gma-1)*k0**2-0*(k0/c_s**2.)*gradient(c_s**2.)[0]/dx_m2)*c_s**2.
    omega_h2=wk_x**2.*c_s**2.
    omega_2=omega_b2+(wk_y**2.+k0**2.)*c_s**2.+omega_h2#+omega_c2
    omega_mais=sqrt(omega_2+sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
    omega_menos=sqrt(omega_2-sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
    omega_mais=sqrt(omega_2)
    omega_menos=sqrt(omega_h2*omega_b2/omega_2)
    return (omega_mais,omega_menos)
    
#%%
def ambiente_iono(x2,y2):
    
    global no_y,n_o,nu_in,nu_0,gyro_i,b_o
    y2=y2[:,15:]
    np=1.e+12;yp=275.;r=y2/yp
    a=2;b=-5.
    n_o=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))                                    #Perfile (em altitude (y)) 
                                                                                #de densidade eletronica (/m³)
    #===
    sc_h=30.
    nu_in=1.e+03*exp(-(y2-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                #da colisão (nu_in)..
#    nu_0=nu_in
#    i_500=abs(y-500.).argmin()
#    for i in range (i_500,ny):
#        nu_0[:,i]=nu_0[:,i_500]
                                                                        
    b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
    q_c=1.6e-19;m_i=1.67e-27;z_i=16.
    gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions

    gyro_e=-gyro_i*1837.
    
    return 
#==============================================================================
def den_iono(n_o,vx,vy,vnx,vny):
    fx_o=vx*gradient(n_o)[0]/2.        
    fy_o=vy*gradient(n_o)[1]/2.
    
    n=n_o
    for iter in range (11):                                                     #ITERATION LOOP PARA GAUSS-SEIDEL CONVERGENCE
        fx_g=vx*gradient(n)[0]/2.
        fy_g=vy*gradient(n)[1]/2.
        fx=(fx_o+fx_g)/2.                                                       #SEMI IMPLICIT CRANK-NICOLSON TIME INTEGRATION
        fy=(fy_o+fy_g)/2.
        n=n_o-(fx/vnx+fy/vny)                                                   #EQUAÇÃO NUMERICA DA DENSIDADE
#        n[:,0]=n[:,1];n[:,-1]=n[:,-2];                                          #condições contorno na ALTITUDE 
    return n
#==============================================================================
def vel(b_o,nu,gyro,wx,wy):
    global mu_p                                                                 
    # |vx| | mu_p mu_h | |Ex|
    # |  |=|           | |  |
    # |vy| |-mu_h mu_p | |Ey|
    
    kappa=gyro/nu
    mu_p=kappa/(b_o*(1.+kappa**2.))                                             #PEDERSON MOBILITY
    mu_h=kappa**2./(b_o*(1.+kappa**2.))                                         #HALL MOBILITY
    lat=radians(35.7)
    mag_m=8.e+15         #Tm^3
    r_ea=6.371e+06
    b1=-2.*mag_m*sin(abs(lat))/r_ea**3.;
    b3=mag_m*cos(lat)/r_ea**3.;
    b2=0
    wz=0
    ey=wz*b2-wx*b3
    ex=wy*b3-wz*b1
    ez=wx*b1-wy*b2
#    ex=wy*b_o*cos(lat);ey=-wx*b_o*cos(lat)
    vx=mu_p*ex+mu_h*ey
    vy=mu_p*ey-mu_h*ex
    return (vx,vy)

#%%============================FONTE SISMICA============================================#

def fonte(v_phase):
    global sigma_t,t0,sigma_x
    #t_sism=3600*load('t_sism.npy')[1200:-1:2];
    #v_sism=1.e-02*load('data_sism.npy')[1200:-1:2]
    ##v_sism=data_smooth(t_sism,v_sism)
    #it=0
    #t=0;t_f=t_sism[71]-t_sism[0]
    #dt=t_sism[1]-t_sism[0]#
    sigma_t=36.*dt/2.;omega_o=2.*pi/(0.5*sigma_t)                  

    t0=5*sigma_t;v0=0.025;ft0=v0*skew(t-y2_m/v_phase,t0,sigma_t,1)
    t1=4350-t0;v1=0.0004;ft1=v1*skew(t-y2_m/v_phase,t0+t1,sigma_t/1.,2)
    t2=5046-t0;v2=0.0003;ft2=v2*skew(t-y2_m/v_phase,t0+t2,sigma_t/1.,2)
    t3=5508-t0;v3=-0.00025;ft3=v3*skew(t-y2_m/v_phase,t0+t3,sigma_t/1.,2)
    t4=6040-t0;v4=-0.0005;ft4=v4*skew(t-y2_m/v_phase,t0+t4,sigma_t/1.,2)
    t5=7550-t0;v5=0.00025;ft5=v5*skew(t-y2_m/v_phase,t0+t5,sigma_t/1.,2)
    v_sism=ft0+0*1.*(ft1+ft2+ft3+ft4+ft5)
    v_sism=v_sism/10.
    sigma_x=4.*dx_m*1.e-03;x_o=x2.mean()
    f_lon=skew(x2,x_o,sigma_x/1.,0)

    wy0=(f_lon-f_lon[0,0])*v_sism*(1-0.*cos(omega_o*t))
#    wy0=1.e-03*sin(omega_0*t)*f_lon#
#    wy0=v_sism[it]*f_lon[:,:]#*cos(omega_0*t)
#    wy0=wy0-wy0[0,0]
    return wy0


#%%==================================MAIN====================================== 

global wx_m,wy_m,wx_o,wy_o                                                      
global rho_o,tn_o,pn_o,rho_amb,tn_amb,pn_amb
global fac_amp,mask
global wk_x,wk_y

#%%
dy=10;dx=dy;dt=dy;                                                            #A resoluções espaciais no kilometros
y=arange(0,400+dy,dy);ny=len(y)                                           #A faixa de altitude
x=arange(-1400,1400+dx,dx);nx=len(x)                                        #A faixa de longitude
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
dy_m=dy*1.e+03;dx_m=dx*1.e+03                                                 #As resoluções espaciais no metros

it=0;nt=900;t=0;t_f=nt*dt
vnx=dx_m/dt;vny=dy_m/dt #VELOCIDADE NUMERICIAS

#%%
wx_m=zeros((nx,ny));wy_m=zeros((nx,ny));
wx_o=0*wx_m;wy_o=0*wy_m;
time=[]
dtn3=[];wx3=[];wy3=[];wy3_ray=[];n3=[];vw3=[];data_arrival=[]
wave_all=[];data_amb=[]

#%%============================================================================

f=ambiente_atmos(0,x2,y2)
f=ambiente_iono(x2,y2)
f=fonte(1.)
data_amb.append((rho_amb[0,:],sn[0,:]))

#%%============================INTIALIZATION===================================


pn_amb=r_g*rho_amb*tn_amb
rho_o=rho_amb;tn_o=tn_amb;pn_o=pn_amb

fac_amp=sqrt(data_antes(2,1,rho_amb)/data_proximo(2,1,rho_amb))#
#fac_amp=sqrt(abs(rho_amb/rho_amb.max()))
fac_amp=exp(-fac_amp)
mask=mask_b()
                                                          
#%%SOLUCAO ANALYTICA

i_pl=1
pdf_frente=zeros((nx,ny))
pdf_tras=zeros((nx,ny))
while t <=t_f:
    wy_ray=zeros((nx,ny))
    wy_ana=zeros((nx,ny));
    for ik in range (15):
        lambda_x=10.*dx_m;
        if ik==0:
            lambda_y=(2.+2*ik)*dy_m
        else:
            lambda_y=(4*ik)*dy_m
        for ikx in range (ik,15):
            lambda_x=max(sigma_x,4*ikx*dx_m)
#        if lambda_y>lambda_x:
#            print ('Longest wavelength='+str(lambda_y/1.e+03))
#            break
            wk_x=2.*pi/lambda_x;wk_y=2.*pi/lambda_y    
            
            f=AGW_ana()
            mu=f[0];omega_mais=f[1];omega_menos=f[2];nu_col=f[3];
            wx_mais=f[4];wx_menos=f[5]
            omega_br=sqrt(f[6]);omega_ac=sqrt(f[7])
            f=AGW_ana_x()
            omega_mais_x=f[0];omega_menos_x=f[1]
            w_amp=1
            if omega_mais.max() > 1./dt :
                w_amp=0
            
            wy_alt=exp(-mu)
            n=1.
            wy_damp=exp(2*nu_col*t/(2.*n))
            
            #%%ONDAS ACUSTICAS
            omega_aw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_mais*t_f)**2.)*omega_mais
            #ONDAS ACUSTICAS
            omega_gw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_menos*t_f)**2.)*omega_menos
            for i_wv in range (2):
                if i_wv==0: 
                    omega=omega_aw
                    wx_amp=wx_mais/10.
                if i_wv==1: 
                    omega=omega_gw
                    wx_amp=wx_menos/10.
                    
                v_phase=omega/wk_y;v_phase_x=omega/wk_x
                wy0=fonte(v_phase);
                ondas_frente=cos(-omega*t+wk_y*y2_m+wk_x*x2_m)
                ondas_tras=cos(omega*t+wk_y*y2_m+wk_x*x2_m)
                wy_frente=wy0*ondas_frente*wy_alt*wy_damp
                wy_tras=wy0*ondas_tras*wy_alt*wy_damp
                wy_ondas=w_amp*(wy_frente+wy_tras)
                
                phase_frente=wk_x*(x2_m-v_phase_x*t)
                phase_tras=wk_x*(x2_m+v_phase_x*t)
                ondas_frente=cos(phase_frente)
                ondas_tras=cos(phase_tras)
                    
                pdf_frente=pdf_frente+pdf(phase_frente/(2.*pi))
                pdf_tras=pdf_tras+pdf(phase_tras/(2.*pi))
                    
                prop_x=(ondas_frente*pdf_frente
                            +ondas_tras*pdf_tras)/(2*15*15*len(arange(0,t+dt,dt)))
                
                wy_x=wy_ondas.max(0)*prop_x
                wy_ana=wy_ana+wy_ondas+wy_x
                wy_ana[:,0]=wy0[:,0]
                wx_ana=wx_amp*gradient(wy_ana)[0]
            
            #%%RAY TRACING
            omega_ray=wk_y*sn#omega_menos
            v_phase=omega_ray/wk_y
            wy0=fonte(v_phase);
            ondas_tras=cos(omega_ray*t+wk_y*y2_m+wk_x*x2_m)
            wy_tras=wy0*ondas_tras*wy_alt*wy_damp
            
            wy_x=wy_tras.max(0)*cos(wk_x*x2_m)*pdf(wk_x*x2_m/(2.*pi))
            wy_ray=wy_ray+wy_tras+wy_x
            wy_ray[:,0]=wy0[:,0]
        
        if it==0:
            wave_all.append([omega_aw[0,:],omega_gw[0,:],omega_ray[0,:],wk_x,wk_y])
        
#    print (omega_menos.max(),omega_mais.max())
    #%%
    f=rho_tn(rho_o,tn_o)
    rho=f[0];tn=f[1]
    
#%%SIMULACAO NAO-LINEAR 
#    wy_o[:,0]=wy0[:,0]
#    f=AGW()
#    wx=f[0];wy=f[1];rho=f[2];tn=f[3];pn=f[4]
#
    f=vel(b_o,nu_in,gyro_i,wx_ana[:,15:],wy_ana[:,15:])
    vx=f[0];vy=f[1]
    
    n=den_iono(n_o,vx,vy,vnx,vny)
       
#%%===========================ATUALIZAÇÃO EM TEMPO===================#
    dtn=100*(tn-tn_o)/tn_amb;
    drho=100*(rho-rho_o)/rho_amb
    dn=abs(n-n_o)/n_o
#    wx_m=wx_o;wx_o=wx 
#    wy_m=wy_o;wy_o=wy
    rho_o=rho;tn_o=tn;#pn_o=pn
    n_o=n
    
    time.append(t/60.)
#    data_arrival.append([t_arrival/60.,y_arrival])
    dtn3.append(dtn);
    wx3.append(wx_ana)
    wy3.append(wy_ana)
    wy3_ray.append(wy_ray)
    n3.append(n)
#    vw3.append(vy[i_xo,:]/wy0.max())

#=========================================================================#
    
    print ('======================================================')
    print ("TIME=", t,'TIME_STEP=',dt)
    print ('FORCING AMPLITUDE=',round(wy0[:,0].max(),5))
    print ('AGWs Amplitudes=',round(wx_ana.max(),5),round(wy_ana.max(),5)) 
    print ('Amplification ratio=',abs(wy_alt).max())
    print ('TIDs Amplitudes %=',round(100*dn.max(),2)) 
    
#    if wx.max()>100. or wy.max()>100.:
#        break
#    if rho.any()<0 or tn.any()<0:
#        break
    
#    cs=sqrt(1.4*pn/rho);#dt=int(0.5*dy/cs.max())
    t=t+dt
    it=it+1

save('time.npy',array(time))
save('data_amb.npy',array(data_amb))
save('wx3.npy',array(wx3))
save('wy3.npy',array(wy3))
save('wy3_ray.npy',array(wy3_ray))
save('wave_all.npy',array(wave_all))
save('n3.npy',array(n3))
    
i_xo=abs(x-x.mean()).argmin()
t=array(time);
wy3=array(wy3)
wx3=array(wx3)
wy3_ray=array(wy3_ray)
wave_all=array(wave_all)

#%% ESTIMAÇAO DO ARRIVAL TIME 

t_arrival=t[(abs(wy3[:,i_xo,:])>abs(wy3[:,i_xo,0]).max()).argmax(axis=0)]
y_arrival=y

t_ray_arrival=t[(abs(wy3_ray[:,i_xo,:])>abs(wy3_ray[:,i_xo,0]).max()).argmax(axis=0)]
#t_ray_arrival=(1./60.)*y*1.e+03/sn[i_xo,:]
y_ray_arrival=y
t_arrival_0=t_arrival[abs(y_arrival-5).argmin()+1]#.min()
t_ray_arrival_0=t_ray_arrival[abs(y_arrival-5).argmin()]
tau_arrival=cumsum(t_arrival)-t_arrival_0;
tau_ray_arrival=cumsum(t_ray_arrival)-t_ray_arrival_0;

#%%

fig=figure(figsize=(12,12),facecolor='w',edgecolor='k') 
subplot(121)
semilogx(rho_amb[0,:],y2[0,:],'r')
xlabel('Densidade, $kg m^{-3}$');ylabel('Altitude, km')
subplot(122)
plot(sn[0,:],y2[0,:],'b')
xlabel('Acoustic speed, m/s');ylabel('Altitude, km')
draw()


#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(111)
ext=[0,t[-1],x2[0,0],x2[-1,0]]
data=wy3[:,:,0]
vm=1.e-03#data.max()/10.
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');
title('Forcante ou uplift de epicentro')
xlabel('Time, minutos');ylabel('Longitude/Latitude, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('m/s') 
draw()

#%%
for ik in range (1,len(wave_all[:,0])-1,2):
    wl=int(1.e-03*2*pi/wave_all[ik,4])
    omega_aw=wave_all[ik,0];omega_gw=wave_all[ik,1];
    wk_x=wave_all[ik,3];wk_y=wave_all[ik,4]
    
    #%%
    v_aw_phase=omega_aw/wk_y;v_gw_phase=omega_gw/wk_y
    v_ray=sn[i_xo,:]
    
    dtau_aw=0.5*(1./data_antes(1,1,v_aw_phase)+1./data_proximo(1,1,v_aw_phase))*dy_m
    tau_aw_phase=cumsum(dtau_aw)
    dtau_gw=0.5*(1./data_antes(1,1,v_gw_phase)+1./data_proximo(1,1,v_gw_phase))*dy_m
    tau_gw_phase=cumsum(dtau_gw)
    dtau_ray=0.5*(1./data_antes(1,1,v_ray)+1./data_proximo(1,1,v_ray))*dy_m
    tau_ray=cumsum(dtau_ray)
    #%%
    v_aw_group=(wave_all[ik+1,0]-wave_all[ik,0])/(wave_all[ik+1,4]-wave_all[ik,4])
    v_gw_group=(wave_all[ik+1,1]-wave_all[ik,1])/(wave_all[ik+1,4]-wave_all[ik,4])
    dtau_aw=0.5*(1./data_antes(1,1,v_aw_group)+1./data_proximo(1,1,v_aw_group))*dy_m
    tau_aw_group=cumsum(dtau_aw)
    dtau_gw=0.5*(1./data_antes(1,1,v_gw_group)+1./data_proximo(1,1,v_gw_group))*dy_m
    tau_gw_group=cumsum(dtau_gw)

    #%%
    fig=figure(3,figsize=(12,12),facecolor='w',edgecolor='k')
    subplot(111)
    a=1.e+03/(2.*pi)
    semilogx(a*omega_aw,y,'r',lw=5*wl/y[-1],label=str(wl)+'km')
    semilogx(a*omega_gw,y,'b',lw=5*wl/y[-1])
    semilogx(a*omega_br[i_xo,:],y,'g')
    semilogx(a*omega_ac[i_xo,:],y,'c')
    axis('tight')
    legend()
    xlabel('Frequency, mHz');ylabel('Altitude, km');title(r'$\omega$')
    fig=figure(4,figsize=(12,12),facecolor='w',edgecolor='k')
    subplot(121)
    plot(v_aw_phase,y,'r',lw=5*wl/y[-1],label=str(wl)+'km')
    plot(v_gw_phase,y,'b',lw=5*wl/y[-1])
    plot(v_ray,y,'g',lw=3)
    axis('tight')
    legend()
#    axis((0,1500,0,400))
    xlabel('Velocity, m/s');ylabel('Altitude, km');title(r'(A) $\omega/k$')
    ax=subplot(122)
    semilogx(tau_aw_phase/60.,y,'r',lw=5*wl/y[-1])
    semilogx(tau_gw_phase/60.,y,'b',lw=5*wl/y[-1])
    semilogx(tau_ray/60.,y,'g',lw=3)
    axis('tight')
    xlabel('Travel time, Minutes');ylabel('Altitude, km');title(r'(B) $\tau_{ph}$ ')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    fig=figure(5,figsize=(12,12),facecolor='w',edgecolor='k')
    subplot(121)
    plot(v_aw_group,y,'r',lw=5*wl/y[-1],label=r'$\lambda_z$= '+str(wl)+' km')
    plot(v_gw_group,y,'b',lw=5*wl/y[-1])
    plot(v_ray,y,'g',lw=3)
    axis('tight')
    xlabel('Velocity, m/s');ylabel('Altitude, km');title(r'(A) $d \omega/dk$')
    legend()
    ax=subplot(122)
    semilogx(tau_aw_group/60.,y,'r',lw=10*wl/y[-1])
    semilogx(tau_gw_group/60.,y,'b',lw=10*wl/y[-1])
    semilogx(tau_ray/60.,y,'g',lw=3)
    axis('tight')
    xlabel('Travel time, Minutes');ylabel('Altitude, km');title(r'(B) $\tau_{gp}$')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
#%%
vm=10#1.e-03*wy_alt.max()/2.

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(121)
ext=[0,t[-1],y2[0,0],y2[0,-1]]
plot(t_arrival,y_arrival,'ko')
data=wy3[:,i_xo,:]
plot(t,1.e+04*data[:,0],'k-')
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');axis((0,t[-1],-10,400))
title('(A) Vertical propagation of AGWs')
xlabel('Time, minutos');ylabel('Altitude, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('m/s') 
#%%
ax=subplot(122)
plot(t_arrival-t_arrival_0,y_arrival,'ro')
xlabel('Arrival Time, Minutes')
ylabel('Arrival Altitude, km')
title('(B) Arrival diagram')
plot(t-t_arrival_0,1.e+04*data[:,0],'k-')
grid('on')
axis((-5,10,-10,400))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
draw()

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(111)
ext=[0,t[-1],y2[0,0],y2[0,-1]]
#        plot(t_arrival/60.,y_arrival,'ko')
data=wx3[:,i_xo,:]
plot(t,-1.e+04*data[:,0],'k-')
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');axis((0,25,-10,400))
title('(A) Vertical propagation of AGWs')
xlabel('Time, minutos');ylabel('Altitude, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('m/s') 

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(211)
ext=[t[0]-0*t_arrival_0,t[-1]-0*t_arrival_0,x2[0,0],x2[-1,0]]
data=wy3[:,:,abs(y-150.).argmin():abs(y-250.).argmin()].mean(2)
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');
axis((t[0],t[-1],x[0],x[-1]));
plot(t-0*t_arrival_0,1*data[:,i_xo],'g')
title('Horizontal propagation of AGWs')
xlabel('Time, minutos');ylabel('Epicentral Distance, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('m/s') 
ax=subplot(212)
ext=[t[0]-t_arrival_0,t[-1]-t_arrival_0,x2[0,0],x2[-1,0]]
data=wx3[:,:,abs(y-150.).argmin():abs(y-250.).argmin()].mean(2)
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');
axis((t[0],t[-1],x[0],x[-1]));
plot(t-0*t_arrival_0,1*data[:,i_xo],'g')
title('Horizontal propagation of AGWs')
xlabel('Time, minutos');ylabel('Epicentral Distance, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('m/s') 

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
for i_pl in range (1,4):
    it=abs(t-(t_arrival_0+2*i_pl)).argmin()
    ax=subplot(1,3,i_pl)
    ext=[x2[0,0],x2[-1,0],y2[0,0],y2[0,-1]]
    data=array(wy3)[it,:,:]
    im=imshow(data.T,origin='lower',extent=ext,
              cmap=cm.seismic,vmin=-vm,vmax=vm)
    im.set_interpolation('bilinear')
    axis('tight');
    title(str(round(t[it]-t_arrival_0,0))+' Minutos')
    if i_pl ==3:xlabel('Epicentral Distance, km');
    if i_pl==1:ylabel('Altitude, km')
    if i_pl > 1:
        gca().set_yticklabels([])
    if i_pl==5:
        divider = make_axes_locatable(ax);
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=colorbar(im,cax=cax);gca().set_ylabel('m/s') 
    i_pl=i_pl+1
draw()
##%%
#
#fig=figure(5,figsize=(12,12),facecolor='w',edgecolor='k')
#wy_t=array(wy3)[:,i_xo,:]
#subplot(111)
#for iz in range (40,60,4):
#    plot(t-t_arrival_0,wy_t[:,iz])
#grid('on')
#xlabel('Time, minutos');
#ylabel('m/s')

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(121)
ext=[0,t[-1],y2[0,0],y2[0,-1]]
plot(t_ray_arrival,y_ray_arrival,'ko')
data=wy3_ray[:,i_xo,:]
plot(t,1.e+04*data[:,0],'k-')
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');axis((0,25,-10,400))
title('(A) Vertical propagation of AGWs')
xlabel('Time, minutos');ylabel('Altitude, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('m/s') 
#%%
ax=subplot(122)
#if t_arrival > 0:
plot(t_arrival-t_arrival_0,y_arrival,'o')
plot(t_ray_arrival-t_ray_arrival_0,y_ray_arrival,'o')
xlabel('Arrival Time, Minutes')
ylabel('Arrival Altitude, km')
title('(B) Arrival diagram')
plot(t-t_arrival_0,1.e+04*data[:,0],'k-')
grid('on')
axis((-5,10,-10,400))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
draw()


show()


#save('time.npy',array(time))
#save('lon.npy',x)
#save('dtn.npy',array(dtn3))
#save('wx.npy',array(wx3))
#save('wy.npy',array(wy3))
#save('n2.npy',array(n3))
#save('vw1.npy',array(vw3))

#%%
#wy_antes=0*wy_o
#i_arrival=0;
#t_arrival=0;y_arrival=0;i_xo=abs(x-x.mean()).argmin()
#    for iy in range (ny):
#        if wy_antes[i_xo,iy] < wy0[i_xo,iy] and \
#                wy_ana[i_xo,iy] > wy0[i_xo,iy] and i_arrival==0 :
#            t_arrival=t#max(tau_arrival,t)
#            y_arrival=y[iy]#max(y_arrival,y[iy])
#            #if y_arrival==y[0]:
#                #t_arrival_0=t_arrival
#            if y_arrival==y[-1]:
#                i_arrival=1        
#    
#    wy_antes=wy_ana
          #%%ONDAS GRAVIDADE
#            omega_gw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_menos*t_f)**2.)*omega_menos
#            v_phase=omega_gw/wk_y;v_phase_x=omega_gw/wk_x
#            wy0=fonte(v_phase);
#            ondas_frente=cos(-omega_gw*t+wk_y*y2_m+wk_x*x2_m)
#            ondas_tras=cos(omega_gw*t+wk_y*y2_m+wk_x*x2_m)
#            wy_frente=wy0*ondas_frente*wy_alt*wy_damp
#            wy_tras=wy0*ondas_tras*wy_alt*wy_damp
#            wy_ondas=w_amp*(wy_frente+wy_tras)
#            
#                
#            phase_frente=wk_x*(x2_m-v_phase_x*t)
#            phase_tras=wk_x*(x2_m+v_phase_x*t)
#            ondas_frente=cos(phase_frente)
#            ondas_tras=cos(phase_tras)
#    
#            pdf_menos_frente=pdf_menos_frente+pdf(phase_frente/(2.*pi))
#            pdf_menos_tras=pdf_menos_tras+pdf(phase_tras/(2.*pi))
#    
#            prop_x=(ondas_frente*pdf_menos_frente
#                        +ondas_tras*pdf_menos_tras)/(31*31*len(arange(0,t+dt,dt)))
#            
#            wy_x=wy_ondas.max(0)*prop_x
#            wy_menos=wy_menos+wy_ondas+wy_x
#            wy_menos[:,0]=wy0[:,0]
#    wx_mais=wx_mais*gradient(wy_mais)[0]
#    wx_menos=wx_menos*gradient(wy_menos)[0]
#    
#    wy_ana=wy_mais+wy_menos
#    wx_ana=wx_mais+wx_menos