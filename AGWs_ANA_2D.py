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
from pyiri2016 import *
from nrlmsise_2000 import *
from scipy import *
from scipy.ndimage import *
from scipy.special import erf
from scipy.integrate import trapz
from matplotlib.signal_alam import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rc("mathtext",fontset="cm")        #computer modern font 
matplotlib.rc("font",family="serif",size=12)

def d1_3(n2,n3,data):
    return repeat(repeat(data[:,newaxis],n2,axis=1)[:,:,newaxis],n3,axis=2)

def d1_2(n,data):
    return repeat(data[newaxis,:],n,axis=0)

def d1_23(n,data):
    return repeat(data[newaxis,:,:],n,axis=0)

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
        lat_ep=38;lon_ep=143
        year,month,dom=2011,3,9
                
        d0 = datetime.date(year,1,1)
        d1 = datetime.date(year, month, dom)
        delta = d1 - d0
        doy=delta.days 
        ut=3;lt=ut+lon_ep/15.
        
        f107A,f107,ap=150,150,4 #300,300,400
        
        f=nrl_msis(doy,ut*3600.,lt,f107A,f107,ap,lat_ep,lon_ep,dy,y[0],ny)
        tn_msis=f[1];#tn_msis=0*tn_msis+tn_msis.mean()
        den_ox=f[2]*1.e+06;den_n=f[3]*1.e+06;den_o2=f[4]*1.e+06;den_n2=f[5]*1.e+06;
        n_msis=den_ox+den_n+den_o2+den_n2;
        rho_msis=f[6]*1.e+03
        mean_mass=rho_msis/n_msis
        
        b_c=1.38e-23; 
        rg_msis=b_c/mean_mass;
        pn_msis=rg_msis*rho_msis*tn_msis;
        sn_msis=sqrt(1.33*pn_msis/rho_msis)
        
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
        sn=sqrt(1.33*pn/rho_amb)                                                       #velocidade de som
        
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
def rho_tn(rho_o,tn_o,pn_o):
    div_w=0*div_f(wx_ana,wy_ana)
    div_flux=div_f(rho_o*wx_ana,rho_o*wy_ana)
    div_flux_x=wx_ana*gradient(rho_o)[0]/gradient(x2_m)[0]
    div_flux_y=wy_ana*gradient(rho_o)[1]/gradient(y2_m)[1]
    div_flux=div_flux_x+div_flux_y
    rho=rho_o-0.1*dt*div_flux
    
    div_flux=div_f(tn_o*wx_ana,tn_o*wy_ana)
    div_flux_x=wx_ana*gradient(tn_o)[0]/gradient(x2_m)[0]
    div_flux_y=wy_ana*gradient(tn_o)[1]/gradient(y2_m)[1]
    div_flux=div_flux_x+div_flux_y        
    
    gma=1.33#rho_ho/rho
    tn=tn_o-0.1*dt*(div_flux+(gma-1.)*tn_o*div_w)
    
    div_flux=div_f(pn_o*wx_ana,pn_o*wy_ana)        
    div_flux_x=wx_ana*gradient(pn_o)[0]/gradient(x2_m)[0]
    div_flux_y=wy_ana*gradient(pn_o)[1]/gradient(y2_m)[1]
    div_flux=div_flux_x+div_flux_y
    
    pn=pn_o-0.1*dt*(div_flux+(gma-1.)*pn_o*div_w)/1.
    
#    div_flux=1.e-03*wy_ana*gradient(tn_o)[1]/gradient(y2)[1]
    rho_t=rho_to#+dt*(rho_ho+rho_to)*div_flux/tn_o
#
#    div_flux=div_f(rho_ho*wx_ana,rho_ho*wy_ana)
    rho_h=rho_ho#-0.*dt*div_flux
#    
    return (rho,tn,pn)

#%%ANALTICA
def AGW_ana():
    i_xo=abs(x-x.mean()).argmin()
    gma=1.33#0.33+rho_ho/rho_o#1.4#0.01+rho_ho/(rho_ho+rho_to)
    pn=pn_o#r_g*rho_o*tn_o;
    c_s=sqrt(gma*pn/rho_o);#c_s=0*c_s+c_s.mean() 
    gr_pn=gradient(pn)[1];dy_m2=2.*dy_m
    zeta=(1./rho_o)*gr_pn/dy_m2
    k0=zeta/c_s**2.
    k0_antes=data_antes(2,1,k0)
    k0_proximo=data_proximo(2,1,k0)
#    mu=0.5*(k0)*y2_m#(k0)*y2*1.e+03#(k0_proximo+k0_antes)*dy_m#0.1*(k0-k0[0,0])*y2*1.e+03#trapz(k0,x=y2,dx=dy_m,axis=1)
    mu=0.5*(k0_proximo+k0_antes)*dy_m2/2.
    mu=cumsum(mu,1);mu=9.*mu/abs(mu).max()
#    i_mx=abs(abs(mu[0,:])-abs(mu).max()).argmin()
#    mu[:,i_mx:]=0*mu[:,i_mx:]+mu[0,i_mx]
    omega_c2=(gma**2.*k0*c_s)**2./4.
    omega_b2=((gma-1)*k0**2-1.*(k0/c_s**2.)*gradient(c_s**2.)[1]/dy_m2)*c_s**2.
    i_pos=argwhere(omega_b2[i_xo,:]>=0);i_neg=argwhere(omega_b2[i_xo,:]<0)
    ob2_real=0*omega_b2;ob2_im=0*omega_b2
    ob2_real[:,i_pos]=omega_b2[:,i_pos]   
    ob2_im[:,i_neg]=omega_b2[:,i_neg]   
#    omega_b2=sqrt(ob2_real**2.+ob2_im**2.)
    # omega_b2=0*omega_b2+omega_b2.mean()
    omega_h2=wk_x**2.*c_s**2.
    omega_2=omega_b2+(wk_y**2.+0*k0**2./4.)*c_s**2.+omega_h2
    omega_mais=sqrt(omega_2+sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
    omega_menos=sqrt(omega_2-sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
#    omega_mais=sqrt(omega_2)
#    omega_menos=sqrt(omega_h2/omega_2)*sqrt(ob2_real**2.+ob2_im**2.)
#    if omega_menos.any()==0.:
#        omega_menos=omega_mais
    visc_mu=1.3*pn_amb/nu_nn;visc_ki=visc_mu/rho_amb
    nu_col=visc_ki*(-wk_x**2-wk_y**2.+k0**2.-gradient(k0)[1]/dy_m2)
    wx_mais=(omega_mais**2.+omega_h2-omega_2)/(wk_x*wk_y*c_s**2.)
    wx_menos=(omega_menos**2.+omega_h2-omega_2)/(wk_x*wk_y*c_s**2.)
    
    gamma_ad=(gma-1)*k0**2.
    gamma_e=(k0/c_s**2.)*gradient(c_s**2.)[1]/dy_m2
    return (mu,omega_mais,omega_menos,nu_col,wx_mais,wx_menos,omega_b2,\
            omega_c2,ob2_im,gamma_ad,gamma_e,c_s)

#%%ANALTICA
def AGW_ana_x():
    gma=1.33#0.33+rho_ho/rho_o#1.4#0.01+rho_ho/(rho_ho+rho_to)
    pn=pn_o#r_g*rho_o*tn_o;
    c_s=sqrt(gma*pn/rho_o); 
    gr_pn=gradient(pn)[0];dx_m2=2.*dx_m
    zeta=(1./rho_o)*gr_pn/dx_m2
    k0=zeta/c_s**2.
    omega_c2=(gma**2.*k0*c_s)**2./4.
    omega_b2=((gma-1)*k0**2-0.*(k0/c_s**2.)*gradient(c_s**2.)[0]/dx_m2)*c_s**2.
    omega_h2=wk_y**2.*c_s**2.
    omega_2=omega_b2+(wk_x**2.+k0**2.)*c_s**2.+omega_h2
    omega_mais=sqrt(omega_2+sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
    omega_menos=sqrt(omega_2-sqrt(omega_2**2.-4.*omega_h2*omega_b2))/sqrt(2)
    return (omega_mais,omega_menos)
#%%
def amb_iono(iw,x2,y2):
    global no_y,n_o,nu_in,nu_0,gyro_i,b_o
    if iw==0:
        yi=y2[0,8:]
        altlim=[yi[0],yi[-1]]
        iri2016=IRI2016Profile(altlim=altlim,altstp=10,lat=38,\
            lon=143, year=2011, month=3, dom=9, iut=1,hour=3,\
            option=1, verbose=False) 
        
        index = range(len(yi))
        
        i_ne,i_no,i_no2,i_nno=0,4,7,8
        ne_iri = iri2016.a[0, index] # 
        no_iri= iri2016.a[4, index]*50.e+08# 
        no2_iri= iri2016.a[7, index]*10.e+08 # 
        nno_iri= iri2016.a[8, index]*10.e+08 #
            
        ti = iri2016.a[2, index]
        te = iri2016.a[3, index]
        
        n_o=d1_2(nx,ne_iri)
        
        subplot(111)
        semilogx(n_o[0,:], yi[:],'gray',lw=2,label='$n_o, m^{-3}$')   
        ylabel('Altitude, km')
        title('(B)')# Ionospheric Number Density')
        legend(fontsize=12,loc='best')
        
        sc_h=30.
        nu_in=1.e+03*exp(-(yi-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                    #da colisão (nu_in)..
    #    nu_0=nu_in
    #    i_500=abs(y-500.).argmin()
    #    for i in range (i_500,ny):
    #        nu_0[:,i]=nu_0[:,i_500]
                                                                            
        b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
        q_c=1.6e-19;m_i=1.67e-27;z_i=16.
        gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions
    
        gyro_e=-gyro_i*1837.
        
    #    m_p=1.67e-27;
    #    pn_msis=b_c*n_msis*tn_msis
    #    sn_msis=sqrt(1.4*pn_msis/rho_msis)
    #    mass_msis=rho_msis/n_msis
    #    
    #    omg_i=q_charge*mag_o/(mass_msis[n1-n1_i:]);
    #    omg_e=-omg_i*1837.
    #    a=mass_msis[n1-n1_i:]/m_p#16.;#a is neutral mass in amu
    #
    #    nu_ii=0.22*ne_iri*1.e-06/(ti)**(3./2.);
    #    nu_in=2.6e-9*1.e-06*(5*n_msis[n1-n1_i:]+ne_iri)*a**(-1./2.)        
    #    a=2.*sn_msis[n1-n1_i:]*8./(3.*sqrt(pi))
    #    nu_in=1.e+06*a*rho_msis[n1-n1_i:]
    #    nu_in=1.*nu_in+0*nu_ii
    #    
    #    nu_ei=34*ne_iri*1.e-06/(ti)**(3./2.);
    #    nu_en=5.4e-16*n_msis[n1-n1_i:]*sqrt(ti)+0*nu_ei
    #    nu_en=1.*nu_ei+0*nu_en
    if iw==1:
        y2=y2[:,8:]
        np=1.e+12;yp=200.;r=y2/yp
        a=2;b=-10.
        n_of=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))                                    #Perfile (em altitude (y)) 
        
        np=1.e+11;a=5.;b=-60.
        n_oe=2.*np/(exp(a*(r-0.5))+exp(b*(r-0.5)))
        
        n_ef=5.e-01*np*(r+0.5)**2./5.
        n_o=n_of+n_oe+n_ef
        
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
        
    return ()   

#%%
def ambiente_iono(x2,y2):
    
    global no_y,n_o,nu_in,nu_0,gyro_i,b_o
    y2=y2[:,8:]
    np=1.e+12;yp=200.;r=y2/yp
    a=2;b=-10.
    n_of=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))                                    #Perfile (em altitude (y)) 
    
    np=1.e+11;a=5.;b=-60.
    n_oe=2.*np/(exp(a*(r-0.5))+exp(b*(r-0.5)))
    
    n_ef=5.e-01*np*(r+0.5)**2./5.
    n_o=n_of+n_oe+n_ef
    
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
    #vy=vy-0
    fx_o=vx*gradient(n_o)[0]/2.        
    fy_o=vy*gradient(n_o)[1]/2.
    
    n=n_o
    for iter in range (11):                                                     #ITERATION LOOP PARA GAUSS-SEIDEL CONVERGENCE
        fx_g=vx*gradient(n)[0]/2.
        fy_g=vy*gradient(n)[1]/2.
        fx=(fx_o+fx_g)/2.                                                       #SEMI IMPLICIT CRANK-NICOLSON TIME INTEGRATION
        fy=(fy_o+fy_g)/2.
        n=n_o-0.25*(fx/vnx+fy/vny)                                                   #EQUAÇÃO NUMERICA DA DENSIDADE
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
    lat=radians(0)
    mag_m=8.e+15         #Tm^3
    r_ea=6.371e+06
    by=-2.*mag_m*sin(abs(lat))/r_ea**3.;
    bz=mag_m*cos(lat)/r_ea**3.;
    bx=0
    wz=0
    ey=wz*bx-wx*bz
    ex=wy*bz-wz*by
    ez=wx*by-wy*bx
#    ex=wy*b_o*cos(lat);ey=-wx*b_o*cos(lat)
    vx=mu_p*ex+mu_h*ey
    vy=mu_p*ey-mu_h*ex
    return (vx,vy)

#%%============================FONTE SISMICA============================================#
data_sism= load('DATA_ILLAPEL/Dados_sismico.npy')
t_s=3600*data_sism[0,280000:-50000:200];#segundos
vel_s=data_sism[1,280000:-50000:200]#m/s
t_s=t_s-t_s[0]
f=wavelet(0,t_s,vel_s,64)
pd=f[0];pwr=f[1];emd=f[2];amp_sism=f[3][:,1][0]
fr_sism=1.e+00/(pd)

# for iw in range (1):
#     idx_peaks=find_peaks(abs(emd[iw,:]))
#     # idx_peaks[0]=argmax(abs(emd[iw,:]))
#     n_peaks=len(idx_peaks)
#     print (n_peaks)
#     t0=zeros(n_peaks);amp=zeros(n_peaks)
#     for i_peak in range (n_peaks):
#         t0[i_peak]=t_s[idx_peaks[i_peak]];
#         #print (len(idx_peaks),t0)
#         amp[i_peak]=emd[iw,idx_peaks[i_peak]]

def fonte(v_phase):
    global sigma_t,t0,sigma_x,v_s,tn_phase,tn0               
    vel_ana=0
    
    for iw in range (len(pd)):
        idx_peaks=find_peaks(abs(emd[iw,:]))[:,0]
        n_peaks=len(idx_peaks)
        t0=t_s[idx_peaks]
        amp=emd[iw,idx_peaks]
        sigma_t=pd[iw]
        nx=41;ny=41
        tn0=d1_2(ny,t0)
        ampn=d1_2(ny,amp)
        if abs(v_phase).min() !=0:
            t_phase=y2_m/v_phase
        else:
            t_phase=0+0*y2_m
        t_phase=t_phase.mean(0)
        tn_phase=transpose(d1_2(n_peaks,t_phase))     
        f_sism=skew(t-tn_phase,tn0,sigma_t/2.,0)
        vel_ana=vel_ana+(f_sism*ampn).sum(1)
        # idx_peaks[0]=argmax(abs(emd[iw,:]))
        # n_peaks=len(idx_peaks)
        # for i_peak in range (n_peaks):
        #     t0=t_s[idx_peaks[i_peak]];
        #     #print (len(idx_peaks),t0)
        #     sigma_t=pd[iw]
        #     f_sism=skew(t-t_phase,t0,sigma_t/2.,0)
        #     f_sism=f_sism-f_sism[0]
        #     amp=emd[iw,idx_peaks[i_peak]]
        #     vel_ana=vel_ana+f_sism*amp#*cos(2*pi*fr_sism[iw]*(t-t_phase))
    wy0_t=vel_s.max()*vel_ana/emd.max()
    sigma_x=4.*dx_m*1.e-03;x_o=x2.mean();v_s=0.;x_o=0
    f_lon=skew(x2,x_o-v_s*t,sigma_x/1.,0)

    wy0=wy0_t*(f_lon-f_lon[0,0])
    return wy0


#%%==================================MAIN====================================== 

global wx_m,wy_m,wx_o,wy_o                                                      
global rho_o,tn_o,pn_o,rho_amb,tn_amb,pn_amb
global fac_amp,mask
global wk_x,wk_y

#%%
dy=10;dx=1.*dy;dt=dy/2.;                                                            #A resoluções espaciais no kilometros
y=arange(0,400+dy,dy);ny=len(y)                                           #A faixa de altitude
x=arange(-200,200+dx,dx);nx=len(x)                                        #A faixa de longitude
[y2,x2]=meshgrid(y,x)
y2_m=y2*1.e+03;x2_m=x2*1.e+03
dy_m=dy*1.e+03;dx_m=dx*1.e+03                                                 #As resoluções espaciais no metros

it=0;nt=1.5*360;t=0;t_f=nt*dt
vnx=dx_m/dt;vny=dy_m/dt #VELOCIDADE NUMERICIAS

#%%
wx_m=zeros((nx,ny));wy_m=zeros((nx,ny));
wx_o=0*wx_m;wy_o=0*wy_m;
time=[]
dtn3=[];wx3=[];wy3=[];wy3_ray=[];n3=[];vw3=[];data_arrival=[]
wave_all=[];data_amb=[];eta=[];pr3=[];rho3=[];tn3=[];
o_br=[];omega_all=[];gr_ci3=[]

#%%============================================================================

f=ambiente_atmos(0,x2,y2)
#f=ambiente_iono(x2,y2)
f=amb_iono(0,x2,y2)     
f=fonte(1.*y2)
g_e=(1./tn_amb)*gradient(tn_amb)[1]/gradient(y2)[1];
ln_tn=log(tn_amb[0,0])+1.33*cumsum(dy*g_e,1)
tn_new=exp(ln_tn)
#tn_amb=tn_new
#tn_amb=tn_amb.mean()+0*tn_amb
data_amb.append((rho_amb[0,:],sn[0,:]))

#%%============================INTIALIZATION===================================


pn_amb=r_g*rho_amb*tn_amb
rho_o=rho_amb;tn_o=tn_amb;pn_o=pn_amb
rho_to=0*rho_o;rho_ho=rho_amb

fac_amp=sqrt(data_antes(2,1,rho_amb)/data_proximo(2,1,rho_amb))#
#fac_amp=sqrt(abs(rho_amb/rho_amb.max()))
fac_amp=exp(-fac_amp)
mask=mask_b()
                                                          
#%%SOLUCAO ANALYTICA

i_pl=1
pdf_frente=zeros((nx,ny))
pdf_tras=zeros((nx,ny))
a_frente=1.+zeros((nx,ny))
a_tras=1.+zeros((nx,ny))
a_frente[x<0]=0
a_tras[x>=0]=0
lambda_y0=arange(2.*dy_m,ny*dy_m/3.,dy_m)
lambda_x0=arange(4.*dx_m,nx*dx_m/2.,2*dx_m)
while t <=t_f/1.:
    wy_ray=zeros((nx,ny))
    wy_ana=zeros((nx,ny));wx_ana=zeros((nx,ny));
    for ik in range (len(lambda_y0)):
        lambda_x=10.*dx_m;
        lambda_y=lambda_y0[ik]
        lambda_x0=arange(lambda_y,nx*dx_m/2.,2*dx_m)
        for ikx in range (len(lambda_x0)):
            lambda_x=lambda_x0[ikx]#max(sigma_x,2*ikx*dx_m)
            wk_x=2.*pi/lambda_x;wk_y=2.*pi/lambda_y    
            
            f=AGW_ana()
            mu=f[0];omega_mais=f[1];omega_menos=f[2];nu_col=f[3];
            wx_mais=f[4];wx_menos=f[5]
            omega_br=sqrt(f[6]);omega_ac=sqrt(f[7]);
            omega_ci=sqrt(abs(f[8]))
            gamma_ad=f[9];gamma_e=f[10]
            c_s=f[11]
            
            f=AGW_ana_x()
            omega_awx=f[0];omega_gwx=f[1]
            
            if ik==0 and ikx==ik and omega_ci.any()!=0:
                print ('CONVECTIVELY UNSTABLE GWs')                

            w_amp=1
            if omega_mais.max()/(2.*pi) > 1./(2.*dt) or omega_awx.max()/(2.*pi) > 1./(2.*dt):
                w_amp=0
            
            wy_alt=exp(-mu)
            n=1.
            wy_damp=exp(2.*nu_col*t/(2.*n))*exp(-lambda_c*t*wk_y**2.)
            wy_growth=exp(0*omega_ci*t/(2.*pi))
            #%%ONDAS ACUSTICAS
            omega_aw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_mais.max()*t_f)**2.)*omega_mais
            #ONDAS ACUSTICAS
            omega_gw=(1./n)*sqrt(1.-(n-1)**2./(4.*omega_menos.max()*t_f)**2.)*omega_menos
#            omega_gw=omega_menos
#            if omega_aw.any() < omega_ac.any():
#                continue
            for i_wv in range (1):
                if i_wv==0: 
                    omega=omega_aw
                    wx_amp=wx_mais/100.
                    omega_x=omega_awx
                if i_wv==1: 
                    omega=omega_gw
                    wx_amp=wx_menos/100.
                    omega_x=omega_gwx
                
                x2_mv=x2_m+v_s*t*1.e+03
                v_phase=omega/wk_y;
                wy0=fonte(v_phase);
                ondas_frente=cos(-omega*t+wk_y*y2_m+0*wk_x*x2_m)
                ondas_tras=cos(omega*t+wk_y*y2_m+0*wk_x*x2_m)
                wy_frente=wy0*ondas_frente*wy_alt*wy_damp*wy_growth
                wy_tras=wy0*ondas_tras*wy_alt*wy_damp*wy_growth
                wy_ondas=w_amp*(wy_frente+wy_tras)/2.
                
                v_phase_x=omega_x/wk_x
                if abs(v_phase_x).min() !=0:
                    t_phase=x2_m/v_phase_x
                else:
                    t_phase=0
                sigma_tx=2.*pi/omega_x;sigma_x2=2.*pi/wk_x
                
                ondas_frente=cos(-omega_x*t+wk_x*x2_mv)
                ondas_tras=cos(omega_x*t+wk_x*x2_mv)
                wy0_x=(wy_ondas.max(0)+wy_ondas)/2.
                wy_frente=wy0_x*ondas_frente#*skew(0*(t-t_phase),0,sigma_tx,0)
                wy_tras=wy0_x*ondas_tras#*skew(0*(t+t_phase),0,sigma_tx,0)
                wx_ondas=(a_frente*wy_frente+a_tras*wy_tras)*skew(x2_mv,0,sigma_x2,0)
                
                f_smooth=1.-0.9*skew(4.*x2_mv,0,sigma_x2,0)
                wy_ana=wy_ana+(wy_ondas+wx_ondas)*f_smooth
                wy_ana[:,0]=wy0[:,0]
                wx_ana=wx_ana+wx_amp*gradient(wy_ana)[0]
            
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
                wave_all.append([omega_aw[0,:],omega_gw[0,:],omega_ray[0,:],\
                                 wk_x,wk_y,omega_br,omega_ac,c_s[0,:]])
        
#    print (omega_menos.max(),omega_mais.max())
    #%%
    f=rho_tn(rho_o,tn_o,pn_o)
    rho=f[0];tn=f[1];pn=f[2]
#    pn=r_g*rho*tn
    gma=rho_amb/rho
    print (gma.max())
    
#%%SIMULACAO NAO-LINEAR 
#    wy_o[:,0]=wy0[:,0]
#    f=AGW()
#    wx=f[0];wy=f[1];rho=f[2];tn=f[3];pn=f[4]
#
    f=vel(b_o,nu_in,gyro_i,wx_ana[:,8:],wy_ana[:,8:])
    vx=f[0];vy=f[1]
    
    n=den_iono(n_o,vx,vy,vnx,vny)
       
#%%===========================ATUALIZAÇÃO EM TEMPO===================#
    dtn=rho_amb/rho#gradient(tn)[1]/gradient(y2)[1]#100*(tn-tn_o)/tn_amb;
    drho=100*(rho-rho_o)/rho_amb
    dn=abs(n-n_o)/n_o
#    wx_m=wx_o;wx_o=wx 
#    wy_m=wy_o;wy_o=wy
    rho_o=rho;tn_o=tn;pn_o=pn
    n_o=n
    
    time.append(t/60.)
#    data_arrival.append([t_arrival/60.,y_arrival])
    rho3.append(rho);tn3.append(tn)
    pr3.append(pn);
    wx3.append(wx_ana)
    wy3.append(wy_ana)
    wy3_ray.append(wy_ray)
    gr_ci3.append(omega_ci)
    i_xo=abs(x-x.mean()).argmin()
    omega_all.append([omega_ac[i_xo,:],omega_br[i_xo,:],omega_ci[i_xo,:],\
                      gamma_ad[i_xo,:],gamma_e[i_xo,:]])
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
save('wy3.npy',array(wy3))
save('wave_all.npy',array(wave_all))
save('n3.npy',array(n3))

#%%
i_xo=abs(x-x.mean()).argmin()
t=array(time);
n3=array(n3);n3=gradient(n3)[0]
wy3=array(wy3)
wx3=array(wx3)
wy3_ray=array(wy3_ray)
eta=array(o_br)
wave_all=array(wave_all)

#%% ESTIMAÇAO DO ARRIVAL TIME 
t_arrival=t[(abs(wy3[:,i_xo,:])>abs(wy3[:,i_xo,0]).max()).argmax(axis=0)]
y_arrival=y
t_arrival_0=t_arrival[abs(y_arrival-5).argmin()+1]#.min()
tau_arrival=cumsum(t_arrival)-t_arrival_0;

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k') 
subplot(121)
semilogx(rho_amb[0,:],y2[0,:],'r')
xlabel('Densidade, $kg m^{-3}$');ylabel('Altitude, km')
subplot(122)
plot(sn[0,:],y2[0,:],'b')
xlabel('Acoustic speed, m/s');ylabel('Altitude, km')

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(111)
ext=[0,t[-1],x2[0,0],x2[-1,0]]
data=wy3[:,:,0]
vm=1.e-02#data.max()/10.
plot(t,1.e+03*data[:,i_xo])
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');
title('Forcante ou uplift de epicentro')
xlabel('Time, minutos');ylabel('Longitude/Latitude, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('m/s') 

#%%
vm=250
#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(111)
ext=[0,t[-1],y2[0,0],y2[0,-1]]
data=wy3[:,i_xo,:]
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');axis((0,t[-1],0,400))
title('CTDs, Vertical propagation of Pressure Disturbance')
xlabel('Time, minutes');ylabel('Altitude, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('Pa') 

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
ax=subplot(211)
ext=[t[0]-0*t_arrival_0,t[-1]-0*t_arrival_0,x2[0,0],x2[-1,0]]
ax=subplot(111)
ext=[t[0],t[-1],x2[0,0],x2[-1,0]]
data=n3[:,:,abs(y-180.).argmin():abs(y-200.).argmin()].mean(2)
im=imshow(data.T,origin='lower',extent=ext,cmap=cm.seismic)#,vmin=-vm,vmax=vm)
im.set_interpolation('bilinear')
axis('tight');
axis((t[0],t[-1],x[0],x[-1]));
#plot(t,1*data[:,i_xo],'g')
title('Pressure Disturbance, Horizontal cross section')
xlabel('Time, minutos');ylabel('Epicentral Distance, km')
divider = make_axes_locatable(ax);
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar=colorbar(im,cax=cax);gca().set_title('Pa') 

#%%
fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
for i_pl in range (1,6):
    it=abs(t-(8+16*i_pl)).argmin()
    ax=subplot(1,5,i_pl)
    ext=[x2[0,0],x2[-1,0],y2[0,0],y2[0,-1]]
    data=wy3[it,:,:]
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
        cbar=colorbar(im,cax=cax);gca().set_ylabel('Pa') 
    i_pl=i_pl+1
draw()

show()
