#ESTE CÓDIGO RESOLVE AS EQUAÇÕES DADA POR KHERANI ET AL (2012, DOI: 10.1111/j.1365-246X.2012.05617.x)..
#(1) EQUAÇÃO DA ONDA PARA AMPLITUDE (W):
#d2w/dt2=(1/rho)grad(1.4 pn div.W)-((grad pn)/rho)div.(rho W)+(1/rho)grad(W.div)p+d((mu/rho)div.div (W)/dt-d(W.div W)/dt
#(2) EQUAÇÃO DA DENSIDADE (rho)
#d rho/dt+div.(rho W)=0
#(3) EQUAÇÃO DA ENERGIA OU PRESSÃO (pn)
#d pn/dt+div. (pn W)+(1.4-1)div. W=0

#O CODIGO USA MKS UNIT
#ESTE CODIGO NUMERICO É COM ERRO DE SEGUNDA ORDEM EM SPAÇO..
#PORTANTO, EXISTE POSIBILIDADE DE MELHORAR...
#TEMBÉM, ESTE CÓDIGO EMPREGA METODO GAUSS-SEIDEL A RESOLVER A EQUAÇÃO DE
#MATRIZ. ESTE MÉTODO É SUBJETIVO..
#O CÓDIGO PODE REPRODUZ OBSEERVAÇÕES ATE 70%-80% QUALITATIVAMENTE..
    
from pylab import *
from numpy import *
from scipy import *
from scipy.ndimage import *

#=============================================================================#
def ndf(x,mu,sig):
    return exp(-power((x-mu)/sig,2.)/2)

#=============================================================================#
def div_f(f0,f1):
    return gradient(f0)[0]/dx2+gradient(f1)[1]/dy2 
           
#==============================================================================
def ambiente(x2,y2):
    
    global rho_o,tn_o,r_g,nu_nn
    
    a=0.25e+25*32*1.6e-27*1.e-08#*1.e+06; 
    c=-4.5;d=-19.5;c=-3.5
    yr=130.;r_y=y2/yr-1
    rho_o=a*(exp(c*r_y)+exp(d*r_y))                                             #Densidade de massa (kg/m³)
    
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
    tn_o=0.5*sixth                                                              #Temperatura atmosferica (K)

    r_g=150.*(1.+sqrt(y2*1.e-03+5.)/5.);                                        #constante Boltzman/massa
    
    b_c=1.38e-23;                                                               #Constante Boltzmann
    mean_mass=b_c/r_g                                                           #massa atmosferica
    nn=rho_o/mean_mass                                                          #Densidade numerica
    pn=r_g*rho_o*tn_o                                                           #Pressão atmosferica
    sn=sqrt(1.4*pn/rho_o)                                                       #velocidade de som
    
    nu_nn=pi*(7*5.6e-11)**2.*sn*nn                                              #frequencia da colisão
    
    return 

def dissip_coef():
#=============================================================================#
#d((mu/rho)div.div (W)/dt
#==============================================================================
    visc_mu=1.3*pn_amb/nu_nn;visc_ki=visc_mu/rho_amb                            #Viscocidade dinamica e kinamatica
    
    gr_flux=gradient(rho_amb*wx_o)[0]/dx2
    w_visc_x=(1./dx2**2.)*gr_flux*visc_mu/rho_amb**2.      
    gr_flux=gradient(rho_amb*wy_o)[1]/dy2
    w_visc_y=(1./dy2**2.)*gr_flux*visc_mu/rho_amb**2.
    
    w_visc_rho=abs(w_visc_x)+abs(w_visc_y)                                      #viscocidade atraves d rho/dt
    w_visc_w=(1./dx2**2.+1./dy2**2.)*visc_ki/dt                                 #viscocidade atraves d w/dt
      
    w_visc=exp(-0.5*dt**2.*(abs(w_visc_rho)+abs(w_visc_w)))
    
#==============Saturação não linear===========================================#
#-d(W.div W)/dt
#=============================================================================#
    fac_nl=wx_o.max()/(dx2*dt)
    w_nl=exp(-1.*dt**2.*fac_nl)
    wx_damp=w_nl*w_visc                                                         #Damping Amplitude em x 
    
    fac_nl=wy_o.max()**2./(dy2*dt)
    w_nl=exp(-0.05*dt**2.*fac_nl)
    wy_damp=w_nl*w_visc                                                         #Damping amlitude em y 
    
    return (wx_damp,wy_damp)


def AGW_terms(i_axis,delta,pn,rho,div_w,div_flux,w_gr_pn):
#=======subroutina da primeira 4 termo da equação da ondas====================#
#w_1=(1/rho)grad(1.4 pn div.W)
#w_2=-((grad pn)/rho)div.(rho W)
#w_3=+(1/rho)grad(W.div)p
#=============================================================================#
    flx=1.4*pn*div_w
    grad=gradient(flx);grad_flux=grad[i_axis]/delta
    w_1=grad_flux/rho 
    
    grad=gradient(pn);grad_pn=abs(grad[i_axis])/delta    #IMPORTANT OF USING ABS
    rho_m=(gradient(gradient(rho)[1])[1]+4.*rho)/(4.)
    w_2=-grad_pn*div_flux/rho_m**2.;        
    
    grad=gradient(w_gr_pn); 
    w_3=grad[i_axis]/delta; 
    
    return (w_1,w_2,w_3)    

#=============================================================================#

def AGW():
    global wx,wy
    
    wx=wx_o;wy=wy_o;
    rho=rho_o;tn=tn_o;pn=pn_o
    
    for i_sor in range (11):
    
#======Equação das ondas AGWs=================================================#
#d2w/dt2=(1/rho)grad(1.4 pn div.W)-((grad pn)/rho)div.(rho W)
#+(1/rho)grad(W.div)p
#+d((mu/rho)div.div (W)/dt-d(W.div W)/dt
#
#w(t+dt,x,y)=w_damp*[2*w(t,x,y)-w(t-dt,x,y)+dt**2*(w_1+w_2+w_3)]
#=============================================================================#        
        pn=r_g*rho*tn;c_s=sqrt(1.4*pn/rho);  
        
        lambda_c=c_s**2./nu_nn
        lambda_mu=pn/nu_nn
    
        #=====================================================================#
        
        wx_0=2.*wx_o-wx_m;
        wy_0=2.*wy_o-wy_m;
        
        #=====================================================================#
        
        div_w=div_f(wx_o,wy_o)       
        div_flux=div_f(rho_o*wx_o,rho_o*wy_o)
        
        gr_pn=gradient(pn_o)
        w_gr_pn=wx_o*gr_pn[0]/dx2+wy_o*gr_pn[1]/dy2
    
        #=====================================================================#

        f_xo=AGW_terms(0,dx2,pn_o,rho_o,div_w,div_flux,w_gr_pn);
        f_yo=AGW_terms(1,dy2,pn_o,rho_o,div_w,div_flux,w_gr_pn);        
        
        #=====================================================================#
        
        div_w=div_f(wx,wy)       
        div_flux=div_f(rho*wx,rho*wy)
        
        gr_pn=gradient(pn)
        w_gr_pn=wx*gr_pn[0]/dx2+wy*gr_pn[1]/dy2
    
        #=====================================================================#

        f_xg=AGW_terms(0,dx2,pn,rho,div_w,div_flux,w_gr_pn);
        f_yg=AGW_terms(1,dy2,pn,rho,div_w,div_flux,w_gr_pn);        
        
        f_x=f_xo+f_xg
        f_y=f_yo+f_yg
        
        #=====================================================================#
        
        f=dissip_coef()
        wx_damp=f[0];wy_damp=f[1];
        
        
        wx=wx_damp*(wx_0+dt**2.*(f_x[0]+f_x[1]+f_x[2])/2.);                     #
        wy=wy_damp*(wy_0+dt**2.*(f_y[0]+f_y[1]+f_y[2])/2.);
       
        wy[:,0]=wy_o[:,0]                                                       #FORCING BOUNDARY CONDITION

#=============================================================================#
#=======RESOLUÇÃO DA EQUAÇÃO DA DENSIDADE=====================================#
#=============================================================================#
        
        div_flux_o=div_f(rho_o*wx_o,rho_o*wy_o)
        div_flux=div_f(rho*wx,rho*wy)
        
        d2_rho=gradient(gradient(rho_o)[0])[0]/dx2**2.\
                +gradient(gradient(rho_o)[1])[1]/dy2**2.
        
        c_diff=exp(-1.*abs(lambda_c*d2_rho*t/rho_o))
        
        rho=rho_o-0.5*c_diff*dt*(div_flux+div_flux_o)/2.
        
#=============================================================================#
#=======RESOLUÇÃO DA EQUAÇÃO DA ENERGIA=====================================#
#=============================================================================#
        
        div_flux_o=div_f(tn_o*wx_o,tn_o*wy_o)        
        div_flux=div_f(tn*wx,tn*wy)        
        div_w=div_f(wx,wy)
        
        d2_tn=gradient(gradient(tn_o)[0])[0]/dx2**2.\
                +gradient(gradient(tn_o)[1])[1]/dy2**2.
        
        c_diff=exp(-1.*abs(lambda_c*d2_tn*t/tn_o))
        
        tn=tn_o-0.5*c_diff*dt*((div_flux+div_flux_o)/2.\
                        +(1.4-1)*tn_o*div_w\
                        -0*lambda_c*d2_tn)
    
    return (wx,wy,rho,tn,pn)

#==============================================================================
#=Plano (X-Y) de simulação representa a plano equatorial geomagnetica  em que 
#(+X,+Y) representam oeste (longitude) e vertical para cima (altitude) 
#respectivamente, isto é 
#==============================================================================
dy_km=5.;dx_km=dy_km                                                            #A resoluções espaciais no kilometros
y=arange(0,400+dy_km,dy_km);ny=len(y)                                        #A faixa de altitude
x=arange(-600,600+dx_km,dx_km);nx=len(x)                                        #A faixa de longitude

dy=dy_km*1.e+03;dx=dx_km*1.e+03                                                 #As resoluções espaciais no metros

[dy2,dx2]=meshgrid(dy,dx)
[y2,x2]=meshgrid(y,x)

#==============================================================================

f=ambiente(x2,y2)

#==================================MAIN=======================================#
#=============================================================================#    
#(wx,wy) são amplitudes da AGWs na direções (x,y) e (rho_o,tn_o,pn_o) são densidade, temeratura e pressão atmosferica
#(wx_m,wy_m)=(wx(t-dt,x,y),wy(t-dt,x,y))
#(wx_o,wy_o)=(wx(t,x,y),wy(t,x,y)
#(rho_o,tn_o,pn_o)=(rho(t-dt,x,y),tn(t-dt,x,y),pn(t-dt,x,y))

global wx_m,wy_m,wx_o,wy_o                                                      
global rho_o,tn_o,pn_o,rho_amb,tn_amb,pn_amb
global t

wx_m=zeros((nx,ny));wy_m=zeros((nx,ny));
wx_o=0*wx_m;wy_o=0*wy_m;

#============================INTIALIZATION====================================#

pn_o=r_g*rho_o*tn_o
rho_amb=rho_o;tn_amb=tn_o;pn_amb=pn_o

#=============================================================================#

it=0;t=0;
dt=dy_km;

#=============================================================================#
#sigma_t é espressura de pacote Gaussiano de tempo
#t_o é tempo em que forçante atinge amplitude maior e deve ser mair do 2*sigma_t
#t_f é tempo final de simulação e deve ser mais de 2*t_o
#=============================================================================#
                                                          
sigma_t=60.*dt;t_o=4*sigma_t
t_f=6.*t_o;nt=int(t_f/dt)+1                                     

while t <=t_f:
    
    sigma_x=4.*dx2*1.e-03;x_o=x2.mean()
    f_lon=ndf(x2,x_o,sigma_x)[:,0]
    f_t=ndf(t,t_o,sigma_t)
    wy_o[:,0]=1.e-00*f_lon*f_t         
    f=AGW()
    wx=f[0];wy=f[1];rho=f[2];tn=f[3];pn=f[4]

#===========================ATUALIZAÇÃO EM TEMPO===================#

    wx_m=wx_o;wx_o=wx
    wy_m=wy_o;wy_o=wy
    dtn=tn-tn_o
    rho_o=rho;tn_o=tn;pn_o=pn
    
#=========================================================================#
    
    print ('======================================================')
    print ("TIME=", t,'TIME_STEP=',dt)
    print ('FORCING AMPLITUDE=',round(wy[:,0].max(),5))
    print ('AGWs Amplitudes=',round(wx.max(),2),round(wy.max(),2)) 
    
    if wx.max()>100. or wy.max()>100.:
        break
    if rho.any()<0 or tn.any()<0:
        break
    
#    cs=sqrt(1.4*pn/rho);dt=0.5*dy/cs.max()
    t=t+dt
    it=it+1
    
    if remainder(it,10)==0:
        fig=figure(1,figsize=(12,12),facecolor='w',edgecolor='k') 
        subplot(121)
        semilogx(rho_amb[0,:],y2[0,:],'r')
        subplot(122)
        plot(tn_amb[0,:],y2[0,:],'b')
        
        fig=figure(2,figsize=(12,12),facecolor='w',edgecolor='k') 
        subplot(121)
        vm=wx.max()/5.
        contour(x2,y2,wx,11,cmap=cm.seismic,vmin=-vm,vmax=vm)
        pcolormesh(x2,y2,wy,cmap=cm.seismic,vmin=-vm,vmax=vm)
        subplot(122)
        vm=dtn.max()/5.
        pcolormesh(x2,y2,dtn,cmap=cm.seismic,vmin=-vm,vmax=vm)
        draw()
        pause(0.1)
        clf()
show()