#ESTE CÓDIGO RESOLVE AS EQUAÇÕES DADA POR KHERANI ET AL (2004)..
#O CODIGO USA MKS UNIT
#ESTE CODIGO NUMERICO É COM ERRO DE SEGUNDA ORDEM EM SPAÇO..
#PORTANTO, EXISTE POSIBILIDADE DE MELHORAR...
#TEMBÉM, ESTE CÓDIGO EMPREGA METODO GAUSS-SEIDEL A RESOLVER A EQUAÇÃO DE
#MATRIZ. ESTE MÉTODO É SUBJETIVO..
#O CÓDIGO PODE REPRODUZ OBSEERVAÇÕES ATE 70%-80% QUALITATIVAMENTE..
#O CÓDIGO INCLUI INTERIA PORTANTO É APLICAVÉL ATE 1200 KM...
    
from pylab import *
from numpy import *
from scipy import *
from scipy.ndimage import *

#======subroutina usada na operador Laplaciano
    
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

#==============================================================================
def ambiente(x2,y2):
    
    global no_y,n_o,nu_in,nu_0,gyro_i,b_o
    
    np=1.e+12;yp=350.;r=y2/yp
    a=2;b=-5.
    no_y=2.*np/(exp(a*(r-1.))+exp(b*(r-1.)))                                    #Perfile (em altitude (y)) 
                                                                                #de densidade eletronica (/m³)

    #A seeding Perturbation (no_x) ao longo de longitude (x)..

    wl=400.;wk=2.*pi/wl;am=(5./100.)                                            #(wl,am)=comprimento e amplitude
    no_x=am*cos(wk*x2+pi)                                                       #de perturbação respectivamente

    #A Ionosfera perturbada (n_o) a tempo=0 ===

    n_o=no_y*(1.+no_x)
    
    #===
    sc_h=30.
    nu_in=1.e+03*exp(-(y2-80.)/sc_h)                                            #A perfile (em altitude) da frequencia 
                                                                                #da colisão (nu_in)..
    nu_0=nu_in
    i_500=abs(y-500.).argmin()
    for i in range (i_500,ny):
        nu_0[:,i]=nu_0[:,i_500]
                                                                        
    b_o=30.e-06                                                                 #o campo geomagnetico em Tesla
    q_c=1.6e-19;m_i=1.67e-27;z_i=16.
    gyro_i=q_c*b_o/(z_i*m_i)                                                    # a freuqnecia de giração da ions

    gyro_e=-gyro_i*1837.
    
    return 

#==============================================================================
def vel(b_o,nu,gyro,ex,ey):
    global mu_p                                                                 
    # |vx| | mu_p mu_h | |Ex|
    # |  |=|           | |  |
    # |vy| |-mu_h mu_p | |Ey|
    
    kappa=gyro/nu
    mu_p=kappa/(b_o*(1.+kappa**2.))                                             #PEDERSON MOBILITY
    mu_h=kappa**2./(b_o*(1.+kappa**2.))                                         #HALL MOBILITY
    vx=mu_p*ex+mu_h*ey
    vy=mu_p*ey-mu_h*ex
    return (vx,vy)

#==============================================================================
#=Plano (X-Y) de simulação representa a plano equatorial geomagnetica  em que 
#(+X,+Y) representam oeste (longitude) e vertical para cima (altitude) 
#respectivamente, isto é 
#==============================================================================
dy_km=5.;dx_km=dy_km                                                            #A resoluções espaciais no kilometros
y=arange(200,1200+dy_km,dy_km);ny=len(y)                                        #A faixa de altitude
x=arange(-600,600+dx_km,dx_km);nx=len(x)                                        #A faixa de longitude

dy=dy_km*1.e+03;dx=dx_km*1.e+03                                                 #As resoluções espaciais no metros

[y2,x2]=meshgrid(y,x)

#==============================================================================

f=ambiente(x2,y2)

#==============================================================================
#As forças na ionofera
#==============================================================================
gr_x=0;gr_y=-9.8                                                                # Força gravitacional em metros/segundos²
eo_x=-2.e-03;eo_y=0                                                             #campo eletrico em Volts/metros, +ve para oeste
wo_x=0;wo_y=0                                                                   #vento em metros/segundos

ex_eff=b_o*(eo_x/b_o+wo_y+gr_x/gyro_i)                                          #campo eletrico efectivo=
ey_eff=b_o*(eo_y/b_o-wo_x+gr_y/gyro_i)                                          #E+WXB+Mg/q


#==============================================================================
#Resoluções com (tempo e espço) das equações de densidade (n em m⁻3) e 
#potencial electrostatico (phi em volts)
#==============================================================================
#A equação numerica da densidade:
#fx=vx*(n(t+dt,x+dx,y)-n(t+dt,x-dx,y))/(2.*dx)+vx*(n(t,x+dx,y)-n(t,x-dx,y))/(2.*dx)
#fy=vy*(n(t+dt,x,y+dy)-n(t+dt,x,y-dy))/(2.*dy)+vy*(n(t,x,y+dy)-n(t,x,y-dy))/(2.*dy)
# n(t+dt,x,y)=n(t,x,y)-dt*(fx+fy)/2.
#==============================================================================
#A equação numerica da potancial
#
#phi_x=a*(phi(t+dt,x+dx,y)-phi(t+dt,x-dx,y))/(2.*dx)
#phi_y=b*(phi(t+dt,x,y+dy)-phi(t+dt,x,y-dy))/(2.*dy)
#
#phi_x2=(phi(t+dt,x+dx,y)+phi(t+dt,x-dx,y))/(2.*dx**2)
#phi_y2=(phi(t+dt,x,y+dy)+phi(t+dt,x,y-dy))/(2.*dy**2)
#
#phi(t+dt,x,y)=c*(s-phi_x-phi_y-phi_x2-phi_y2)
#==============================================================================

n=empty((nx,ny));phi=empty((nx,ny));                                            #Declarações como array
it=0;dt=10.;t=0
dv_y=0*phi                                                                      #velocidade da bolha

i_inert=-1                                                                      #i_inert=1 conta inertia

while dv_y.max()<1500.:                                                         #COMEÇA LOOP DE TEMPO QUE TERMINARIA 
                                                                                #QUANDO VELOCIDADE DA BOLHA ULTRAPASSAR
    if i_inert==1:                                                                           #1.5 KM/S
        nu=nu_in+0.05/dt
    else:
        nu=nu_0
    
#==============================================================================        
    f=vel(b_o,nu,gyro_i,eo_x,eo_y)
    vo_ex=f[0];vo_ey=f[1]                                                       #velocidade devido o campo eletrico 
    
    f=vel(b_o,nu,gyro_i,ex_eff,ey_eff)
    vo_x=f[0];vo_y=f[1]                                                         #velocidade devido o campo eletrico effectivo
    
    ep_x=gradient(-phi)[0]/(2.*dx)                                              #campo eletrico de polarização
    ep_y=gradient(-phi)[1]/(2.*dy)
    
    f=vel(b_o,nu,gyro_i,ep_x,ep_y)
    dv_x=f[0];dv_y=f[1]                                                         #velocidade da depelção e blobs
    
    vx=vo_ex+dv_x                                                               #velocidade total
    vy=vo_ey+dv_y
    
    vnx=dx/dt;vny=dy/dt                                                         #VELOCIDADE NUMERICIAS

#==============================================================================    
    #RESOLUÇÂO DE EQUAÇÂO de CONTINUIDADE para DENSIDADE ELETRONICA
    
    fx_o=vx*gradient(n_o)[0]/2.        
    fy_o=vy*gradient(n_o)[1]/2.
    
    n=n_o
    for iter in range (11):                                                     #ITERATION LOOP PARA GAUSS-SEIDEL CONVERGENCE
        fx_g=vx*gradient(n)[0]/2.
        fy_g=vy*gradient(n)[1]/2.
        fx=(fx_o+fx_g)/2.                                                       #SEMI IMPLICIT CRANK-NICOLSON TIME INTEGRATION
        fy=(fy_o+fy_g)/2.
        n=n_o-(fx/vnx+fy/vny)                                                   #EQUAÇÃO NUMERICA DA DENSIDADE
        n[:,0]=n[:,1];n[:,-1]=n[:,-2];                                          #condições contorno na ALTITUDE 
    
    #ATUALIZAÇÂO DE DENSIDADE NO TEMPO    
    n_o=n  
         
#==============================================================================    
    #RESOLUÇÂO DA EQUAÇÂO DA POTENCIAL
    
    a=gradient(n)[0]/(dx*sum_gr(0,nx,n))
    b=gradient(n)[1]/(dy*sum_gr(1,ny,n))
    s_x=vo_x*a/mu_p
    s_y=vo_y*b/mu_p
    s=s_x                                                                       #SOURCE FUNCTION QUE DERIGE A INSTABILIDADE

    for iter in range (11):                                                     #ITERATION LOOP PARA GAUSS-SEIDEL CONVERGENCE
        
        phi_x=a*gradient(phi)[0]/(2.*dx)
        phi_y=b*gradient(phi)[1]/(2.*dy)
        
        phi_2x=sum_gr(0,nx,phi)
        phi_2y=sum_gr(1,ny,phi)
        
        phi_x2=phi_2x/(2.*dx**2.)
        phi_y2=phi_2y/(2.*dy**2.)
        
        c=-(1./dx**2.)-(1./dy**2.)
        phi=(1./c)*(s-phi_x-phi_y-phi_x2-phi_y2)                                #EQUAÇÃO NUMERICA DA POTENCIAL                                                            #

#==============================================================================
        
    #CONDIÇÃO DE FREDRICK-COURANT-LEVY A DETERMINAR TIME STEP (DT)
    
    dt_s=0.5*dy/vy.max()
    dt=max(dt_s,1.)
    
    t=t+dt
    it=it+1
    
    print (dt_s,t,dv_y.max())
#==============================================================================
    
    #PLOTs
    
    if remainder(it,10)==0:
        
        fig=figure(1,figsize=(12,12),facecolor='w',edgecolor='k')
        plot(t,dv_y.max(),'bo')
        title(str(round(vo_ey.max(),2))+', '+str(round(dv_y.max(),2)))
        
        fig=figure(2,figsize=(12,12),facecolor='w',edgecolor='k') 
        plot(100.*no_y[0,:]/no_y.mean(),y2[0,:],'k',lw=1)         
        contour(x,y,n.T,31,colors='g')
        contour(x,y,phi.T,21,cmap=cm.seismic);
        #clabel(cs)
        title(str(round(t,2))+','+str(round(vo_ey.max(),2))+', '+str(round(dv_y.max(),2)))
        xlabel('East-West Distance, km');ylabel('Altitude, km')
        text(-700,100,\
             'Iso-Density contours (em verde) e'+'\n'
             'Iso-Potencial contours (em Red-blue, Red representa +ve potencial)'+'\n'
             'A curva preta reresenta Perfile de densidade eletronica ambiente')
        draw()
        pause(0.1)
        clf()
        
    #LOOP DE TEMPO 
    

fig=figure(2,figsize=(12,12),facecolor='w',edgecolor='k')          
contour(x,y,n.T,31,colors='g')
cs=contour(x,y,phi.T,21,cmap=cm.seismic);
clabel(cs)
title(str(round(t,2))+','+str(round(vo_ey.max(),2))+', '+str(round(dv_y.max(),2)))  

show()
