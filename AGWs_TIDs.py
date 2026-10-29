#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:45:58 2026

@author: alam
"""

from pylab import *
from scipy import *
from numpy import *
from scipy.special import erf

matplotlib.rc("mathtext",fontset="cm")        #computer modern font 
matplotlib.rc("font",family="serif",size=15)


def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)

pgv=5.e-01;t0=500
def A_lb(am,omega,t,t0,Dt):
    u_cos=cos(omega*(t-t0+Dt))
    sigma=2./(omega)
    u_gauss=skew(t+Dt,t0,sigma,5)+0*skew(t+Dt,t0+7200,sigma,5)
    #u_gauss=u_gauss*u_cos
    x=arange(1,2,0.1)
    for it in x:
        u_gauss=u_gauss+0.*skew(t+Dt,t0+3600*it,sigma,5)
    
    u_gauss=u_gauss/max(1.e-10,u_gauss.max())
    return am*u_gauss*u_cos

import time
start = time.time()

t=arange(0,24000,10);nt=len(t)
#dy=10.
yi=arange(0,410,10);ny=len(yi)
#dym=dy*1.e+03


sr=300*skew(yi,0,5,0)+300*skew(yi,40,20,0)+500*skew(yi,150,1000,25)
figure(1,(12,12))
plot(sr/1.5,yi)

lmy=200.e+03  #meters
wk=2*pi/lmy
wkx=wk/1.

pgv_dw=zeros((10,nt))

for i0 in range (0,1):
    y0=0
    x0=3000
    A0=0
    for x in range (x0,x0+4*x0,200):
        ux_fonte=skew(x,x0,200,0)
        dtec=0
        for y in range (0,410,50):
            sr=300*skew(y,0,5,0)+300*skew(y,40,20,0)+500*skew(y,150,1000,25)
            s=sr/1.25#.mean()/2
            omega=wk*s
            v_ph=omega/wk
            v_rlg=3.
            
            Dt=-abs(y-y0)*1.e+03/v_ph-abs(x-x0)/v_rlg
            A0_rlg=A_lb(pgv,omega,t,t0,Dt)*skew(x,x0,2*pi*6371/5,0)
            
            Dt=-abs(y-y0)*1.e+03/v_ph-abs(x-x0)*1.e+03/s
            A0_aw=A_lb(pgv,omega,t,t0,Dt)*skew(x,x0,8*pi*2.e-03/wkx,0)
            
            A0=(A0_aw+A0_rlg)
            
            ga=5./3.;sc_h=3.+y/37.
            mu=-(1./ga)*(y-y0)/sc_h
            
            A_amp=exp(-mu/2.)
            
            rho=exp(-y/sc_h)
            nu=1.e-04/rho
            
            A_dis=exp(-nu*wk**2./omega)
            
            
            w=A0*A_amp*A_dis#*cos(wkx*x+wk*y-omega*(t+Dt))
            A_nl=exp(-w**2./v_ph**2.)
            
            A_aw=A_nl*w
            #%%
            A_uw=A_aw
            
            for i in range (7):
                it=abs(A_uw-A_uw.max()).argmin()
                Dt=abs(y-y0)*1.e+03/v_ph#-0*abs(x-x0)*1.e+03/v_ph-abs(x-x0)/v_rlg
                A_dw=A_lb(A_uw[it],omega,t,t0+t[it],Dt)
                        
                if y==0:pgv_dw[i,:]=A_dw
                
                it=abs(pgv_dw[i,:]-pgv_dw[i,:].max()).argmin()
                Dt=-abs(y-y0)*1.e+03/v_ph#-0*abs(x-x0)*1.e+03/s-abs(x-x0)/v_rlg
                A_uw=A_amp*A_dis*A_nl*A_lb(pgv_dw[i,it],omega,t,t0+t[it],Dt)
                
                
                if x==x0:
                    print (i,y,t[it],A_uw.max())
                    figure(11,(12,12))
                    subplot(111)
                    plot(t,y+0.1*A_dw,'r')
                    plot(t,y+0.1*A_uw,'b')
                    plot(t,y+0.1*A_aw,'g')
                A_aw=A_aw+A_uw+A_dw
                
            #%%
            
            w_bv=sqrt(ga-1)*9.8/s
            omega=w_bv*wkx/sqrt(wkx**2.+wk**2.)
            
            v_ph=omega/wk
            Dt=-abs(y-y0)*1.e+03/v_ph-abs(x-x0)*1.e+03/v_ph-abs(x-x0)*1.e+03/s
            
            
            A0=A_lb(pgv,omega,t,t0,Dt)#*ux_fonte
            
            A_dis=exp(-nu*wk**2./omega)
            w=A0*A_amp*A_dis#*cos(wkx*x+wk*y-omega*t)
            A_nl=exp(-w**2./v_ph**2.)
            
            A_gw=A_nl*w    
            
            #%%
            A=(A_aw+A_gw)
            
            wy=A;wx=(wkx/wk)*wy
            
            #%%
            ang_i=10*pi/180.
            n=1.e+12/(exp(2*(y/400-1))+exp(-5*(y/400-1)))
            nu=1.e+03*exp(-(y-80)/sc_h);omg=100.;b0=2.5e-05
            ka=omg/nu;mup=(1./b0)*ka/(1+ka**2.);muh=ka*mup
            ex=wy*b0*cos(ang_i);ey=-wx*b0*cos(ang_i)
            wix=mup*ex+muh*ey;wiy=mup*ey-muh*ex
            dn=-(0.5/omega)*(wkx*wix+wk*wiy)*n
            if y >=150 and y <=250:dtec=dtec+dn*50*1.e+03/1.e+16
            if x<x0+100:
                figure(2,(12,12))
                subplot(111)
                plot(t/60,y+50*dn/n,'g')
               
            
           
        figure(4,(12,12))
        subplot(111)
        plot((t-t0)/60.,x-x0+100*dtec)
        vlines(0, -200, 200, colors='grey')
        xlabel('Time from onset, Minutes')
        ylabel('EW epicnetral distance, km')
        title(r'PGV='+str(pgv)+'m/s, TEC-Residual='+str(around(dtec.max(),decimals=2))+'TECU')
    
print (dtec.max())

import time
end = time.time()
print(end - start)