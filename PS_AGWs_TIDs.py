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

import time
start = time.time()


#%%
pgv=5.e-01;t0=250
def A_lb(am,omega,t,Dt):
    u_cos=am*cos(omega*(t+Dt))
    sigma=40./omega
    u_gauss=skew(t+Dt,t0,sigma,5)
    u_gauss=u_gauss/max(1.e-10,u_gauss.max())
    return u_gauss*u_cos

t=arange(0,1500,0.5);nt=len(t)

rhos=1.e+03 #kgm⁻3
la=1.e+08   #Nm⁻2
mu=la
vp=10*sqrt((la+2*mu)/rhos)
vs=sqrt(mu/rhos)


lmy=2.5e+03  #meters
wk=2*pi/lmy
wkx=wk

for i0 in range (0,1):
    y0=i0**2.*20
    x0=500
    
    for x in range (x0,x0+600,25):
        ux_fonte=skew(x,x0,200,0)
        dtec=0
        for y in range (0,50,5):
            rho=rhos*exp(-(y-10)/20)
            #sr=300*skew(y,0,5,0)+300*skew(y,40,20,0)+500*skew(y,150,1000,25)
            s=vp#.mean()/2
            omega=wk*s
            v_ph=omega/wk
            Dt=-abs(y-y0)*1.e+03/v_ph-abs(x-x0)*1.e+03/s
            
            A0=A_lb(pgv,omega,t,Dt)#*ux_fonte
            
            A_amp=sqrt(rho/rhos)
            A_disp=exp(-omega*20/s)
            
            A_pw=A_amp*A_disp*(A0)
            
            #%%
            
            w_bv=omega/3.#sqrt(ga-1)*9.8/s
            omega=w_bv*wkx/sqrt(wkx**2.+wk**2.)
            
            v_ph=omega/wk
            Dt=-abs(y-y0)*1.e+03/v_ph-abs(x-x0)*1.e+03/v_ph
            
            
            A0=A_lb(pgv,omega,t,Dt)#*ux_fonte
            
            A_disp=exp(-omega*20/v_ph)
            
            A_sw=A_amp*A_disp*(A0)
            
            #%%
            
            w_bv=(wk*vp+omega)/6.#sqrt(ga-1)*9.8/s
            omega=w_bv*wkx/sqrt(wkx**2.+wk**2.)
            
            v_ph=omega/wk
            Dt=-abs(y-y0)*1.e+03/v_ph-abs(x-x0)*1.e+03/v_ph
            
            
            A0=A_lb(pgv*ux_fonte,omega,t,Dt)#*ux_fonte
            
            Dt=-abs(x-x0)*1.e+03/v_ph
            A1=A_lb(A0,omega,t,Dt)
            
            A_disp=exp(-10*omega*20/v_ph)
            
            A_rw=(A_disp/A_amp)*(A1)
            
            #%%
            A=A_rw#(A_pw+A_sw+A_rw)
            
            wy=A;wx=(wkx/wk)*wy
            
            #%%
          
            if x==x0 and i0==0:
            #     print (A.max())
                
                figure(2,(12,12))
                subplot(111)
                plot(t,y+100*A,'g')
                
        if mod(x,100)==0:
            figure(4,(12,12))
            subplot(111)
            plot((t-t0),(x-x0)+50*A)
            #vlines(0, -20, 20, colors='grey')
            xlabel('Time from onset, Minutes')
            ylabel('EW epicnetral distance, km')
        
print (v_ph,A.max())        

import time
end = time.time()
print(end - start)