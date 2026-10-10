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


def A_lb(omega,t,Dt):
    am=1.e-02;t0=500
    u_cos=am*cos(omega*(t+Dt))
    sigma=2./omega
    u_gauss=skew(t+Dt,t0,sigma,5)
    u_gauss=u_gauss/u_gauss.max()
    return am*u_gauss

t=arange(0,4000,1);nt=len(t)
dy=2.
y=arange(0,410,2);ny=len(y)
dym=dy*1.e+03

sr=300*skew(y,0,5,0)+300*skew(y,40,20,0)+500*skew(y,150,1000,25)

lmy=250.e+03  #meters
wk=2*pi/lmy

for i0 in range (0,3):
    y0=i0**2.*20
    for x in range (0,1,20):
        for iy in range (0,ny,5):
            s=sr[iy]#.mean()/2
            omega=wk*s
            v_ph=omega/wk
            Dt=-abs(y[iy]-y0)*1.e+03/v_ph-abs(x)*1.e+03/v_ph
            
            A0=A_lb(omega,t,Dt)
            
            ga=5./3.;sc_h=3.+iy/18.
            mu=-(1./ga)*(y[iy]-y0)/sc_h
            
            A_amp=exp(-mu/2.)
            
            rho=exp(-y[iy]/sc_h)
            nu=1.e-04/rho
            
            A_dis=exp(-nu*wk**2./omega)
            
            
            w=A0*A_amp*A_dis*cos(wk*y[iy]-omega*t)
            A_nl=exp(-w**2./v_ph**2.)
            
            A_aw=A_nl*w
            
            #%%
            wkx=wk
            w_bv=sqrt(ga-1)*9.8/s
            omega=w_bv*wkx/sqrt(wkx**2.+wk**2.)
            
            v_ph=omega/wk
            Dt=-abs(y[iy]-y0)*1.e+03/v_ph-abs(x)*1.e+03/v_ph
            
            A0=A_lb(omega,t,Dt)
            
            ga=5./3.;sc_h=3.+iy/18.
            mu=-(1./ga)*(y[iy]-y0)/sc_h
            
            A_amp=exp(-mu/2.)
            
            rho=exp(-y[iy]/sc_h)
            nu=1.e-04/rho
            
            A_dis=exp(-nu*wk**2./omega)
            
            
            w=A0*A_amp*A_dis*cos(wk*y[iy]-omega*t)
            A_nl=exp(-w**2./v_ph**2.)
            
            A_gw=A_nl*w    
            
            #%%
            A=A_aw+A_gw
            
            if x==0 and i0==0:
                figure(1,(12,12))
                subplot(131)
                plot(t/60,y[iy]+A_aw,'r')#,lw=5*abs(w).max())
                subplot(132)
                plot(t/60,y[iy]+A_gw,'b')
                subplot(133)
                plot(t/60,y[iy]+A_aw+A_gw,'g')
                draw()
                pause(0.1)
            if x==0:
                figure(2,(12,12))
                subplot(1,3,i0+1)
                plot(t/60,y[iy]+5*A_aw+5*A_gw,'r')
                if y[iy]==y0:plot(t/60,y[iy]+5*A_aw+5*A_gw,'g',lw=4)
                draw()
                pause(0.1)
            # if y[iy]==200:
            #     figure(3,(12,12))
            #     plot(t,x+A_aw+A_gw,'r')
            #     pause(0.1)
        