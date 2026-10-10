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
    am=1.e-00;t0=200
    u_cos=am*cos(omega*(t+Dt))
    sigma=2./omega
    u_gauss=skew(t+Dt,t0,sigma,5)
    u_gauss=u_gauss/u_gauss.max()
    return am*u_gauss

t=arange(0,20000,1);nt=len(t)
dy=2.
y=arange(0,4000,10);ny=len(y)
dym=dy*1.e+03

sr=300*skew(y,0,5,0)+3000*skew(y,400,200,0)+50000*skew(y,1500,10000,25)
figure (11,(12,12))
plot(sr,y)
ylabel('Height from the solar interior, km')
xlabel('Sound speed, m/s')

lmy=1.e+03*(y[-1]-y[0])/6.  #meters
wk=2*pi/lmy

for i0 in range (0,1):
    y0=i0**2.*20
    for x in range (0,1,20):
        for iy in range (0,ny,10):
            s=sr[iy]#10*sr.mean()/2
            omega=wk*s
            v_ph=omega/wk
            Dt=-abs(y[iy]-y0)*1.e+03/v_ph-abs(x)*1.e+03/v_ph
            
            A0=A_lb(omega,t,Dt)
            
            ga=5./3.;sc_h=100.+0*iy/18.
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
            w_bv=sqrt(ga-1)*9.8*30/s
            omega=w_bv*wkx/sqrt(wkx**2.+wk**2.)
            
            v_ph=omega/wk
            Dt=-abs(y[iy]-y0)*1.e+03/v_ph-abs(x)*1.e+03/v_ph
            
            A0=A_lb(omega,t,Dt)
            
            A_amp=exp(-mu/2.)
            
            A_dis=exp(-nu*wk**2./omega)
            
            
            w=A0*A_amp*A_dis*cos(wk*y[iy]-omega*t)
            A_nl=exp(-1*w**2./v_ph**2.)
            
            A_gw=A_nl*w    
            
            #%%
            A=A_aw+A_gw
            
            print (A_aw.max(),s)
            if x==0 and i0==0:
                figure(1,(12,12))
                plot(t/3600,y[iy]+(A_aw+A_gw)/1,'r')
                draw()
                pause(0.1)
hlines(500,t[0]/3600,t[-1]/3600,'g',lw=4)
hlines(2000,t[0]/3600,t[-1]/3600,'g',lw=4)
xlabel('Hours')           
ylabel('Height from the solar interior, km')       