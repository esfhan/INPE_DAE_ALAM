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

pgv=5.e-01;t0=0
def A_lb(am,omega,t,t0,Dt):
    u_cos=cos(omega*(t-t0+Dt))
    sigma=2./(omega)
    u_gauss=skew(t+Dt,t0,sigma,5)
    #u_gauss=u_gauss*u_cos
    x=arange(1,4,0.5)
    for it in x:
        u_gauss=u_gauss+0.5*skew(t+Dt,t0+200*it,sigma,5)/it
    
    #u_gauss=u_gauss/u_gauss.max()
    return am*u_gauss*u_cos

import time
start = time.time()

t=arange(0,3600,60);nt=len(t)

lmy=300.e+03  #meters
wky=2*pi/lmy
wkx=wky/1.

pgv_dw=zeros((10,nt))

dx=10
xi=-500;xf=500+dx
x0=arange(xi,xf,dx);y0=x0
nx=len(x0);ny=len(y0)
A=zeros((nt,nx,ny))
w_fonte=zeros((nx,ny))

for i0 in range (0,1):
    x0=0+i0*100;y0=x0
    A_aw=0;A_dw=0
    ix=-1;
    for x in range (xi,xf,dx):
        ix=ix+1;iy=-1
        for y in range (xi,xf,dx):
            iy=iy+1
            r=sqrt((x-x0)**2.+(y-y0)**2)
            sc_h=5.e+03;
            v_ph=sqrt(9.8*sc_h)#omega/wk
            omega=v_ph*sqrt(wky**2.+wkx**2.)            
            #fault=pgv*skew(x,x0+v_ph*(t-t0)*1.e-03,ax*dx,0)*skew(y,y0+v_ph*(t-t0)*1.e-03,ay*dx,0)
            
            ilha0=skew(x,x0,4*dx,0)*skew(y,y0,4*dx,0)
            
            depth=sc_h#*abs(1-ilha0)+1
            
            ax=2;ay=5
            r=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(x-x0)**2.+ax**2.*(y-y0)**2)
            dr=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(dx)**2.+ax**2.*(dx)**2)
            
            fault=skew(r,0+5*dr+v_ph*(t-t0)*1.e-03,2*dr,0)
            fault[fault<0.5]=0            
            fonte=0*pgv+pgv*fault
            
            theta=arctan(y/(x+0.00001))
            #if theta>pi/3 and theta <2*pi/3:fonte=0
            
            w_fonte[ix,iy]=w_fonte[ix,iy]+fonte[0]
            
            it=abs(fault-fault.max()).argmin()
            Dt=-abs(r)*1.e+03/v_ph#-abs(x-x0)*1.e+03/v_ph
            A_aw=(sc_h/depth)*A_lb(fonte[it],omega,t,t0+t[it],Dt)
            
#%%
            A_dw=0;ilha1=0
            for i_ilha in range (1,4,2):
                x01=i_ilha*20*dx;y01=20*dx
                r01=sqrt((x01)**2.+(y01)**2)
                t01=r01*1.e+03/v_ph
                #ilha1=skew(x,x0+x01-v_ph*(t-t0-t01)*1.e-03,ax*dx,0)*skew(y,y0+y01-v_ph*(t-t0-t01)*1.e-03,ax*dx,0)
                ilha1=ilha1+skew(x,x0+x01,4*dx,0)*skew(y,y0+y01,4*dx,0)
                
                it=abs(t-t01).argmin()
                r=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(x-x0-x01)**2.+ax**2.*(y-y0-y01)**2)
                Dt=-abs(r)*1.e+03/v_ph
                A_dw=A_dw+A_lb(A_aw[it],omega,t,t0+t01,Dt)
            
#%%            
            A_tot=A_aw+A_dw
            
            A_tot[abs(ilha0)>1/2]=0
            A_tot[abs(ilha1)>1/5]=0
            
            ax=4;ay=8
            r=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(x-x0)**2.+ax**2.*(y-y0)**2)
            wk=sqrt(wkx**2.+wky**2.)
            A[:,ix,iy]=A[:,ix,iy]+A_tot#*cos(wk*r*1.e+03)
            
            #%%
            
            wy=A;wx=(wkx/wky)*wy
            
            #%%
x=arange(xi,xf,dx);y=x
figure(1,(12,12))
for it in range (0,nt):
    pcolormesh(x,y,A[it,:,:].T,cmap=cm.seismic,vmax=pgv,vmin=-pgv)
    contour(x,y,w_fonte[:,:].T,21)
    pause(0.5)

#figure(2,(12,12))
#pcolormesh(t,x,A[:,:,0].T)

import time
end = time.time()
print(end - start)