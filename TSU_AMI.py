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
from scipy.ndimage import *

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
    u_gauss=skew(t+Dt,t0,sigma,0)
    #u_gauss=u_gauss*u_cos
    x=arange(1,12*8,1)
    for it in x:
        u_gauss=u_gauss+0.5*skew(t,t0,2*3600,0)*skew(t+Dt,t0+300*it,sigma,0)
    
    #u_gauss=u_gauss/u_gauss.max()
    return am*u_gauss*u_cos

import time
start = time.time()

t=arange(0,6*3600,5*60);nt=len(t)

lmy=300.e+03  #meters
wky=2*pi/lmy
wkx=wky/1.
wk=sqrt(wkx**2.+wky**2.)

pgv_dw=zeros((10,nt))


#%%
a=imread('color_etopo1_ice_low.jpg')

b=a[:,:,1];

b=zoom(b,3);b=b[::-3,::3];b=zoom(b[:,::5],(1,5))
ny=len(b[:,0])
nx=len(b[0,:])
x=linspace(-180,180,nx);dx=x[1]-x[0]		#x-y in degrees
y=linspace(-90,90,ny);dy=y[1]-y[0]

nroll=int(120./dx)
b=roll(b,-nroll,axis=1)

xep=143-120;yep=39
xr=arange(xep-5,xep+55,dx);nx=len(xr)
yr=arange(yep-30,yep+20,dy);ny=len(yr)

ix0=abs(x-xr[0]).argmin();ix1=abs(x-xr[-1]).argmin()
iy0=abs(y-yr[0]).argmin();iy1=abs(y-yr[-1]).argmin()

ilha_peak=argwhere(b[iy0:iy1,ix0:ix1]>210)
print (len(ilha_peak))

arc_180=180+abs(y[iy1]-y[iy0])
arc=6371*sin(1*pi/arc_180)
dx=int(dx*arc)
dy=int(dy*arc)

#dx0=dx;dy0=dy
A=zeros((nt,nx,ny))
w_fonte=zeros((nx,ny))


for i0 in range (0,1):
    x0=0+i0*100;y0=-500		#x-y in cartesian
    A_aw=0;A_dw=0
    for iy in range (0,ny):
    	y=(yr[iy]-yep)*arc
    	for ix in range (0,nx):
            x=(xr[ix]-xep)*arc
            #fault=pgv*skew(x,x0+v_ph*(t-t0)*1.e-03,ax*dx,0)*skew(y,y0+v_ph*(t-t0)*1.e-03,ay*dx,0)
            
            ilha0=b[iy0+iy,ix0+ix]#skew(x,x0,4*dx,0)*skew(y,y0,4*dx,0)
            
            sc_h=5.e+03;
            depth=sc_h*abs(1-ilha0/220)/2.+sc_h
            amp=sc_h/(depth)
            
            v_ph=sqrt(9.8*depth)
            omega=v_ph*wk        
            A_aw=0
            for i_fault in range (0,4):
                ax=1;ay=2
                r=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(x-x0-i_fault*1*dx)**2.+ax**2.*(y-y0-i_fault*4*dy)**2)
                dr=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(dx)**2.+ax**2.*(dy)**2)
                
                t01=abs(r)*1.e+03/v_ph
                fault=skew(r,0+2*dr+v_ph*(t-t0-t01)*1.e-03,dr,0)
                fault[fault<0.5]=0
                
                #d=4*dr
                #fault=heaviside(r-v_ph*(t-t0-t01)*1.e-03+d/2,0)-heaviside(r-v_ph*(t-t0-t01)*1.e-03-d/2,0)
                
                fonte=0*pgv+pgv*fault
                
                #theta=arctan(y/(x+0.01))
                #if theta>pi/3 and theta <2*pi/3:fonte=0
                
                w_fonte[ix,iy]=w_fonte[ix,iy]+fonte[0]
                
                it=abs(fault-fault.max()).argmin()
                Dt=-abs(r)*1.e+03/v_ph#-abs(x-x0)*1.e+03/v_ph
                A_aw=A_aw+A_lb(fonte[it],omega,t,t0+t[it],Dt)#*cos(wk*r*1.e+03)
                #it=abs(A_awp-A_awp.max()).argmin()
                #A_aw=A_aw+A_lb(A_awp[it],omega,t,t0+t[it],Dt)*cos(wk*r*1.e+03)
                
#%%
            A_dw=0;ilha1=0
            # for i_ilha in range (1,4,2):
            #     x01=i_ilha*20*dx;y01=20*dx
            #     r01=sqrt((x01-x0)**2.+(y01-y0)**2)
            #     t01=r01*1.e+03/v_ph
            #     #ilha1=skew(x,x0+x01-v_ph*(t-t0-t01)*1.e-03,ax*dx,0)*skew(y,y0+y01-v_ph*(t-t0-t01)*1.e-03,ax*dx,0)
            #     ax=4;ay=4
            #     ilha1=ilha1+skew(x,x0+x01,ax*dx,0)*skew(y,y0+y01,ay*dx,0)
                
            #     it=abs(t-t01).argmin()
            #     rdw=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(x-x0-x01)**2.+ax**2.*(y-y0-y01)**2)
            #     Dt=-abs(rdw)*1.e+03/v_ph
            #     A_dw=A_dw+A_lb(A_aw[it],omega,t,t0+t01,Dt)*cos(wk*rdw*1.e+03)

#%%
            for i_peak in range (0):#(0,len(ilha_peak),10):
                #x01=x;y01=y
                y01=(yr[ilha_peak[i_peak][0]]-yep)*110;x01=(xr[ilha_peak[i_peak][1]]-xep)*110
                r01=sqrt((x01-x0)**2.+(y01-y0)**2)
                t01=r01*1.e+03/v_ph
                ax=1;ay=1
                it=abs(t-t01).argmin()
                rdw=(1/sqrt(ax**2+ay**2))*sqrt(ay**2*(x-x0-x01)**2.+ax**2.*(y-y0-y01)**2)
                Dt=-abs(rdw)*1.e+03/v_ph
                A_dw=A_dw+A_lb(A_aw[it],omega,t,t0+t01,Dt)*cos(wk*rdw*1.e+03)

#%%            
            A_tot=amp*A_aw+A_dw
            
            A_tot[abs(ilha0)>150]=0
            #A_tot[abs(ilha1)>1/5]=0
            
            A[:,ix,iy]=A[:,ix,iy]+A_tot
            
            #%%
            
            wy=A;wx=(wkx/wky)*wy
            
            #%%

figure(1,(12,12))
for ilat in range (0,len(yr),2):
    plot(xr+120,yr[ilat]+b[iy0+ilat,ix0:ix1+1]/100,'g')
for it in range (0,nt,5):
    pcolormesh(zoom(xr,2)+120,zoom(yr,2),zoom(A[it,:,:].T,2,mode='nearest'),cmap=cm.seismic,vmax=pgv,vmin=-pgv)
    contour(xr+120,yr,w_fonte[:,:].T,21)
    #axis((130,210,-30,70))
    pause(0.5)

import time
end = time.time()
print(end - start)
