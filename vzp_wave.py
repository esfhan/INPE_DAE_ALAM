#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:16:33 2026

@author: alam
"""
import pywt
from numpy import *
from pylab import *
from scipy import *
from scipy.special import erf

matplotlib.rc("mathtext",fontset="cm")        #computer modern font 
matplotlib.rc("font",family="serif",size=15)

close('all')

def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)


data=loadtxt('Vzp.dat')
temp=data[:,0];wind=data[:,1]
time=temp
figure(1,(12,12))
subplot(211)
plot(time,wind)
plot(time,temp)
xlim(0,90)

subplot(212)
t = time#np.linspace(-1, 1, 200, endpoint=False)
sig  = wind/wind.max()
widths = logspace(log10(1), log10(20), 20)
cwtmatr, freqs = pywt.cwt(sig, widths, 'morl')
contourf(time,1./freqs,cwtmatr,31, cmap='seismic', shading='gouraud', 
          vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
ylim(1,20)
plot(time[0]+1./freqs,1./freqs,'k--')
plot(time[-1]-1./freqs,1./freqs,'k--')
xlim(0,90)



wind=wind/wind.max()
figure(3,(12,12))
dwind=[];prd=[]
d=0
for imode in range (int(len(time)/2),1,-1):
    d1=convolve(wind-d,ones((imode,))/imode,mode='same')
    d=d+d1
    dt=1
    pd=imode*dt
    plot(time[imode:-imode],pd+5*(d1)[imode:-imode],'b')
    prd.append(pd)
    dwind.append(d1)
plot(time,wind,'k')
xlim(0,90)
ylim(1,20)

# contourf(time,1./freqs,cwtmatr,31, cmap='seismic', shading='gouraud', 
#           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
for imode in range (0,20):
    plot(time,1./freqs[imode]+cwtmatr[imode,:],'r')

plot(time[0]+1./freqs,1./freqs,'k--')
plot(time[-1]-1./freqs,1./freqs,'k--')
xlim(0,90)
ylim(1,20)


figure(4,(12,12))
imshow(dwind,origin='upper',cmap=cm.seismic,vmax=0.1,vmin=-0.1,interpolation='bicubic')
#ylim(0,20)
axis('tight')
plot(time,20*wind,'k')

figure(21,(12,12))
data=loadtxt('Vzp.dat')
time=data[:,0];vzp=data[:,1]

#plot(time,vzp,'g-o')
data=loadtxt('Vzp_quiet.dat')
time=data[:,0];vzp=data[:,1]

subplot(211)
plot(time,vzp,'o',color='grey')
xlim(0,90)
ylabel('m/s')
title('(A) Peak vertical drift, V$_{zp}$')
vlines(20,0,100)
vlines(40,0,100)
vlines(70,0,100)

vm_1=vzp[logical_and(time>=3,time<=20)].mean()
vm_2=vzp[logical_and(time>20,time<=40)].mean()
vm_3=vzp[logical_and(time>40,time<=70)].mean()
vm_4=vzp[logical_and(time>70,time<=90)].mean()

hlines(vm_1,0,20,'r')
hlines(vm_2,20,40,'r')
hlines(vm_3,40,70,'r')
hlines(vm_4,70,90,'r')


v0=vzp
v_all=vzp

figure(21,(12,12))
subplot(212)
vlines(20,0,100)
vlines(40,0,100)
vlines(70,0,100)

from scipy.special import j0

for i_mode in range (0,int(len(time)/2)):
# 	t1=time[logical_and(time>=3,time<=20)]
# 	t2=time[logical_and(time>20,time<=40)]
# 	t3=time[logical_and(time>40,time<=70)]
# 	t4=time[logical_and(time>70,time<=90)]
# 	v1=v0[logical_and(time>=3,time<=20)]
# 	v2=v0[logical_and(time>20,time<40)]
# 	v3=v0[logical_and(time>40,time<70)]
# 	v4=v0[logical_and(time>70,time<90)]
# 	if v1.max()!=0:vm_1mn=v1[v1!=0].min();t1mn=time[where(v0==vm_1mn)][0]
# 	if v2.max()!=0:vm_2mn=v2[v2!=0].min();t2mn=time[where(v0==vm_2mn)][0]
# 	if v3.max()!=0:vm_3mn=v3[v3!=0].min();t3mn=time[where(v0==vm_3mn)][0]
# 	if v4.max()!=0:vm_4mn=v4[v4!=0].min();t4mn=time[where(v0==vm_4mn)][0]
# 
# 	if v1.max()!=0:vm_1mx=v1[v1!=0].max();t1mx=time[where(v0==vm_1mx)][0]
# 	if v2.max()!=0:vm_2mx=v2[v2!=0].max();t2mx=time[where(v0==vm_2mx)][0]
# 	if v3.max()!=0:vm_3mx=v3[v3!=0].max();t3mx=time[where(v0==vm_3mx)][0]
# 	if v4.max()!=0:vm_4mx=v4[v4!=0].max();t4mx=time[where(v0==vm_4mx)][0]
# 	
# 	t1x=(t1mn+t1mx)/2;t2x=(t2mn+t2mx)/2.;t3x=(t3mn+t3mx)/2.;t4x=(t4mn+t4mx)/2.
# 	tp1=2*abs(t1mn-t1mx);tp2=2*abs(t2mn-t2mx);tp3=2*abs(t3mn-t3mx);tp4=2*abs(t4mn-t4mx)
# 	amp1=abs(vm_1mn-vm_1mx);amp2=abs(vm_2mn-vm_2mx);amp3=abs(vm_3mn-vm_3mx);amp4=abs(vm_4mn-vm_4mx)
# 
# 	time0=arange(1,91)
# 	
# 	w1=amp1*cos(2*pi*(time0-t1mx)/(tp1+0.1))*skew(time0,t1mx,tp1/2,0)
# 	w2=amp2*cos(2*pi*(time0-t2mx)/(tp2+0.1))*skew(time0,t2mx,tp2/2,0)
# 	w3=amp3*cos(2*pi*(time0-t3mx)/(tp3+0.1))*skew(time0,t3mx,tp3/2,0)
# 	w4=amp4*cos(2*pi*(time0-t4mx)/(tp4+0.1))*skew(time0,t4mx,tp4/2,0)
# 	
# 	
# 	print (i_mode,'Longest periods=',tp1,tp2,tp3,tp4,'days')
# 	print ('Amplitudes=',amp1,amp2,amp3,amp4,'m/s')
# 
# 	v0[v0==vm_1mn]=0;v0[v0==vm_2mn]=0;v0[v0==vm_3mn]=0;v0[v0==vm_4mn]=0
# 	v0[v0==vm_1mx]=0;v0[v0==vm_2mx]=0;v0[v0==vm_3mx]=0;v0[v0==vm_4mx]=0
# 	print (i_mode,v0)
# 	
# 	figure(21,(12,12))
# 	subplot(212)
# 	plot(t1mx,tp1,'go',markersize=int(amp1)/2)
# 	plot(t2mx,tp2,'bo',markersize=int(amp2)/2)
# 	plot(t3mx,tp3,'ro',markersize=int(amp3)/2)
# 	plot(t4mx,tp4,'ko',markersize=int(amp4)/2)
# 	ylim(0,25)
# 	xlim(0,90)
# 	xlabel('Days from 1st of January 2024')
# 	ylabel('Periods, days')
# 	title('(B) Hilbert-Huang spectrum')
# 	savefig('figure_hht.png')
# 	figure(31,(12,12))
# 	subplot(212)
# 	plot(time0,tp1+w1/10,'grey')
# 	plot(time0,tp2+w2/10,'grey')
# 	plot(time0,tp3+w3/10,'grey')
# 	plot(time0,tp4+w4/10,'grey')
	time0=arange(1,91)
	
	if v_all.max()!=0:
		vm_mn=v_all[v_all!=0].min();tmn=time[where(v_all==vm_mn)][0]
		vm_mx=v_all[v_all!=0].max();tmx=time[where(v_all==vm_mx)][0]
		tx=(tmn+tmx)/2.;tp=abs(tmn-tmx);amp=abs(vm_mn-vm_mx);
		wave=amp*cos(2*pi*(time0-tmx)/(tp+0.1))*skew(time0,tmx,tp,0)
		v_all[v_all==vm_mn]=0;v_all[v_all==vm_mx]=0;
		if tp>=(time[-1]-time[0])/2:continue
		print(i_mode,v_all)
		figure(41,(12,12))
		subplot(211)
		plot(time,i_mode+v_all/v_all.max(),'bo')
		subplot(212)
		plot(time0,tp+wave/20)
		#ylim(0,25)
		plot(tmx,tp,'o')#,markersize=int(amp+2))
	
	if v_all.max()==0:
		break
	draw()
	pause(0.5)
	if i_mode==0:vlines(20,0,40);vlines(40,0,40);vlines(70,0,40)