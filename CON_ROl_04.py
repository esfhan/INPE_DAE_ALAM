# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:40:29 2019

@author: saul
"""
from pylab import *
from numpy import *
from scipy.signal import *
from scipy.ndimage import *
import scipy.fftpack
from matplotlib.pyplot import *
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import matplotlib as mpl
import wavelet as wv
import glob,os

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter
import ftplib
import scipy.signal as signal
from math import degrees as deg, radians as rad  
from termcolor import colored
import math
import pandas as pd
import os

  
params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':20};matplotlib.rc('font', **font);

cc=cm.get_cmap('jet',20);cc_vals=cc(arange(20));i_tt=0;
col_map=['r','g','b','c','m','y','k','gray',(0.5,0.5,0.0),'k'];

i_plot=1
#est=['islk']
n1,n2=0,400
janela=18 #tamanho da janela em min, para setar a média corrida
est=[]
#
path='/media/alam/SaulSanchez/ELVI/SUITE04/tec/2019/185/'#caminho até a pasta dos arquivos
a=(path.title())
list_of_files=sorted(glob.glob(path+'EEU06.txt'));   
for file_name in list_of_files:                        
    f_obs=open(file_name, 'r')     
    for i_read in range (600): #número de receptores aproximado        
        a_head=f_obs.readline()
        est.append(a_head[0:4])

est=array(est);est=est[range(n1,n2)]
#est=['islk']       
i_r=0
for i in range(len(est)):
    list_of_files=sorted(glob.glob(path+est[i]+'/*G06*.dat'));#G10 bom,G18mas o menos, G20, G21 no,G27 mas omeno, R08 bom, R01 bom, (R22,R23,24)
    nr=len(est);    
    ns=len(list_of_files)
    lat_rec=zeros((nr));lon_rec=zeros((nr))
    if i_r==0:
        lon_map=zeros((ns*nr,12));lat_map=zeros((ns*nr,12));
        tec_map=zeros((ns*nr,12));t_map=zeros((12))
    i_st=0;
    for file_name in list_of_files:                    
        f_obs=open(file_name, 'r')
        i_obs=0        
        for i_read in range (0,60):         #READING HEADERS
            a_head=f_obs.readline()
            if str.find(a_head,'Sources')>0:
                sta_name=a_head[15:19]
                print (colored("NUMERO DE RECEPTOR=",'blue'),i_r,colored(",NOME DE RECEPTOR=",'blue'),sta_name.upper())
                print (colored("NUMERO DE SATELLITE=",'green'),i_st)
            if str.find(a_head,"X, Y") > 0:                 #READING RECEIVER COORDINATES
                b_head=array([ float(val.replace(',', '')) for val in  a_head.split()[5:8] ]);            
                x_r=b_head[0];y_r=b_head[1];z_r=b_head[2]; 
                lat_r=(arctan2(z_r,sqrt(x_r**2.+y_r**2.)));lon_r=(arctan2(y_r,x_r))
                lat_rec[i_r]=deg(lat_r);lon_rec[i_r]=deg(lon_r);
                print (colored("LOCACAO DE RECEPTOR=",'red'), deg(lat_r),deg(lon_r))
                if (deg(abs(lon_r))<180 and deg(abs(lat_r))>0)and deg(abs(lat_r))<90:
                    reg_select=1
                    sta_select=sta_name
                else:
                    reg_select=1
            if str.find(a_head,"Interval") > 0:
                b_head=array([ float(val) for val in a_head.split()[2:] ]);
                dt_rec=b_head[0]
                print (colored("INTERVAL=",'yellow'),dt_rec)
            if str.find(a_head,"I11") > 0: 
                break
            
        f=loadtxt(f_obs)
        t_t=f[:,1]; # hour (UT)
        elev=f[:,2]; #elevation (degree) 
        elev_ang=radians(f[:,2]); #elevation (rad) 
        az_ang=radians(f[:,3]); # azimuth  (degree)   
        tec_phase=f[:,4]; # tec (TECU)
           # stec=stec/stec.max()
        tec_pseudo=f[:,5]; # tec (TECU)
        hora1=17
        hora2=20
        pos=where((t_t>hora1) & (t_t< hora2))[0]
        nt=len(t_t[pos])
        lon_3=zeros((nt,nr));lat_3=zeros((nt,nr));
        
        pos1=where(elev[pos]<15)[0]
#        print (pos1)
        elev_ang[pos1]='nan'
        tec_phase[pos1]='nan'
        
        r_e=6.37e+06;h=300e+03;
        
        ang_p=pi/2.-elev_ang-arcsin(r_e*cos(elev_ang)/(r_e+h))
        arc_angle=sin(lat_r)*cos(ang_p)+cos(lat_r)*sin(ang_p)*cos(az_ang)
        lat_ipp=degrees(arcsin(arc_angle))# ojo degre analizar
        arc_angle=sin(ang_p)*sin(az_ang)/cos(lat_r)
        lon_ipp=degrees(lon_r+arcsin(arc_angle))# ojo degre analizar 
        
    
        lat_3=lat_ipp[pos];
        lon_3=lon_ipp[pos]; 
    
        sl_v=1.*sqrt(1.-(cos(elev_ang)*r_e/(r_e+h))**2.)
        tec_phas=tec_phase[pos]#*sl_v[pos]
        tec_3=tec_phas

        n_m=int((janela*60)/dt_rec)       
        ep_lon=-117.50;ep_lat=35.71;t_ep=17.56
       # fig = figure(3,figsize=(12,8),facecolor='w',edgecolor='k') 
        #ax=subplot(111) 
        nx=arange(0,len(t_t[pos]))
        
        lon_2=lon_3;lat_2=lat_3;
        tec_2=1*(tec_phas);
        t_2=t_t[pos];
        dtec=tec_2-convolve(tec_2,ones((n_m,))/n_m)
        
        
  ######################################      
#        fig = figure(1,figsize=(8,12),facecolor='w',edgecolor='k')
#        ax=subplot(2,4,i_st+1)
#        plot(t_2,dtec,linewidth=2)
#       ## text(2.5,0.2,str(file_name[52:56].upper()))
#        xlim(17,19)
#       # ylim(-0.3,0.3)
#        axvline(x=t_ep,color='g',lw=2)
#       # xlabel('Time,UT ');ylabel('dTEC');suptitle('$July$ $6$, $2019$')
#    i_st=i_st+1;
  ###############################################      
        
        
        nm1=40;nm2=120
        dtec=convolve(tec_2,ones((nm2,))/nm2)-convolve(tec_2,ones((nm1,))/nm1)
        
        plot(ep_lon,ep_lat,'*',markersize=18,color='b')
        
        if len(t_2)!=0:
            for i_tt in range (12):
                t_11=t_ep+(i_tt-6)*0.1
                i_ep=abs(t_2-t_11).argmin()
                lon_map[i_st+i_r*ns,i_tt]=lon_2[i_ep]
                lat_map[i_st+i_r*ns,i_tt]=lat_2[i_ep]
                tec_map[i_st+i_r*ns,i_tt]=dtec[i_ep]
                t_map[i_tt]=t_11
        
        fig = figure(1,figsize=(12,12),facecolor='w',edgecolor='k') 
        ax=subplot(121) 
        scatter(t_2,lon_2-ep_lon,c=dtec,s=20,vmax=0.25,vmin=-0.25,marker='o',cmap=cm.seismic)
        ax=subplot(122) 
        scatter(t_2,lat_2-ep_lat,c=dtec,s=20,vmax=0.25,vmin=-0.25,marker='o',cmap=cm.seismic)
#        axis((2,4.5,-3,3))
      
        i_st=i_st+1;
    i_r=i_r+1

fig = figure(2,figsize=(12,12),facecolor='w',edgecolor='k') 
for i_tt in range (12):
    subplot(3,4,i_tt+1)
    plot(ep_lon,ep_lat,'*',markersize=11,color='k')
    im=scatter(lon_map[:,i_tt],lat_map[:,i_tt],c=tec_map[:,i_tt],s=20,vmax=0.25,vmin=-0.25,
               marker='o',cmap=cm.jet,alpha=1) 
    axis('tight');axis((-125,-110,32,40));
    gca().set_xticks([]);gca().set_yticks([])
    gca().set_xticklabels([]);gca().set_yticklabels([])
    title(str(t_map[i_tt]),fontsize=8)
show()    
    #==========================================================================      