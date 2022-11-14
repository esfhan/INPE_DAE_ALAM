from pylab import *
from numpy import *
from scipy import signal
from scipy.ndimage import *
import scipy.fftpack
from matplotlib.pyplot import *
#from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.wavelet import *
from scipy import fftpack
import glob,os
import pandas as pd
from matplotlib.signal_alam import *
from matplotlib.patches import Ellipse

matplotlib.rc("mathtext",fontset="cm")        #computer modern font 
matplotlib.rc("font",family="serif",size=12)

#%%
        
def axis_param(im,vm,nm,str_c):
    divider = make_axes_locatable(ax);
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar=colorbar(im,cax=cax);gca().set_title(str_c,fontsize=11) 
    cbar.set_ticks(linspace(-vm,vm,3))
    vm_n=around(vm*nm,decimals=2)
    cbar.set_ticklabels(linspace(-vm_n,vm_n,3)) 
    return()

#%%
path='/home/alam/INPE/SAUL/EQ_RIDGCREST/ANA_DADOS/TEC/02/'#caminho até a pasta dos arquivos
a=(path.title())

f_d=['*04.npy','*05.npy','*06.npy']
t_ep=[17.55,11.1,3.3]
f_d=['*02.npy','*04.npy','*03.npy','*05.npy','*06.npy']
t_ep=17.55
lat_ep=35.7;lon_ep=-117.6
lon_sism=-117.36453;lat_sism=35.52495
lon_mag=240.28-360.;lat_mag=37.09

dt=2;n_size=2;vm=0.1;nt_s=int(2*(dt)*240)

t_nearest=[];tec_nearest=[];dist_ep=[];sta_name=[];
ep_lat=[];ep_lon=[]

for i_d in range (0,1):
    l_of=sorted(glob.glob(path+'*'+f_d[i_d]));   
    i_r=0
    for f_n in l_of:
        data=load(f_n)
        t=data[0,:];lat=data[1,:];lon=data[2,:];tec=data[3,:];
        tec=40+tec-tec.mean()   

        #%%
        fig = figure(1,figsize=(12,12),facecolor='w',edgecolor='k')
        ax=subplot(111)
        im=scatter(lon[::5],lat[::5],c=t[::5],s=0.25,
                marker='o',cmap=cm.jet,vmax=20,vmin=14,alpha=1.)
        axis((-125,-100,24,40))
        if f_n==l_of[-1]:
            plot(lon_ep,lat_ep,'*',markersize=9,color='k')
            plot(lon_sism,lat_sism,'s',markersize=6,color='k')
#            plot(lon_mag,lat_mag,'o',markersize=6,color='k')
#                map=Basemap(llcrnrlon=lon[0]-5,llcrnrlat=lat[0]-2,urcrnrlon=lon[-1]+5,
#                            urcrnrlat=lat[-1]+7,suppress_ticks=False)
#                map.etopo(alpha=0.5)
            xlabel('Longitude, $^o$');ylabel('Latitude, $^o$')
            title('Trajectories of PRN=19 from several GNSS receivers')#,x=0.1,y=0.9)
            divider=make_axes_locatable(ax);cax=divider.append_axes("right",size="2%",pad=0.05)
            colorbar(im,cax=cax);gca().set_title('Time, UT',fontsize=10)
            
        #%%
        n_smooth=int(16./((t[1]-t[0])*60.))
        win=ones((n_smooth,))/n_smooth
        dtec=tec-convolve_al(tec,win)
        dtec=convolve_al(dtec,ones((7,))/7)
#        f=wavelet(0,t*60.,tec,32)    
#        pd=f[0];pwr=f[1];emd=f[2];
#        fr=1000./(pd*60.);nd=len(fr)
#        fr_s=1.35;i_o=abs(fr-fr_s).argmin()
#        dtec=emd[i_o,:];
        
        i_o=abs(t-t_ep).argmin()
        ep_dist=(lat[i_o]-lat_ep)
#        if str.find(f_n,'islk')>0: #bvpp,bepk,bemt
#            t_bepk=t;tec_bepk=tec
#            print ('FIND THE STATION',ep_dist)
            
        i_nearest=0
        if abs(lat[i_o]-lat_ep)<10.5 and abs(lon[i_o]-lon_ep)<10.5 \
        and abs(gradient(tec)).max() <=0.1*abs(tec).mean():
            t1=14.5;t2=19.75
            i1=argwhere(t==t1);i2=argwhere(t==t2) 
            i_nearest=1
            if len(i1)==1 and len(i2)==1:
#                print ('Nearest station',f_n[-11:-7],ep_dist,abs(tec.max()))
                i_1=i1[0][0];i_2=i2[0][0]
                if len(t[i_1:i_2])!=int((t[i_2]-t[i_1])*3600./15.):
                    continue
                t_nearest.append(t[i_1:i_2])
                tec_nearest.append(tec[i_1:i_2])
                dist_ep.append(ep_dist)
                ep_lat.append(lat[i_1:i_2]-lat_ep)
                ep_lon.append(lon[i_1:i_2]-lon_ep)
                sta_name.append(f_n[-11:-7])
#                t_nearest.append(t[100:800])
#                tec_nearest.append(tec[100:800])
#                dist_ep.append(ep_dist)
#                ep_lat.append(lat[100:800]-lat_ep)
#                ep_lon.append(lon[100:800]-lon_ep)
#                sta_name.append(f_n[-11:-7])                    
             #%%
#                fig = figure(4,figsize=(12,12),facecolor='w',edgecolor='k')
#                ax=subplot(1,2,i_d+1)
#                vm=0.15
#                plot(lon[i_o]-lon_ep,lat[i_o]-lat_ep,'o',markersize=4)
#                im=scatter(lon-lon_ep,2.*dtec+lat-lat_ep,c=dtec,s=5*abs(dtec),
#                        marker='o',cmap=cm.seismic,vmax=vm,vmin=-vm,alpha=1.)
#                axis((-1.5,1.5,-1.5,1.5))
#                if i_nearest==1:
#                    xlabel('Epicentral Longitude, $^o$');
#                    if i_d==0:
#                        title('(A) TIDs on event day')
#                        ylabel('Epicentral Latitude, $^o$')
#                    if i_d==1:
#                        title('(B) TIDs on previous day')
#                        f=axis_param(im,vm,1,'TECU')
            
        #%%
#                fig = figure(5,figsize=(12,12),facecolor='w',edgecolor='k') 
#                ax=subplot(1,1,i_d+1) 
#        #        
#                ellipse = Ellipse(xy=(0.4, -0.4), width=0.3, height=1.5,angle = 10, 
#                                edgecolor='g', fc='None', lw=0.1)
#                ax.add_patch(ellipse)
#                ellipse = Ellipse(xy=(1.05, -0.1), width=0.3, height=2,angle = 25, 
#                                edgecolor='g', fc='None', lw=0.1)
#                ax.add_patch(ellipse)
#                ellipse = Ellipse(xy=(1.85, 1), width=0.3, height=2,angle = -10, 
#                                edgecolor='g', fc='None', lw=0.1)
#                ax.add_patch(ellipse)
#                ellipse = Ellipse(xy=(1.6, -0.5), width=0.3, height=1.5,angle = 2, 
#                                edgecolor='g', fc='None', lw=0.1)
#                ax.add_patch(ellipse)
#        #        
#                dtec[abs(dtec)>1]=0;dtec[abs(dtec)<0.05]=0;
#        
#                if f_n.find('p2') ==-1:
#                    vm=0.15
#                    im=scatter(t-t_ep,2*dtec+(lat-lat_ep),c=dtec,s=0.5*abs(dtec),
#                               vmax=vm,vmin=-vm,marker='o',cmap=cm.seismic,alpha=1)
        #            im=scatter(t-t_ep,1*dtec+(0.025+lat-lat_ep),c=dtec,s=0.5*(dtec),
        #                       vmax=vm,vmin=-vm,marker='o',cmap=cm.seismic,alpha=1)
        #            im=scatter(t-t_ep,1*dtec+(-0.025+lat-lat_ep),c=dtec,s=0.5*(dtec),
        #                       vmax=vm,vmin=-vm,marker='o',cmap=cm.seismic,alpha=1)
        #            plot(t-t_ep,2*dtec+lat-lat_ep,linewidth=0.2)
        #            vm=3
        #            im=scatter(t-t_ep,0.1*dtec+(lat-lat_ep),c=dtec,s=0.015*abs(dtec),
        #                       vmax=vm,vmin=-vm,marker='o',cmap=cm.seismic,alpha=1)
#                    axis((-1.5,1.5,-3,4))
#                if f_n==l_of[-1]:
#                    plot([0,0],[-4,4],'--',color='gray')
#                    plot([(73./60.),(73./60.)],[-4,4],'--',color='gray')
#                    xlabel('Time from the Earthquake onset, hours')
#                    ylabel('Epicentral Distance, $^o$')
#                    if i_d==0:
#                        title('TIDs on event day')
#                    if i_d==1:
#                        title('TIDs on previous day')
#                    f=axis_param(im,vm,1,'TECU')
        
        #%%
        i_r=i_r+1
            
save('t_nearest_02.npy',array(t_nearest))
save('tec_nearest_02.npy',array(tec_nearest))   
save('dist_ep_02.npy',array(dist_ep))   
save('ep_lat_02.npy',array(ep_lat))   
save('ep_lon_02.npy',array(ep_lon))   
save('sta_name_02.npy',array(sta_name))   
#   
#%%
#t_nearest=load('t_nearest.npy')
#tec_nearest=load('tec_nearest.npy')
#dist_ep=load('dist_ep.npy')
#sta_name=load('sta_name.npy')
#print(min(abs(dist_ep)))
#print (len(t_nearest[:]))
#for i in range (len(t_nearest[:])):
#    t=t_nearest[i][:]
#    tec=tec_nearest[i][:];
#    fig = figure(3,figsize=(12,12),facecolor='w',edgecolor='k')
#    subplot(111)    
#    plot(t,tec)
#fig = figure(3,figsize=(12,12),facecolor='w',edgecolor='k')
#i_o=abs(abs(dist_ep)-min(abs(dist_ep))).argmin()
#print (sta_name[i_o])
#plot(t_nearest[i_o][:],tec_nearest[i_o][:])
    

