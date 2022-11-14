from pylab import *
#from pywt import *
from numpy import *
from numpy.core.defchararray import join
from scipy.signal import *
from scipy.ndimage import *
from matplotlib.pyplot import *
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import glob
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import spline
import scipy.interpolate as inter
import scipy.fftpack
import os
from math import degrees as deg, radians as rad  
import wavelet as wv
from termcolor import colored
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import math

params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':14};matplotlib.rc('font', **font);

file_path=raw_input(colored('FILE PATH ','red'))#"/home/fatima/GPS/CHAIN/"

#==============================================================================

def runningMean(x,n1,n2):
    y = zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = sum(x[ctr:(ctr+n1)])/n1-sum(x[ctr:(ctr+n2)])/n2
    return y

def runningavg(x,n):
    x_av=0*x
    for ctr in range(0,len(x)-n,n):
         nx=arange(ctr,ctr+n);
         x_av[nx]=x[nx]-x[nx].mean()
    return x_av

def filt_data(delta,t_l,t_u,x):
    dt=delta;f_n=1/(2.*dt);f_l=1./(t_u);f_h=1./(t_l);wl=f_l/f_n;wh=f_h/f_n;
    [b,a]=bessel(2,[wl,wh],btype='bandpass'); 
    delta_data=transpose(lfilter(b,a,x));
    return delta_data    
    
def wavelet(t_data,data):
    dt_wv=(t_data[2]-t_data[1])
    maxscale=2
    notes=32
    scaling="log" #or "linear"
    cw=wv.MorletReal(data,maxscale,notes,scaling=scaling)
    scales=cw.getscales()     
    cwt=cw.getdata()
    pwr=cw.getpower();
    pwr=data.max()*pwr/pwr.max();
    scalespec=np.sum(pwr,axis=1)/scales # calculate scale spectrum
    dt_freq=cw.fourierwl*scales*dt_wv
    vm=pwr.max()/1.
    return (dt_freq,pwr)
    
#==============================================================================
#==============================================================================

#params={'axes.labelweight':'bold','font.color':'red'}
#rcParams.update(params)
#font = {'family' : 'serif','weight':'normal','size': 14};
#matplotlib.rc('font', **font);
#
cc=cm.get_cmap('jet',20);cc_vals=cc(arange(20));i_tt=0;
col_map=['r','g','b','c','m','y','k','gray',(0.5,0.5,0.0),'k'];

#==============================================================================
#===============================POR DE SOL ====================================

def por_de_sol (i_r,nr,ano,mes,dia,lat_r,lon_r):
    j_a=int((14-int(mes))/12);j_y=2000+int(ano)+4800-j_a;j_m=int(mes)+12*j_a-3;
    j_d=int(dia)+int((153*j_m+2)/5.)+365*j_y+int(j_y/4.)-int(j_y/100.)+int(j_y/400.)-32045.
    n=int(j_d-2451545+0.0008);   #julian day
    j_star=-lon_r/360.+n; #mean solar noon, west +ve
    m=(357.5291+0.98560028*j_star)  #solar mean anomaly
    m=m-int(m/360.)*360.
    c=1.9148*sin(rad(m))+0.02*sin(2.*rad(m))+0.0003*sin(3.*rad(m)) #equation of the center
    lambda_e=(m+c+180.+102.9372)    #eliptic longitude
    lambda_e=lambda_e-int(lambda_e/360.)*360.
    j_trans=2451545.+j_star+0.0053*sin(rad(m))-0.0069*sin(2.*rad(lambda_e))
    delta=arcsin(sin(rad(lambda_e))*sin(rad(23.44)))
    a=sin(-rad(0.83))-sin(rad(lat_r))*sin(delta);
    b=cos(rad(lat_r))*cos(delta)
    omega_o=deg(arccos(a/b))
    j_h=12+(j_trans-int(j_trans))*24
    t_rise=j_h-omega_o/15.;t_set=j_h+omega_o/15.;
    if t_rise > 24:t_rise=t_rise-24;
    if t_set > 24:t_set=t_set-24;
    result=[];
    result.append(delta)
    return (delta,t_rise,t_set)

#=======================CALCULATING GPS TRAJECTORIES===========================
ns=31;nh=24;nt=2*120*nh+1;
dt_nav=float(raw_input('TIME RESOLUTION '))
def gps_traj(ns,nh,nt):
    data=zeros((ns,29));prn=zeros((ns));
    x=zeros((ns,nt));y=zeros((ns,nt));z=zeros((ns,nt));t=zeros((nt))
    s=[file_path,'nav.dat']
    dir_nav=str.join("",s)
    f=open(dir_nav,'r')
    for i_read in range (0,13):         #READING HEADERS
        a=f.readline()
    
    for i_s in range(0,ns):
        ij=0
        for i in range (0,8):
            a=f.readline()#[0:81]
            b=array([ float(val) for val in a.split()[0:] ])
            if i==0: 
                prn[i_s]=b[0];
                for ii in range (7,10):
                    data[i_s,ij]=b[ii];ij=ij+1;
            if logical_and (i > 0, i < 7) : 
                for ii in range (0,4):
                    data[i_s,ij]=b[ii];ij=ij+1; 
            if i == 7: 
                for ii in range (0,2):
                    data[i_s,ij]=b[ii];ij=ij+1; 
        param_nav_1=data[i_s,0] #SV clock bias (SV)
        param_nav_2=data[i_s,1] #SV clock drift
        param_nav_3=data[i_s,2] #SV clock drift rate
        param_nav_4=data[i_s,3] #Issue of data Ephimeris (IODE)
        param_nav_5=data[i_s,4] #Radius correction sinus (Crs)
        param_nav_6=data[i_s,5] #Delta n
        param_nav_7=data[i_s,6] #Mo angle (Mo)
        param_nav_8=data[i_s,7] #Latitude correction cosinus (Cuc)
        param_nav_9=data[i_s,8] #Eccentricity
        param_nav_10=data[i_s,9] #Latitude correction sinus (Cus)
        param_nav_11=data[i_s,10] #Square of semi major axis (sqrt a)
        param_nav_12=data[i_s,11] #Time of Ephimeris  (ToE)  
        
        param_nav_13=data[i_s,12] #Inclination correction cosinus (Cic)
        param_nav_14=data[i_s,13] #Omega angle (OMEGA)
        param_nav_15=data[i_s,14] #Angular velocity (CIS)
        param_nav_16=data[i_s,15] #Initial inclination (Io)
        param_nav_17=data[i_s,16] #Radius correction cosinus (Crc)
        param_nav_18=data[i_s,17] #Omega angle (omega)
        param_nav_19=data[i_s,18] #Angular velocity (OMEGA dot)
        param_nav_20=data[i_s,19] #Inclination rate (IDOT)
        param_nav_21=data[i_s,20] #L2 codes channel
        param_nav_22=data[i_s,21] #GPS week
        param_nav_23=data[i_s,22] #L2 P data flag
        param_nav_24=data[i_s,23] #SV accuracy
        
        param_nav_25=data[i_s,24] #SV health
        param_nav_26=data[i_s,25] #Total group delay (TGD)
        param_nav_27=data[i_s,26] #Issue of data clock (IODC)
        param_nav_28=data[i_s,27] #Transmission time (TT)
        param_nav_29=data[i_s,28] #Fit interval
        
        
        for i_t in range (0,nt):
            t_oe=param_nav_12;
            t[i_t]=t_oe+i_t*dt_nav;
            a_s=(param_nav_11)**2.;e_s=param_nav_9;
            m_o=param_nav_7;delta_n=param_nav_6;
            m=m_o+(sqrt(3.98e+14/a_s**3.)+delta_n)*(t[i_t]-t_oe)
            cap_e=m+(e_s-e_s**3./8+e_s**5./192-e_s**7./9216.)*sin(m)
            
            angle_nu=arctan2(sqrt(1-e_s**2.)*sin(cap_e),(cos(cap_e)-e_s));#nu
            angle_inc=param_nav_16+param_nav_20*(t[i_t]-t_oe);#i
            angle_lat=param_nav_18 #omega
            angle_long=param_nav_14+param_nav_19*(t[i_t]-t_oe)-7.79e-05*t[i_t];#OMEGA
        
            c_rs=param_nav_5; c_rc=param_nav_17;   
            c_ls=param_nav_10; c_lc=param_nav_9;
            c_is=param_nav_15; c_ic=param_nav_13;
    
            phi=angle_lat+angle_nu        
            delta_r=c_rs*sin(2*phi)+c_rc*cos(2*phi)
            delta_l=c_ls*sin(2*phi)+c_lc*cos(2*phi)
            delta_i=c_is*sin(2*phi)+c_ic*cos(2*phi)
            
            phi=phi+delta_l
            r=a_s*(1.-e_s)+delta_r
            angle_inc=angle_inc+delta_i
        
            x_op=r*cos(phi);y_op=r*sin(phi);
        
            x[i_s,i_t]=x_op*cos(angle_long)-y_op*cos(angle_inc)*sin(angle_long);
            y[i_s,i_t]=x_op*sin(angle_long)+y_op*cos(angle_inc)*cos(angle_long);
            z[i_s,i_t]=y_op*sin(angle_inc);
    return (t_oe,prn,x,y,z,t)


#==============================================================================



#====================READING NAVIGATION========================================

ns=31;nh=24;nt=2*120*nh+1;

f=gps_traj(ns,nh,nt)

t_oe=f[0];prn=f[1];x=f[2];y=f[3];z=f[4];t=(f[5]-t_oe)/3600.
r_h=sqrt(x**2.+y**2.);lon=(arctan2(y,x));lat=(arctan2(z,r_h))

#a_pi=180./pi;i_prn=18
#map=Basemap(llcrnrlon=-180,llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90,suppress_ticks=False)
#map.etopo(cmap=cm.gray,alpha=0.25)
#plot(lon*a_pi,lat*a_pi,'o')
##scatter(lon[i_prn,:]*a_pi,lat[i_prn,:]*a_pi,c=t,s=2*t,marker='o',alpha=0.2);
##colorbar()
#axis('tight')
#show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           READING OBSERVATION
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nh=23-20;nt=2*120*nh+1

t_obs=zeros((nt));
delta_tec=zeros((ns,nt));d=zeros((ns))
year=raw_input(colored('YEAR ','green'))
day=raw_input(colored('DAY ','blue'))
print day,year

i_plot=0
for i_dia in range (0,1):
    nd=str(int(float(day)+i_dia))#str(342+i_dia)
#    s=['unzip','/home/fatima/INPE/XICO/ANALISE_DADOS/GNSS/2008/114/\*.zip']#["unzip"," ",file_path,year,"/",nd,"/","*.zip"]
#    dir_1=str.join(" ",s)
#    print dir_1
#    os.system(dir_1)
    s=[file_path,year,"/",nd,"/","*.08o"]
    dir_2=str.join("",s)
    print "NUMERO DE DIA=",nd
    list_of_files = glob.glob(dir_2)#('/home/fatima/INPE/XICO/ANALISE_DADOS/GNSS/2008/114/ubat*.08o')           # create the list of file
    nr=len(list_of_files);    
    print "NUMBER OF RECEIVERs=",nr
    lon_3=zeros((ns,nt,nr));lat_3=zeros((ns,nt,nr));
    tec_3=zeros((ns,nt,nr));t_3=zeros((ns,nt,nr));
    dtec_corr=zeros((ns,nt,nr))
    lat_rec=zeros((nr));lon_rec=zeros((nr))
    t_rec=zeros((nr,nt));
    nss=zeros((ns,nt,nr))
    prn_obs=zeros((ns,nt,nr))
    
    i_r=0;
    for file_name in list_of_files:
        f_obs=open(file_name, 'r')
        for i_read in range (0,36):         #READING HEADERS
            a_head=f_obs.readline()
            if str.find(a_head,'MARKER NAME')>0:
                sta_name=a_head[0:3]
                print colored("NUMERO DE RECEPTOR=",'blue'),i_r,colored(",NOME DE RECEPTOR=",'blue'),sta_name.upper()
            if str.find(a_head,"APPROX POSITION XYZ") > 0:                 #READING RECEIVER COORDINATES
                b_head=array([ float(val) for val in a_head.split()[0:3] ]);
                x_r=b_head[0];y_r=b_head[1];z_r=b_head[2]; 
                lat_r=(arctan2(z_r,sqrt(x_r**2.+y_r**2.)));lon_r=(arctan2(y_r,x_r))
                lat_rec[i_r]=deg(lat_r);lon_rec[i_r]=deg(lon_r);
                print colored("LOCACAO DE RECEPTOR=",'red'), deg(lat_r),deg(lon_r)
                if (deg(abs(lon_r))<50 and deg(abs(lat_r))>22.75)and deg(abs(lat_r))<26:
                    reg_select=1
                    sta_select=sta_name
                else:
                    reg_select=0
            if str.find(a_head,"TYPES OF OBSERV") > 0:
                b_head=array([ int(val) for val in a_head.split()[0:1] ]);
                data_types=b_head[0];print "NUMEROS DAS MEDIDAS=", data_types
            if str.find(a_head,"TIME OF FIRST OBS") > 0:
                b_head=array([ int(val) for val in a_head.split()[0:4] ]);
                ano=str(b_head[0]-0);mes=str(b_head[1]);dia=str(b_head[2]);
                am=[ano,mes,dia];date=str.join("/",am)
                print colored("DATA=",'green'), date
            if str.find(a_head,"END OF HEADER") > 0: 
                break
            
        i_t=0;t_p=0
        while t_p<nh and reg_select==1:
            a_prn=f_obs.readline()
            if not a_prn:break
            if logical_and(str.find(a_prn,".")>0,str.find(a_prn,"COMMENT") < 0):#RECOGNIZING THE LINE HAVING TIME AND PRN INFORMATION
#                print "READING DATA"
                a_prn=str.replace(a_prn,'G',' ');
                a_prn=str.replace(a_prn,'R',' ');
                b_prn=array([ float(val) for val in a_prn.split()[0:] ]);
#                if int(b_prn[7])>12:break   #AVOIDING THE CURRENT TIME IF NEXT LINE HAS PRN INFORMATION INSTEAD OF DATA
                if int(b_prn[7])>12:
                    a_prn_rest=f_obs.readline()
                    a_prn_rest=str.replace(a_prn_rest,'G',' ');
                    a_prn_rest=str.replace(a_prn_rest,'R',' ');
                    b_prn_rest=array([ float(val) for val in a_prn_rest.split()[0:] ]);
                    b_prn=concatenate((b_prn,b_prn_rest))
                    
                t_obs[i_t]=t_oe+b_prn[3]*3600+b_prn[4]*60+b_prn[5]
                
                for i_s in range (0,int(b_prn[7])): #READING ALL PRNs DATA
                    i_prn=int(b_prn[8+i_s])-1
                    if i_prn > 30:i_prn=0
                    if i_s < len(b_prn[8:]):prn_obs[i_prn,i_t,i_r]=b_prn[8+i_s] #BUT USING ONLY PRN FROM FIRST LINE
                    prn_obs[i_prn,i_t,i_r]=int(b_prn[8+i_s])                    
                    a_data=f_obs.readline()
                    if str.find(a_data,'G')>0:break
                    b_data=array([ float(val) for val in a_data.split()[0:] ])
                    b_data=b_data[abs(b_data)>20]
#                    a_temp=f_obs.readline()
                    for i_ss in range(0,ns):
                        if prn_obs[i_prn,i_t,i_r]==prn[i_ss]:
                            lat_s=(lat[i_ss,i_t]);lon_s=(lon[i_ss,i_t]);
                            r_e=6.37e+06;h=300.e+03;
                            r_geo=sqrt(x[i_ss,i_t]**2.+y[i_ss,i_t]**2.+z[i_ss,i_t]**2.)
                            d_lon=lon_r-lon_s;
                            ang_cc=cos(d_lon)*cos(lat_s);ang_cs=sin(d_lon)*cos(lat_s)
                            r_g1=r_geo*ang_cc-r_e*cos(lat_r);
                            r_g2=r_geo*ang_cs;r_g3=r_geo*sin(lat_s)-r_e*sin(lat_r)
                            r_g=sqrt(r_g1**2.+r_g2**2.+r_g3**2.)
                            lat_cc=cos(lat_r)*cos(lat_s)
                            lat_ss=sin(lat_r)*sin(lat_s)
                            if lat_r>0:
                                elev_ang=arcsin(-(r_geo*lat_cc*cos(d_lon)+r_geo*lat_ss-r_e)/r_g)
                            else:
                                elev_ang=-arcsin(-(r_geo*lat_cc*cos(d_lon)+r_geo*lat_ss-r_e)/r_g)
                            lat_cs=cos(lat_r)*sin(lat_s);
                            lat_sc=sin(lat_r)*cos(lat_s);
                            az1=-r_geo*cos(lat_s)*sin(d_lon)/(r_g*cos(elev_ang));
                            az2=-r_geo*(lat_sc*cos(d_lon)-lat_cs)/(r_g*cos(elev_ang))
                            az_ang=arctan2(az1,az2)
                            ang_p=pi/2.-elev_ang-arcsin(r_e*cos(elev_ang)/(r_e+h))
                            arc_angle=sin(lat_r)*cos(ang_p)+cos(lat_r)*sin(ang_p)*cos(az_ang)
                            lat_ipp=deg(arcsin(arc_angle))
                            arc_angle=sin(ang_p)*sin(az_ang)/cos(lat_r)
                            if abs(arc_angle) > 1.:
                                arc_angle=abs(arc_angle)-int(abs(arc_angle))
                            lon_ipp=deg(lon_r+arcsin(arc_angle))
                            if str(lon_ipp)=='nan':
                                print lat_r,arc_angle,i_r,i_s
                                
                            t_3[i_prn,i_t,i_r]=(t_obs[i_t]-t_oe)/3600.
                            if logical_and(abs(lat_s-lat_r)<10+0,abs(lon_s-lon_r)<10+0):
                                if abs(elev_ang) > rad(25.-0):
                                    e0=8.8541878176204e-12;  # Permittivity of a vacuum = 8.8541878176204E-12_F/_m
                                    Me = 9.1093897e-31;  # Me Electron rest mass = 9.1093897E-31_kg
                                    q = 1.60217733e-19;  # q Electron charge = 1.60217733E-19_coul
                                    fl1 = 1.57542e9;  # GPS L1 frequency = 1.57542 GHz
                                    fl2 = 1.22760e9;  # GPS L2 frequency = 1.22760 GHz
                                    lambda1=3.e+08/fl1;lambda2=3.e+08/fl2
                                    
                                    tec_f=(((8*pi**2*e0*Me/q**2)*((fl2**2*fl1**2)/(fl1**2-fl2**2)))/10**(16));
#                                    if data_types ==8:
#                                        if len(b)>=8:c1=b[0];p2=b[6];p1=b[8];tec_3[i_prn,i_t,i_r]=tec_f*abs(p1-p2)
#                                        if len(b)<8:tec_3[i_prn,i_t,i_r]=0;
#                                    if data_types ==6:
#                                        if len(b)>=5:
#                                            c1=b[0];p2=b[1];p1=c1;tec_3[i_prn,i_t,i_r]=tec_f*abs(p1-p2);
##                                            l1=b[2];l2=b[4];tec_3[i_prn,i_t,i_r]=abs(l1/fl1-l2/fl2)
#                                        if len(b)<5:tec_3[i_prn,i_t,i_r]=tec_3[i_prn,i_t-1,i_r];        
#    #                                if abs(elev_ang) < rad(30.): tec[i_ss,i_t]=0;
                                    
#                                    if len(b_data)>data_types:
#                                        b_data=b_data[abs(b_data)>20]
                                        
                                    if len(b_data)==data_types:
                                        c1=b_data[1];p2=b_data[3];p1=c1;tec_pseudo=tec_f*abs(p2-p1)
                                        l1=b_data[0];l2=b_data[2];tec_phase=100*modf((lambda1*l1-lambda2*l2))[0]#SELECTING DECIMAL 
                                        tec_3[i_prn,i_t,i_r]=tec_phase
                                        dtec_corr[i_prn,i_t,i_r]=tec_pseudo-tec_phase
                                    if len(b_data)<data_types:
                                        tec_3[i_prn,i_t,i_r]=tec_3[i_prn,i_t-1,i_r];
                                        
                                    lat_3[i_prn,i_t,i_r]=lat_ipp;
                                    lon_3[i_prn,i_t,i_r]=lon_ipp;                                   
#                                    if lat_3[i_prn,i_t,i_r]< 40.:tec_3[i_prn,i_t,i_r]=0;
                                    sl_v=1.#sqrt(1.-(cos(elev_ang)*r_e/(r_e+h))**2.)
                                    tec_3[i_prn,i_t,i_r]=sl_v*tec_3[i_prn,i_t,i_r]
                                    nss[i_prn,i_t,i_r]=1
                t_p=(t_obs[i_t]-t_oe)/3600. 
                t_rec[i_r,i_t]=t_p;
                i_t=i_t+1;
        for i_prn in range (0,ns):
            tec_3[i_prn,:,i_r]=tec_3[i_prn,:,i_r]+dtec_corr[i_prn,:,i_r].mean()
        i_r=i_r+1
#        t_3=ma.masked_equal(t_3,0);tec=ma.masked_equal(tec,0);
        tec_3=ma.masked_equal(tec_3,0)
        print colored("TEC, MIN, MAX= ",'blue'),tec_3[:,:,i_r-1].min(),tec_3[:,:,i_r-1].max()
        print "========================================"
        
#================================POR_DE_SOL====================================        
#        f=por_de_sol(i_r,nr,ano,mes,dia,lat_r,lon_r)
#        print f
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                           Figures
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        if reg_select==1:
            
            cc=cm.get_cmap('jet',5);col_map=cc(arange(5))    
            col_map=['r','b','k','deeppink','purple']
            ep_lon=-45.29;ep_lat=-25.65
            fig = figure(1,figsize=(12,12),facecolor='w',edgecolor='k') 
            ax=subplot(111)       
            nx=arange(0,len(t_rec[i_r-1,:])-120)
            i_prn=13
            lon_2=0*(i_dia-2)+lon_3[i_prn,nx,i_r-1];lat_2=0*(i_dia-2)+lat_3[i_prn,nx,i_r-1];
            r_lon=lon_2.mean()-ep_lon;r_lat=lat_2.mean()-ep_lat
            ep_dist=sqrt(r_lon**2.+r_lat**2.)
            tec_2=tec_3[i_prn,nx,i_r-1];
            t_2=t_3[i_prn,nx,i_r-1];prn_2=prn_obs[i_prn,nx,i_r-1]
            n_m=int(len(t_2)/8)
            tec_av=tec_2-convolve(tec_2,ones((n_m,))/n_m)
            map=Basemap(llcrnrlon=-80,llcrnrlat=-35,urcrnrlon=-35,urcrnrlat=10,suppress_ticks=False)
            map.fillcontinents(color='gray',lake_color='aqua',alpha=0.05)
            
            plot(ep_lon,ep_lat,'*',markersize=18,color='b')
            text(ep_lon,ep_lat-0.25,'Epicenter')
            plot(-47.42,-23.59,'s',markersize=18,color='w')
#            text(-47,-23,'Seismometer')
            lon_x,lat_y=map(lon_rec[i_r-1],lat_rec[i_r-1])
            plot(lon_x,lat_y,'^',markersize=14,color=col_map[i_plot-1])
            lon_x,lat_y=map(lon_2,lat_2)
            text(lon_x[0]-0.35,lat_y[0]-0.25,sta_name.upper(),color=col_map[i_plot-1])
            plot(0.25*tec_av+lon_x,0.5+0.25*tec_av+lat_y,'w')
            im=scatter(lon_x,lat_y,c=tec_2,s=t_2,vmax=60,vmin=10,
                       marker='o',color='k',alpha=1) 
            im=scatter(0.25*tec_av+lon_x,0.5+0.25*tec_av+lat_y,c=tec_2,s=50*t_2,vmax=60,vmin=10,
                       marker='o',cmap=cm.jet,alpha=1) 
            im1=scatter(lon_x,0.5*tec_av+lat_y,c=0*tec_2,s=0*t_2,vmax=60,vmin=10,
                       marker='o',cmap=cm.jet,alpha=1) 
            axis('tight');axis((-53,-43,-28,-20));
            xlabel('Longitude, $^o$');ylabel('Latitude, $^o$');#title('$TEC$')
        
            divider=make_axes_locatable(ax);cax=divider.append_axes("right",size="2%",pad=-0.05)
            colorbar(im1,cax=cax);gca().set_title('TECU',fontsize=10)
            
            #======================================================================
            
            fig = figure(2,figsize=(12,12),facecolor='w',edgecolor='k') 
            i_prn=13;
            nx=arange(0,len(t_rec[i_r-1,:]),4)
            t_t=zoom(t_3[i_prn,nx,i_r-1],4)
            tec_t=zoom(tec_3[i_prn,nx,i_r-1],4)
            n_m=int(len(t_t)/8)
            tec_av=convolve(tec_t,ones((n_m,))/n_m)
            str_label=sta_name.upper()
            str_label=str(i_dia-2)
            subplot(121)
            if str.find(file_name,'114')>0:
                plot(t_t,tec_t,'b',linewidth=2,label='%s' %str_label)  
                fill_between(t_t,tec_t,tec_av,color='r')
            else:
                plot(t_t,tec_t,'gray',linewidth=2,label='%s' %str_label) 
                fill_between(t_t,tec_t,tec_av,color='k',alpha=0.5)
#            legend(loc=0)
            if sta_name.upper()=='UFP':
                text(t_t[0],tec_t[0]-3,sta_name.upper()) 
            else:
                text(t_t[0],tec_t[0],sta_name.upper()) 
            title('(A)')
            xlabel('Time, UT');ylabel('TECU')
            axis((0.,3.,20-20,35+25))
            ax=subplot(5,2,2*i_plot+2)
            f=wavelet(t_t*3600.,tec_t)
            dt_freq=f[0];
            pwr=1.e+02*f[1]
            im=pcolormesh(t_t,1.e+03/dt_freq,pwr,cmap=cm.gray_r,vmax=25,vmin=0.0);
            axis((0.,3.,0.2e-00,5.e-00));
            str_title=str.join('',[sta_name.upper(),',','$d_{es}$=',str(round(ep_dist,1)),'$^o$'])
            title(str_title)
#            title(str(i_dia-2))
            if i_plot==0:text(1.5,6,'(B)')
            if i_plot==4:
                xlabel('Time, UT')
            else:
                ax.set_xticklabels([]);
            
            gca().add_patch(Rectangle((1.5,1.5),1,2.,alpha=1,fill=None,color='g'))
            if str.find(file_name,'uba')>0:
                ax.tick_params(axis='y',colors='blue')
                ylabel('mHz')
                divider = make_axes_locatable(ax);cax = divider.append_axes("right",size="2%",pad=0.01)
                colorbar(im,cax=cax,ticks=[0,10,20]);gca().set_ylabel('Spectral Density',fontsize=10)
    
            #======================================================================
    
            fig = figure(3,figsize=(12,12),facecolor='w',edgecolor='k')
            i_tt=1
            for i_t in range(0,450,50):
                ax=subplot(3,3,i_tt)            
                nx=arange(i_t,i_t+8)
                i_prn=13
                lon_2=lon_3[:,nx,i_r-1];lat_2=lat_3[:,nx,i_r-1];tec_2=tec_3[:,nx,i_r-1];
                t_2=t_3[:,nx,i_r-1];prn_2=prn_obs[:,nx,i_r-1]
                map=Basemap(llcrnrlon=-80,llcrnrlat=-60,urcrnrlon=-30,urcrnrlat=10,suppress_ticks=False)
                map.fillcontinents(color='gray',lake_color='aqua',alpha=0.01)
                
                lon_x,lat_y = map(lon_2,lat_2)
                im=scatter(lon_x,lat_y,s=50.+0*tec_2/tec_2.max(),c=tec_2,vmin=20,vmax=50,
                           marker='s',cmap=cm.jet,alpha=0.75) 

                axis((-80,-30,-60,10))
                title(str(i_t))
                ax.set_xticklabels([]);ax.set_yticklabels([]);
            
    #            divider=make_axes_locatable(ax);cax=divider.append_axes("right",size="2%",pad=-0.05)
    #            colorbar(im,cax=cax);gca().set_title('TECU',fontsize=10)
                i_tt=i_tt+1
            i_plot=i_plot+1

show()    
    #==========================================================================
    
        
            
    