from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import georinex as gr
import gzip
import subprocess
import os
import pandas as pd
import glob
import datetime
import juliandate as jd
import gpsdatetime as gt
import pymap3d as pm


close('all')

def decompress_file(fnc):
    if fnc.endswith('.gz'):
        fn = fnc[:-3]
        # with gzip.open(fnc, 'rb') as f_in, open(fn, 'wb') as f_out:
        #     f_out.write(f_in.read())
        subprocess.run(['gunzip', fnc])
        return fn
    elif fnc.endswith('.Z'):
        fn = fnc[:-2]
        subprocess.run(['uncompress', fnc])
        return fn
    return fnc

#%%
ns=31;#nh=24;nt=60*nh+1;dt_nav=2*60#float(input('TIME RESOLUTION '))nt=int(nh*3600./dt_nav)
prn_nm=[]
def gps_traj_old(ti,tf,nt,dt_nav):
    data=zeros((ns,29));prn=zeros((ns));
    x=zeros((ns,nt));y=zeros((ns,nt));z=zeros((ns,nt));t=zeros((nt))
    fn='nav.dat'
    f=open(fn,'r')
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
        
        if prn[i_s]==19:
            print (data[i_s,0:28])
        t=linspace(ti,tf+dt_nav,nt)
        
        t_oe=param_nav_12;
        toe_d=t_oe/86400
        toe_h=(toe_d-int(toe_d))*86400/3600.
        toe_m=(toe_h-int(toe_h))*3600/60
        toe_s=(toe_m-int(toe_m))*60.
        
        toe_uts=toe_h*3600.+toe_m*60.+toe_s
        print (ti/3600,tf/3600,toe_uts/3600)
        for i_t in range (0,nt):
            #t[i_t]=t_oe+i_t*dt_nav;
            a_s=(param_nav_11)**2.;e_s=param_nav_9;
            m_o=param_nav_7;delta_n=param_nav_6;
            m=m_o+(sqrt(3.98e+14/a_s**3.)+delta_n)*(t[i_t]-toe_uts)
            cap_e=m+(e_s-e_s**3./8+e_s**5./192-e_s**7./9216.)*sin(m)
            
            angle_nu=arctan2(sqrt(1-e_s**2.)*sin(cap_e),(cos(cap_e)-e_s));#nu
            angle_inc=param_nav_16+param_nav_20*(t[i_t]-toe_uts);#i
            angle_lat=param_nav_18 #omega
            angle_long=param_nav_14+param_nav_19*(t[i_t]-toe_uts)-7.79e-05*(t[i_t]);#OMEGA
        
            c_rs=param_nav_5; c_rc=param_nav_17;   
            c_ls=param_nav_10; c_lc=param_nav_8;
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
        prn_nm.append('G'+str(int(prn[i_s])))
    return array(prn_nm),x,y,z,t

#%%
# Función para calcular la trayectoria GPS
def gps_traj(ti,tf,nt, dt_nav):
    fn='salu3451.21n'#
    # fn='ABMF00GLP_R_20240010000_01D_MN.rnx.gz'#'brdc0010.24n.gz'#
    fn='ELKO00USA_R_20182100000_01D_MN.rnx'
    info = gr.rinexinfo(fn)
    print ('NAVIGATION RINEX',info)
    nav_data=gr.load(fn)
    
    satellites = nav_data.sv.values#[18:19]
    i0=abs(np.char.find(satellites,'G19')-0).argmin()
    print(i0)
    #satellites = satellites[i0:i0+1]
    x = np.zeros((len(satellites), nt))
    y = np.zeros((len(satellites), nt))
    z = np.zeros((len(satellites), nt))
    t = np.zeros(nt)
    
    params = [
        'Toe', 'SVclockBias', 'sqrtA', 'Eccentricity', 'M0', 'DeltaN',
        'Io', 'IDOT', 'omega', 'Omega0', 'OmegaDot', 'Crs', 'Crc', 'Cus',
        'Cuc','Cic', 'Cis'
    ]
    
    for i, sv_id in enumerate(satellites):
        
        try:
            if not all(isinstance(sv_data[param].values.item(), (float, np.floating, int)) for param in params):
                raise ValueError("Datos de navegación faltantes o no numéricos")
            
            sv_data= nav_data.sel(sv=sv_id).dropna(how='all', dim='time')
                            #iG02=abs(char.find(satellites,'sv_id')-0).argmin()
                            #print (sv_data.variables)
                            #print (sv_data['Toe'].values[0])
            param_nav = {param: sv_data[param].values[0] for param in params}
            
            print (param_nav)
            # Calcular la posición del satélite en cada tiempo
            t_oe = param_nav['Toe']
            a_s = param_nav['sqrtA'] ** 2
            e_s = param_nav['Eccentricity']
            m_o = param_nav['M0']
            delta_n = param_nav['DeltaN']
            omega_dot = param_nav['OmegaDot']
            omega = param_nav['omega']
            omega_0 = param_nav['Omega0']
            idot = param_nav['IDOT']
            io = param_nav['Io']
            crs = param_nav['Crs']
            crc = param_nav['Crc']
            cus = param_nav['Cus']
            cuc = param_nav['Cuc']
            cic = param_nav['Cic']
            cis = param_nav['Cis']
            
            #a_s=(5.15367114e+03)**2.
            #e_s=4.56996483e-03
            #m_o=-8.00150823e-01
            #delta_n=4.44732811e-09
            #io=9.57593511e-01
            #idot=-5.76452583e-10
            # omega=-6.23673898e-01
            #omega_0=-5.13406698e-01
            #omega_dot=-7.88425698e-09
            #crs=-2.14687500e+01
            #crc=2.08937500e+02
            #cus=8.90716910e-06
            #cuc=-1.06729567e-06
            #cic=-1.75088644e-07
            #cis=-1.02445483e-07
            
            t=linspace(ti,tf+dt_nav,nt)
            
            toe_d=t_oe/86400
            toe_h=(toe_d-int(toe_d))*86400/3600.
            toe_m=(toe_h-int(toe_h))*3600/60
            toe_s=(toe_m-int(toe_m))*60.
            
            toe_uts=(toe_h*3600.+toe_m*60.+toe_s)
            print (ti,tf,t_oe)
            
            mu_ea=3.986005e+14 #m³/s²
            Omega_dot_ea=7.2921151467e-05 #rad/s
            for i_t in range(nt):
                # t[i_t] = t_oe + i_t * dt_nav
                delta_t=t[i_t]-t_oe
                # print (delta_t)
                if delta_t>302400:
                    delta_t=delta_t-2*302400
                if delta_t<-302400:
                    delta_t=delta_t+2*302400
                # print (delta_t)
                
                n0=sqrt(mu_ea/a_s**3.)
                m = m_o + (n0 + delta_n)*delta_t #Mean anomaly
                
                cap_g = m#m + (e_s - e_s**3./8 + e_s**5/192 - e_s **7./ 9216.) * np.sin(m) #Eccentric anomaly
                for iter in range (11):
                    cap_e=m+e_s*sin(cap_g)
                    cap_g=cap_e
                angle_nu = np.arctan2(np.sqrt(1 - e_s ** 2) * np.sin(cap_e), np.cos(cap_e) - e_s) #Tru anomaly
                phi = omega + angle_nu
                    
                delta_r = crs * np.sin(2 * phi) + crc * np.cos(2 * phi)
                delta_u = cus * np.sin(2 * phi) + cuc*cos(2.*phi)
                delta_i = cis * np.sin(2 * phi) + cic * np.cos(2 * phi)

                phi =phi+ delta_u        
                angle_inc =io + idot * delta_t+ delta_i
                angle_long = omega_0 + omega_dot * delta_t - Omega_dot_ea * t[i_t] 
                
                r = a_s * (1 - e_s*cos(cap_e)) + delta_r
                x_op = r * np.cos(phi)
                y_op = r * np.sin(phi)

                x[i, i_t] = x_op * np.cos(angle_long) - y_op * np.cos(angle_inc) * np.sin(angle_long)
                y[i, i_t] = x_op * np.sin(angle_long) + y_op * np.cos(angle_inc) * np.cos(angle_long)
                z[i, i_t] = y_op * np.sin(angle_inc)
                #print (satellites[i],x[i,i_t])
                
        except Exception as e:
            print(f"No se pudo calcular la posición para el satélite {sv_id}: {e}")

    return satellites, x, y, z, t

#%%
def weeksecondstoutc(gpsweek,gpsseconds,leapseconds):
    import datetime, calendar
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    elapsed = datetime.timedelta(days=(gpsweek*7),seconds=(gpsseconds-leapseconds))
    return datetime.datetime.strftime(epoch + elapsed,datetimeformat)

def utctoweekseconds(utc,leapseconds):
    """ Returns the GPS week, the GPS day, and the seconds 
            and microseconds since the beginning of the GPS week """
    import datetime, calendar
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    tdiff = utc -epoch  + datetime.timedelta(seconds=leapseconds)
    gpsweek = tdiff.days // 7 
    gpsdays = tdiff.days - 7*gpsweek         
    gpsseconds = tdiff.seconds + 86400 * (tdiff.days -7*gpsweek) 
    return (gpsweek,gpsdays,gpsseconds,tdiff.microseconds)

def gmst(tut):
    "Greenwich mean sidereal utc_time, in radians."
    
    # ut1 = jdays2000(utc_time) / 36525.0
    # theta = 67310.54841 + ut1 * (876600 * 3600 + 8640184.812866 + ut1 *
    #                              (0.093104 - ut1 * 6.2 * 10e-6))
    jday=2458484.833333
    jday=jd.from_gregorian(2019,7,19,tut.hour,tut.minute,tut.second)
    print (jday)
    J0= floor(jday) + 0.5
    dJ0= jday - J0
    H = dJ0 * 24              
    D = jday - 2451545.0     
    T= D / 36525  
    D0 = J0 - 2451545.0

    gmst = 6.697374558 + 0.06570982441908 * D0+ 1.00273790935 * H+ 0.000026 * T**2 
    gh=gmst % 24; gm=(gmst-floor(gmst))*60.;gs=(gm-floor(gm))*60.
    gt=gh+gm/60+gs/3600
    return gt#np.deg2rad(theta / 240.0) % (2 * np.pi)

  
#%%
def ipp_traj(sat_x,sat_y,sat_z,obs_x,obs_y,obs_z,lon_r,lat_r):
    obs_x /= 1000;obs_y /= 1000;obs_z /= 1000
    sat_x /= 1000;sat_y /= 1000;sat_z /= 1000
    
    dx = (sat_x - obs_x)
    dy = (sat_y - obs_y)
    dz = (sat_z - obs_z)

    dxdy = (sat_x * obs_y - sat_y * obs_x)
    dxdz = (sat_x * obs_z - sat_z * obs_x)
    
    h=250.
    arg_1 = pow((dy * dxdy + dz * dxdz), 2)
    arg_2 = pow(dx, 2) + pow(dy, 2) + pow(dz, 2)
    arg_3 = pow(dxdy, 2) + pow(dxdz, 2) - pow(dx, 2) * pow(h, 2)
    arg_4 = pow(dy, 2) + pow(dz, 2) + pow(dx, 2)

    factor = np.sqrt(arg_1 - arg_2 * arg_3)
    
    sub_ion_x = (-1 * (dy * dxdy + 
                       dz * dxdz) - factor) / arg_4
    
    sub_ion_y = (dy * sub_ion_x + dxdy) / dx
    sub_ion_z = (dz * sub_ion_x + dxdz) / dx
    
    meridional = (-cos(lon_r) * sin(lat_r) * dx - 
                  sin(lon_r) * sin(lat_r) * dy + 
                  cos(lat_r) * dz)

    zonal = - sin(lon_r) * dx + cos(lon_r) * dy

    vertical = (cos(lon_r) * cos(lat_r) * dx + 
                sin(lon_r) * cos(lat_r) * dy + 
                sin(lat_r) * dz)
    
    
    arg_1 = pow(cos(lon_r) * cos(lat_r), 2)
    arg_2 = pow(sin(lon_r) * cos(lat_r), 2)
    arg_3 = pow(sin(lat_r), 2)
    vertical_norm = sqrt(arg_1 + arg_2 + arg_3)
    r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))

    zangle=arccos(vertical / (r * vertical_norm))
    
    el_angle=((pi / 2.0) - zangle) #/ (pi/180.)  
    az_angle = arctan2(zonal, meridional)
    az_angle[az_angle<0]=az_angle[az_angle<0]+2.*pi
    # if az_angle < 0:
    #     az_angle += 2.0 * pi 
    
    el = el_angle #* (pi/180.)
    Re = 6.371e+03
    zangle_ion=((pi / 2.0) - el - arcsin((Re / (Re + h)) * cos(el)))

    lat_ipp = arcsin(sin(lat_r) * cos(zangle_ion) + cos(lat_r) * 
                       sin(zangle_ion) * cos(az_angle))

    lon_ipp = lon_r + arcsin((sin(zangle_ion) * sin(az_angle)) / 
                             (cos(lat_ipp)))
    return  (lat_ipp,lon_ipp,el)

#%%
def tec_calc (i_so,info,obs,prn,obs_ty):
    if char.find(prn[i_so],'G')==0:
        if int(info['version'])<3:obs_tys=obs_ty
        if int(info['version'])==3:obs_tys=obs_ty['G']
        f1 = 1575.42e6;f2 = 1227.60e6;f5=1176.45e+06
        f_1=f1;f_2=f2
        lf1=obs[obs_tys[abs(char.find(obs_tys,'L1')).argmin()]].values[:,i_so]
        lf2=obs[obs_tys[abs(char.find(obs_tys,'L2')).argmin()]].values[:,i_so]
    if char.find(prn[i_so],'R')==0:#GLONASS
        if int(info['version'])<3:obs_tys=obs_ty
        if int(info['version'])==3:obs_tys=obs_ty['R']
        f4=1600.99e6;f2=1248.06e6 #L1/L2
        f_1=f4;f_2=f2
        lf1=obs[obs_tys[abs(char.find(obs_tys,'L1')).argmin()]].values[:,i_so]
        lf2=obs[obs_tys[abs(char.find(obs_tys,'L2')).argmin()]].values[:,i_so]
    if char.find(prn[i_so],'E')==0:#GALILEO
        if int(info['version'])<3:obs_tys=obs_ty
        if int(info['version'])==3:obs_tys=obs_ty['E']
        f1=1575.42e6;f5=1176.45e6 #L1/L5
        f_1=f1;f_2=f5
        lf1=obs[obs_tys[abs(char.find(obs_tys,'L1')).argmin()]].values[:,i_so]
        lf2=obs[obs_tys[abs(char.find(obs_tys,'L5')).argmin()]].values[:,i_so]
    if char.find(prn[i_so],'C')==0:#BEIDOU
        if int(info['version'])<3:obs_tys=obs_ty
        if int(info['version'])==3:obs_tys=obs_ty['C']
        f1=1575.42e6;f5=1176.45e6 #L1/L5
        f_1=f1;f_2=f5
        lf1=obs[obs_tys[abs(char.find(obs_tys,'L1')).argmin()]].values[:,i_so]
        lf2=obs[obs_tys[abs(char.find(obs_tys,'L5')).argmin()]].values[:,i_so]
    if char.find(prn[i_so],'S')==0:#SBAS
        if int(info['version'])<3:obs_tys=obs_ty
        if int(info['version'])==3:obs_tys=obs_ty['S']
        f1=1575.42e6;f5=1176.45e6 #L1/L5
        f_1=f1;f_2=f5
        lf1=obs[obs_tys[abs(char.find(obs_tys,'L1')).argmin()]].values[:,i_so]
        lf2=obs[obs_tys[abs(char.find(obs_tys,'L5')).argmin()]].values[:,i_so]
    
    
    lf1[np.isnan(lf1)!=0] = 0
    lf2[np.isnan(lf2)!=0] = 0
    
    alpha = 40.3e16  # Constante para el cálculo de TECU
    lambda1 = 3e8 / f_1
    lambda2 = 3e8 / f_2
    const = (f_1**2*f_2**2)/(alpha*(f_1**2 - f_2**2))
    tec =const*(lf1 * lambda1 - lf2*lambda2)
    return tec

#%%
def calculate_tec_from_rinex(fnc):
    #fn= decompress_file(fnc)
    info = gr.rinexinfo(fnc)
    #print ('OBSERVATION RINEX INFO',info)
    hdr = gr.rinexheader(fnc)
    #print (hdr)
    
    try:
        obs = gr.load(fnc)#,tlim=('2019-07-04T16:00', '2019-07-04T18:00'))
    except Exception as e:
        print(f"Error al cargar el archivo RINEX: {e}")
        return None

    #print (obs.data_vars)
    #print (obs.sv.to_index())
    if int(info['version'])<3:
        obs_ty=hdr.get('# / TYPES OF OBSERV')
    if int(info['version'])==3:
        obs_ty=hdr.get('fields')
    dto=float(hdr.get('INTERVAL'))
    prn = obs['sv'].values
    times = obs.time
    
    ts = gr.to_datetime(obs.time)
    ti=3600*ts[0].hour+60*ts[0].minute+ts[0].second
    tf=3600*ts[-1].hour+60*ts[-1].minute+ts[-1].second
    ts_h=linspace(ti/3600.,tf/3600.,len(ts))
    print (ts[0])
    
    ti=utctoweekseconds(datetime.datetime.strptime(str(ts[0]),"%Y-%m-%d %H:%M:%S"),0)[2]
    tf=utctoweekseconds(datetime.datetime.strptime(str(ts[-1]),"%Y-%m-%d %H:%M:%S"),0)[2]
    
    print (ti,tf)
    x_r,y_r,z_r=obs.position
    lat_r=arctan2(z_r,sqrt(x_r**2.+y_r**2.));lon_r=arctan2(y_r,x_r)
    
    lat_rec, lon_rec, h_rec =obs.position_geodetic
    lat_r=radians(lat_rec);lon_r=radians(lon_rec+360)
    
    satellites, x, y, z, t = gps_traj(ti,tf,int(len(times)), dto)
    
    r_h = np.sqrt(x**2. + y**2.);
    lon_n = np.arctan2(y, x);lat_n = np.arctan2(z, r_h)
        
    r_e = 6.37e+06;h=250.e+03;
    #lon_ipp, lat_ipp, el, az= [], [], [], []
    
    for i_so in range(len(prn)):
        
        i0=np.char.find(satellites,prn[i_so])
        i_ss=abs(i0-0).argmin()
        
        if satellites[i_ss]!=prn[i_so]:
            print('PRN NOT FOUND=',satellites[i_ss],prn[i_so])
            continue
        print('PRN FOUND=',satellites[i_ss],prn[i_so])
        
        lat_s=(lat_n[i_ss,:]);lon_s=(lon_n[i_ss,:])
        
        
        
        #%%
        
        f=ipp_traj(x[i_ss,:],y[i_ss,:],z[i_ss,:],x_r,y_r,z_r,lon_r,lat_r)
        lat_ipp=degrees(f[0]);lon_ipp=degrees(f[1]);el_angle=degrees(f[2])
        
        #%%
     
        tec=tec_calc(i_so,info,obs,prn,obs_ty)
        if len(tec[tec==0])>len(ts)/4.:
            print ('MEASUREMENTS NOT AVAIL')
            #continue
        #%%
        lon0=degrees(lon_r);lat0=degrees(lat_r)
        dlon=lon_ipp-lon0;dlat=lat_ipp-lat0
        elmx=30;
        lon_ip=lon_ipp[abs(el_angle)>70]
        lat_ip=lat_ipp[abs(el_angle)>70]
        t_ip=ts_h[abs(el_angle)>70]
        tec_ip=tec[abs(el_angle)>70]
        
        figure(1,(12,12))
        plot(lon_ip,lat_ip,'o')
        plot(lon0,lat0,'*')  
        
        figure(2,(12,12))
        plot(t_ip,tec_ip,'o')
        
        draw()
        
    return times, tec, prn, lon_ipp, lat_ipp, lon_r, lat_r, satellites, x, y, z#,el, az


#%%
fl =['salu3451.21d']#
#fl=['SALU00BRA_R_20213450000_01D_15S_MO.crx.gz']#

plot_data = []
#fl = glob.glob('/home/alam/INPE/GNSS_NOVO/TEC/ncedc.2019.185/ucd*.Z')
#print (fl)
for fn in fl:
   print (fn)
   f=calculate_tec_from_rinex(fn)
