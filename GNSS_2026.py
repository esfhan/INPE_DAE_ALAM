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

#%%
def gal_utctoweekseconds(utc,leapseconds):
    from datetime import datetime, timezone, timedelta

    # Data UTC de exemplo
    utc_time = utc
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)#, tzinfo=timezone.utc)
    gps_epoch = datetime(1999, 8, 22, 0, 0, 0)
    # Offset atual do GPS em relação ao UTC (18 segundos)
    leap_seconds = 18
    
    # Converter UTC para tempo GPS
    gps_time = utc_time + timedelta(seconds=leap_seconds)
    delta = gps_time - gps_epoch
    
    total_seconds = int(delta.total_seconds())
    gps_week = total_seconds // 604800
    gps_seconds_of_week = total_seconds % 604800
    
    print(f"GPS Week: {gps_week}")
    print(f"Seconds of Week: {gps_seconds_of_week}")
    return (gps_week,gps_seconds_of_week)

#%%
def sat_ephimeries():
    fn='ELKO00USA_R_20182100000_01D_MN.rnx'
    info = gr.rinexinfo(fn)
    print ('NAVIGATION RINEX',info)
    nav_data=gr.load(fn)
    
    satellites = nav_data.sv.values#[18:19]
    
    eph_name = [
        'Toe', 'SVclockBias', 'sqrtA', 'Eccentricity', 'M0', 'DeltaN',
        'Io', 'IDOT', 'omega', 'Omega0', 'OmegaDot', 'Crs', 'Crc', 'Cus',
        'Cuc','Cic', 'Cis'
    ]
    
    eph_all=[]
    for i, sv_id in enumerate(satellites):
        sv_data= nav_data.sel(sv=sv_id).dropna(how='all', dim='time')
        param_nav = {param: sv_data[param].values[0] for param in eph_name}
        eph_all.append([param_nav[param] for param in eph_name])
    return satellites,eph_name,eph_all

#%%
def kepler_orbit_all(ti,tf,dt,nt):
    
    satellites,eph_name,eph_all=sat_ephimeries()
    
    eph_all=np.array(eph_all)
    
    t_oe = eph_all[:,eph_name.index("Toe")]
    a_s = eph_all[:,eph_name.index("sqrtA")] ** 2
    e_s = eph_all[:,eph_name.index('Eccentricity')]
    m_o = eph_all[:,eph_name.index('M0')]
    delta_n = eph_all[:,eph_name.index('DeltaN')]
    omega_dot = eph_all[:,eph_name.index('OmegaDot')]
    omega = eph_all[:,eph_name.index('omega')]
    omega_0 = eph_all[:,eph_name.index('Omega0')]
    idot = eph_all[:,eph_name.index('IDOT')]
    io = eph_all[:,eph_name.index('Io')]
    crs = eph_all[:,eph_name.index('Crs')]
    crc = eph_all[:,eph_name.index('Crc')]
    cus = eph_all[:,eph_name.index('Cus')]
    cuc = eph_all[:,eph_name.index('Cuc')]
    cic = eph_all[:,eph_name.index('Cic')]
    cis = eph_all[:,eph_name.index('Cis')]
    
    print (len(t_oe),len(eph_all))
    x=zeros((109,nt));y=zeros((109,nt));z=zeros((109,nt))
    
    t=linspace(ti,tf+dt,nt)
    
    mu_ea=3.986005e+14 #m³/s²
    Omega_dot_ea=7.2921151467e-05 #rad/s
    for i_t in range(nt):
        delta_t=t[i_t]-t_oe
        
        delta_t[delta_t>302400]=delta_t[delta_t>302400]-2*302400
        delta_t[delta_t<-302400]=delta_t[delta_t<-302400]+2*302400
        
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

        x[:, i_t] = x_op * np.cos(angle_long) - y_op * np.cos(angle_inc) * np.sin(angle_long)
        y[:, i_t] = x_op * np.sin(angle_long) + y_op * np.cos(angle_inc) * np.cos(angle_long)
        z[:, i_t] = y_op * np.sin(angle_inc)
        
    return (satellites,x,y,z,t)
      
#%%
def ipp_traj(sat_x,sat_y,sat_z,obs_x,obs_y,obs_z,lon_r,lat_r):
    r_e = 6.37e+03;h=250.;
    
    obs_x /= 1000;obs_y /= 1000;obs_z /= 1000
    sat_x /= 1000;sat_y /= 1000;sat_z /= 1000
    
    dx = (sat_x - obs_x)
    dy = (sat_y - obs_y)
    dz = (sat_z - obs_z)

    dxdy = (sat_x * obs_y - sat_y * obs_x)
    dxdz = (sat_x * obs_z - sat_z * obs_x)
    
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
    zangle_ion=((pi / 2.0) - el - arcsin((r_e / (r_e + h)) * cos(el)))

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
    tec=tec-tec[0]
    return tec

#%%
def obs_rinex_to_tec(fnc):
    #fn= decompress_file(fnc)
    info = gr.rinexinfo(fnc)
    #print ('OBSERVATION RINEX INFO',info)
    hdr = gr.rinexheader(fnc)
    
    sta_name=hdr['MARKER NAME']
    btime=hdr['TIME OF FIRST OBS'].split()
    #etime=hdr['TIME OF LAST OBS'].split()
    print (sta_name,btime)
    b_str=btime[0]+'-'+btime[1]+'-'+btime[2]+'T'+'12:00'
    e_str=btime[0]+'-'+btime[1]+'-'+btime[2]+'T'+'16:00'
    
    try:
        obs = gr.load(fnc,tlim=(b_str,e_str))#,tlim=('2021-12-11T12:00', '2021-12-11T16:00'))
    except Exception as e:
        print(f"Error al cargar el archivo RINEX: {e}")
        return None

    #print (obs.data_vars)
    #print (obs.sv.to_index())
    if int(info['version'])<3:
        obs_ty=hdr.get('# / TYPES OF OBSERV')
    if int(info['version'])==3:
        obs_ty=hdr.get('fields')
    dto=float(hdr.get('INTERVAL'));n_skip=int(300./dto)
    prn = obs['sv'].values
    times = obs.time
    
    ts = gr.to_datetime(obs.time)
    tis=3600*ts[0].hour+60*ts[0].minute+ts[0].second
    tfs=3600*ts[-1].hour+60*ts[-1].minute+ts[-1].second
    ts_h=linspace(tis/3600.,tfs/3600.,len(ts))
    
    ti=utctoweekseconds(datetime.datetime.strptime(str(ts[0]),"%Y-%m-%d %H:%M:%S"),18)[2]
    tf=utctoweekseconds(datetime.datetime.strptime(str(ts[-1]),"%Y-%m-%d %H:%M:%S"),18)[2]
    
    #utc=datetime.datetime(ts[0].year,ts[0].month,ts[0].day,ts[0].hour,ts[0].minute,ts[0].second)
    #tig=gal_utctoweekseconds(utc, 0)[1]print(ts[0].year,ts[0].month,ts[0].day,ts[0].hour,ts[0].minute,ts[0].second)
    
    x_r,y_r,z_r=obs.position
    lat_r=arctan2(z_r,sqrt(x_r**2.+y_r**2.));lon_r=arctan2(y_r,x_r)
    
    lat_rec, lon_rec, h_rec =obs.position_geodetic
    lat_r=radians(lat_rec);lon_r=radians(lon_rec+360)
       
    satellites,x,y,z,t=kepler_orbit_all(ti,tf, n_skip*dto,int(len(times)/n_skip))
    
    print ('NAVIGATION SATELLITES=',satellites)
    print ('OBSERVATION SATELLITES=',prn)
    for i_so in range(len(prn)):
        
        i0=np.char.find(satellites,prn[i_so])
        i_ss=abs(i0-0).argmin()
        
        if satellites[i_ss]!=prn[i_so]:
            print('PRN NOT FOUND=',satellites[i_ss],prn[i_so])
            continue
        print('PRN FOUND=',satellites[i_ss],prn[i_so])
        
        
        #%%
        
        f=ipp_traj(x[i_ss,:],y[i_ss,:],z[i_ss,:],x_r,y_r,z_r,lon_r,lat_r)
        lat_ipp=degrees(f[0]);lon_ipp=degrees(f[1]);el_angle=degrees(f[2])
        
        #%%
     
        tec=tec_calc(i_so,info,obs,prn,obs_ty)[::n_skip]
        if len(tec[tec==0])>len(ts)/4.:
            print ('MEASUREMENTS NOT AVAIL')
            continue
        #%%
        lon0=degrees(lon_r);lat0=degrees(lat_r)
        dlon=lon_ipp-lon0;dlat=lat_ipp-lat0
        elmx=30;
        lon_ip=lon_ipp[abs(el_angle)>elmx]
        lat_ip=lat_ipp[abs(el_angle)>elmx]
        t_ip=ts_h[0:-2:n_skip][abs(el_angle)>elmx]
        tec_ip=tec[0:-1][abs(el_angle)>elmx]
        
        figure(1,(12,12))
        plot(lon_ip,lat_ip,'o')
        plot(lon0,lat0,'*')  
        
        figure(2,(12,12))
        plot(t_ip,tec_ip,'o')
        axis((0,24,-200,200))
        draw()
        
    return times, tec, prn, lon_ipp, lat_ipp, lon_r, lat_r, satellites, x, y, z#,el, az


#%%
fl =['salu3451.21d']#
fl=['SALU00BRA_R_20213450000_01D_15S_MO.crx.gz']#

plot_data = []
#fl = glob.glob('/home/alam/INPE/GNSS_NOVO/TEC/2019.185/a*.Z')
#print (fl)
for fn in fl:
   print (fn)
   f=obs_rinex_to_tec(fn)

#%%
# def weeksecondstoutc(gpsweek,gpsseconds,leapseconds):
#     import datetime, calendar
#     datetimeformat = "%Y-%m-%d %H:%M:%S"
#     epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
#     elapsed = datetime.timedelta(days=(gpsweek*7),seconds=(gpsseconds-leapseconds))
#     return datetime.datetime.strftime(epoch + elapsed,datetimeformat)

# def utctoweekseconds(utc,leapseconds):
#     """ Returns the GPS week, the GPS day, and the seconds 
#             and microseconds since the beginning of the GPS week """
#     import datetime, calendar
#     datetimeformat = "%Y-%m-%d %H:%M:%S"
#     epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
#     tdiff = utc -epoch  + datetime.timedelta(seconds=leapseconds)
#     gpsweek = tdiff.days // 7 
#     gpsdays = tdiff.days - 7*gpsweek         
#     gpsseconds = tdiff.seconds + 86400 * (tdiff.days -7*gpsweek) 
#     return (gpsweek,gpsdays,gpsseconds,tdiff.microseconds)

# def gmst(tut):
#     "Greenwich mean sidereal utc_time, in radians."
    
#     # ut1 = jdays2000(utc_time) / 36525.0
#     # theta = 67310.54841 + ut1 * (876600 * 3600 + 8640184.812866 + ut1 *
#     #                              (0.093104 - ut1 * 6.2 * 10e-6))
#     jday=2458484.833333
#     jday=jd.from_gregorian(2019,7,19,tut.hour,tut.minute,tut.second)
#     print (jday)
#     J0= floor(jday) + 0.5
#     dJ0= jday - J0
#     H = dJ0 * 24              
#     D = jday - 2451545.0     
#     T= D / 36525  
#     D0 = J0 - 2451545.0

#     gmst = 6.697374558 + 0.06570982441908 * D0+ 1.00273790935 * H+ 0.000026 * T**2 
#     gh=gmst % 24; gm=(gmst-floor(gmst))*60.;gs=(gm-floor(gm))*60.
#     gt=gh+gm/60+gs/3600
#     return gt#np.deg2rad(theta / 240.0) % (2 * np.pi)

#%%
# def kepler_orbit(i,sv_id,ti,tf,dt_nav,nt,param_nav,x,y,z):
    
#     t_oe = param_nav['Toe']
#     a_s = param_nav['sqrtA'] ** 2
#     e_s = param_nav['Eccentricity']
#     m_o = param_nav['M0']
#     delta_n = param_nav['DeltaN']
#     omega_dot = param_nav['OmegaDot']
#     omega = param_nav['omega']
#     omega_0 = param_nav['Omega0']
#     idot = param_nav['IDOT']
#     io = param_nav['Io']
#     crs = param_nav['Crs']
#     crc = param_nav['Crc']
#     cus = param_nav['Cus']
#     cuc = param_nav['Cuc']
#     cic = param_nav['Cic']
#     cis = param_nav['Cis']
    
#     # a_s=(5.15367114e+03)**2.
#     # e_s=4.56996483e-03
#     # m_o=-8.00150823e-01
#     # delta_n=4.44732811e-09
#     # io=9.57593511e-01
#     # idot=-5.76452583e-10
#     # omega=-6.23673898e-01
#     # omega_0=-5.13406698e-01
#     # omega_dot=-7.88425698e-09
#     # crs=-2.14687500e+01
#     # crc=2.08937500e+02
#     # cus=8.90716910e-06
#     # cuc=-1.06729567e-06
#     # cic=-1.75088644e-07
#     # cis=-1.02445483e-07
    
    
#     t=linspace(ti,tf+dt_nav,nt)
    
#     mu_ea=3.986005e+14 #m³/s²
#     Omega_dot_ea=7.2921151467e-05 #rad/s
#     for i_t in range(nt):
#         # t[i_t] = t_oe + i_t * dt_nav
#         delta_t=t[i_t]-t_oe
#         # print (delta_t)
#         if delta_t>302400:
#             delta_t=delta_t-2*302400
#         if delta_t<-302400:
#             delta_t=delta_t+2*302400
#         # print (delta_t)
        
#         n0=sqrt(mu_ea/a_s**3.)
#         m = m_o + (n0 + delta_n)*delta_t #Mean anomaly
        
#         cap_g = m#m + (e_s - e_s**3./8 + e_s**5/192 - e_s **7./ 9216.) * np.sin(m) #Eccentric anomaly
#         for iter in range (11):
#             cap_e=m+e_s*sin(cap_g)
#             cap_g=cap_e
#         angle_nu = np.arctan2(np.sqrt(1 - e_s ** 2) * np.sin(cap_e), np.cos(cap_e) - e_s) #Tru anomaly
#         phi = omega + angle_nu
            
#         delta_r = crs * np.sin(2 * phi) + crc * np.cos(2 * phi)
#         delta_u = cus * np.sin(2 * phi) + cuc*cos(2.*phi)
#         delta_i = cis * np.sin(2 * phi) + cic * np.cos(2 * phi)

#         phi =phi+ delta_u        
#         angle_inc =io + idot * delta_t+ delta_i
#         angle_long = omega_0 + omega_dot * delta_t - Omega_dot_ea * t[i_t] 
        
#         r = a_s * (1 - e_s*cos(cap_e)) + delta_r
#         x_op = r * np.cos(phi)
#         y_op = r * np.sin(phi)

#         x[i, i_t] = x_op * np.cos(angle_long) - y_op * np.cos(angle_inc) * np.sin(angle_long)
#         y[i, i_t] = x_op * np.sin(angle_long) + y_op * np.cos(angle_inc) * np.cos(angle_long)
#         z[i, i_t] = y_op * np.sin(angle_inc)
        
#     return (x,y,z,t)

# #%%
# # Función para calcular la trayectoria GPS
# def sat_traj(ti,tf,nt, dt_nav):
#     #fn='salu3451.21n'#
#     # fn='ABMF00GLP_R_20240010000_01D_MN.rnx.gz'#'brdc0010.24n.gz'#
#     fn='ELKO00USA_R_20182100000_01D_MN.rnx'
#     info = gr.rinexinfo(fn)
#     print ('NAVIGATION RINEX',info)
#     nav_data=gr.load(fn)
    
#     satellites = nav_data.sv.values#[18:19]
    
#     params = [
#         'Toe', 'SVclockBias', 'sqrtA', 'Eccentricity', 'M0', 'DeltaN',
#         'Io', 'IDOT', 'omega', 'Omega0', 'OmegaDot', 'Crs', 'Crc', 'Cus',
#         'Cuc','Cic', 'Cis'
#     ]
    
#     param_all=[]
#     for i, sv_id in enumerate(satellites):
        
#         try:
#             if not all(isinstance(sv_data[param].values.item(), (float, np.floating, int)) for param in params):
#                 raise ValueError("Datos de navegación faltantes o no numéricos")
            
#             sv_data= nav_data.sel(sv=sv_id).dropna(how='all', dim='time')
                            
#             param_nav = {param: sv_data[param].values[0] for param in params}
            
#             param_all.append([param_nav[param] for param in params])
            
#             # Calcular la posición del satélite en cada tiempo            
#             #x,y,z,t=kepler_orbit(i,sv_id,ti,tf,dt_nav,nt,param_nav,x,y,z)
            
#         except Exception as e:
#             print(f"No se pudo calcular la posición para el satélite {sv_id}: {e}")
#     x,y,z,t=kepler_orbit_all(ti,tf,dt_nav,nt,params,param_all)
    
#     return satellites, x, y, z, t
