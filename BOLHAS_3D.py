from pylab import *
from numpy import *
from scipy.signal import *
from scipy.special import erf
from matplotlib.signal_alam import wavelet,find_peaks
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import *
from mpl_toolkits.mplot3d import Axes3D

params={'axes.labelweight':'bold'};rcParams.update(params)
font={'family':'serif','weight':'bold','size':14};matplotlib.rc('font', **font);

global i_o,i_amb,i_pot,i_hemi,i_mag,i_inert,n_sor

i_hemi=1;i_mag=0;i_inert=1.;i_amb=1;i_pot=1;i_o=1.e-00;n_sor=11

#=============================================================================#
def ndf(x,mu,sig):
    return exp(-power((x-mu)/sig,2.)/2)

#=============================================================================#
def pdf(x):
    return exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2)))

def skew(x,mu,sig,a):
    t=(x-mu) / sig
    return pdf(t)*cdf(a*t)    
#=============================================================================#

def sim_vol():
    global r_ea,np,nf,nq,np_i,r,theta,dr,dtheta,dphi,iphi_m,iq_m
    global dp_m,df_m,dq_m,dp_i,df_i,dq_i
    global alt,alt_i,lat,lon,alt_3,lon_3,lat_3,angle_inc,b_p,b_q
    global y_f,z_f,y_sph,z_sph,x_sph,p,q,r_o,angle_I
    
    r_ea=6371.;
    
    dr=10.;
    alt=arange(0-0*r_ea,800+dr,dr);np=len(alt)
    alt_i=alt[15:];np_i=len(alt_i)
    
    dh=dr/2.;
    dphi=round(arctan(dh/(r_ea+300.)),5);
    lon=arange(-100.*dphi,100.*dphi+dphi,dphi);
    nf=len(lon)
    
    dtheta=2.*60.*dphi;
    lat=i_hemi*arange(-2*dtheta,2*dtheta+dtheta,dtheta);
    nq=len(lat)
    
    iphi_m=int(arange(0,nf).mean());iq_m=int(arange(0,nq).mean())
    print (lat)
    
    #=============================================================================#
    
    r=zeros((np,nf,nq));theta=zeros((np,nf,nq));phi=zeros((np,nf,nq))
    
    for i in range (0,np):
        r[i,:,:]=(alt[i]+r_ea)*1.e+03
    for j in range (0,nf):
        phi[:,j,:]=lon[j]
    for k in range (0,nq):
        theta[:,:,k]=lat[k]
    
    #======

#    delta=sqrt(1.+3.*(sin(theta))**2.)
#    r_e=r/r_ea
#    q=sin(theta)/r_e**2.
#    p=r_e/cos(theta)**2.
#    hq=r*r_e**2./delta
#    hp=(r/r_e)*cos(theta)**3./delta
#    hphi=r*cos(theta)

    r_m=r-0*r_ea*1.e+03
    r_o=r_m/(cos(theta))**2.
    delta=sqrt(1.+3.*(sin(theta))**2.)
    q=r_o**3.*sin(theta)/r_m**2.
    p=r_m/cos(theta)**2.
    hq=(r_m/r_o)**3./delta
    hp=cos(theta)**3./delta
    hphi=r_m*cos(theta)

    x_f=hphi*tan(phi)*1.e-03;y_f=(p*hp)*1.e-03;z_f=q*hq*1.e-03
    x_sph=x_f;y_sph=y_f*delta;z_sph=z_f*delta
    
    
    alt_3=r*1.e-03+0*y_sph-r_ea;alt_i=alt_3[np-np_i:,:,:];
    lat_3=degrees(arctan(z_f/(y_f+0.)))
    lon_3=degrees(arctan(x_f/(y_f+0.)))
        
    b_o=8.e+15         #Tm^3
    b_r=-2*b_o*sin(theta)/r**3.;b_theta=b_o*cos(theta)/r**3.
    b=b_o*delta/r**3.
    angle_inc=arccos(b_theta/b);angle_I=angle_inc[np-np_i:,:,:]
    b_p=b_r/b;b_q=b_theta/b
    
    #======

    r_o=r/(cos(theta))**2.
    delta=sqrt(1.+0*3.*(sin(theta))**2.)
    q=r_o**3.*sin(theta)/r**2.
    p=r/cos(theta)**2.
    hq=(r/r_o)**3./delta
    hphi=r*cos(theta)
    hp=cos(theta)**3./delta
    
    dp=abs(gradient(p)[0]+abs(i_mag*gradient(p)[2]))
    dq=abs(i_mag*gradient(q)[0]+(gradient(q)[2]))
    
    dp_m=dp*hp;dq_m=dq*hq;df_m=dphi*hphi   
    dp_i=dp_m[np-np_i:,:,:];df_i=df_m[np-np_i:,:,:];dq_i=dq_m[np-np_i:,:,:]
    
    print ('=======================================================================')
    print ('GRID_SIZES',dp_m.min(),dp_m.max(),dq_m.min(),dq_m.max(),df_m.min(),df_m.max())

#     fig=figure(1,figsize=(8,8),facecolor='khaki',edgecolor='k')
# #    subplot(221)
# #    plot(z_sph[0:np:10,iphi_m,:].T,angle_inc[0:np:10,iphi_m,:].T,'r')
# #    plot(z_sph[0:np:10,iphi_m,:].T,abs(theta[0:np:10,iphi_m,:]).T,'g')    

# #    subplot(223)
#     plot(z_sph[0:np:10,iphi_m,:].T,y_sph[0:np:10,iphi_m,:].T,'r')
#     plot(z_sph[:,iphi_m,0:nq:1],y_sph[:,iphi_m,0:nq:1],'g')
#     plot(z_f[0:np:10,iphi_m,:].T,y_f[0:np:10,iphi_m,:].T,'b')
#     labels=around(y_f[0:np:10,iphi_m,iq_m]-r_ea,decimals=0)
#     ii=0
#     for i in range (0,np,10):
#         text(z_f[i,0,iq_m],y_f[i,0,iq_m],str(int(labels[ii])),fontsize=8)
#         ii=ii+1

#     gca().add_patch(Circle((0,0),radius=r_ea,fc='gray',fill=False,alpha=1))
#     gca().add_patch(Circle((0,0),radius=r_ea-100,fc='w',fill=False,alpha=1))
#     axis('off')        
    return()

#=============================================================================#

def iono_amb():
    global mag_o, q_charge, omega_i,omega_e,den_amb
    
    q_charge=1.6e-19;b_c=1.38e-23;eps_o=8.854e-12;m_i=1.67e-27;z_i=16.
    mag_o=0.25e-04;omega_i=q_charge*mag_o/(z_i*m_i);r_li=600./omega_i;
    omega_e=-omega_i*1837.
    
    den_100=1.e+18;sc_h=40.;den_neu=den_100*exp(-(alt_i-100.)/sc_h)
#    den_neu[alt_i>600]=0*den_neu[alt_i>600]+den_neu[41,0,0]
    
    yp=2.e+02;enp=2.e+12;enpe=enp*1.e-02*1.e+01
    af=2.;bf=-5.;ae=5.;be=-60.#bf=-15
    r_n=(alt_i)/yp;r_of=1.;r_oe=0.255;r_ov=0.5#r_of=0.8
    
    den_f=enp/(exp(af*(r_n-r_of))+exp(bf*(r_n-r_of)))
    den_e=enpe/(exp(ae*(r_n-r_oe))+exp(be*(r_n-r_oe)))
    den_ef=5.e-01*enpe*(r_n+r_ov)**2./5.
    den_amb=den_f+0*den_e+0*den_ef
    
    t_i=800.;a=16.;#a is neutral mass in amu
    nu_ii=0.22*den_amb*1.e-06/(t_i)**(3./2.);
    nu_in=(0*nu_ii+2.6e-9*1.e-06*(den_neu+den_amb)*a**(-1./2.))/1.
    nu_ei=34*den_amb*1.e-06/(t_i)**(3./2.);
    nu_en=5.4e-16*den_neu*sqrt(t_i)+nu_ei

    omega_pl=sqrt(den_amb*q_charge**2./(eps_o*m_i*z_i))
        
    return (nu_in,nu_en)

#=============================================================================#
def mobility():        #(mu_p,-mu_h//mu_h,mu_p)
    
    f=iono_amb()

    nu_in=f[0];nu_en=f[1]
    
    gr_p=abs(up_tot*gradient(log(den_t))[0]/dp_i)
    for j in range (0,nf):
        for k in range (0,nq):
            gr_p[:,j,k]=gr_p[:,iphi_m,iq_m]
    tau=dt/1.
    omega=i_inert*(1./dt+2.*gr_p.max());nu_eff=nu_in+omega
    kappa=omega_i/nu_eff

    mu_p_i=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_i=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_i=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
    nu_eff=nu_en+omega
    kappa=omega_e/nu_eff
    
    mu_p_e=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_e=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_e=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
    return (mu_p_i,mu_h_i,mu_o_i,mu_p_e,mu_h_e,mu_o_e)
    
def conductivity(den_t):        #(s_p,-s_h//s_h,s_p)
    f=mobility()
    mu_p_i=f[0];mu_h_i=f[1];mu_o_i=f[2];
    mu_p_e=f[3];mu_h_e=f[4];mu_o_e=f[5];    

    sigma_p=q_charge*den_t*(mu_p_i-0*mu_p_e)
    sigma_h=q_charge*den_t*(mu_h_i-mu_h_e)
    sigma_o=q_charge*den_t*(mu_o_i-mu_o_e)
    
    return (sigma_p,sigma_h,sigma_o)

#=============================================================================#
    
def den_ion(t_e,den_t):
    den_o=den_t;den_g=den_t
    for i_sor in range (0,n_sor):
 
#        flux_p=den_o*up_tot;flux_f=den_o*uf_tot;flux_q=den_o*uq_tot;
#        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)   
#        div_flux_o=gr_p[0]/dp_i+gr_f[1]/df_i+0*i_o*gr_q[2]/dq_i
#        
#        flux_p=den_g*up_tot;flux_f=den_g*uf_tot;flux_q=den_g*uq_tot;
#        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)   
#        div_flux=gr_p[0]/dp_i+gr_f[1]/df_i+0*i_o*gr_q[2]/dq_i

        # up_t=up_tot
        # if t_e < 3000:up_t=up_tot-up_amb
        gr_p=gradient(log(den_o))
        div_flux_o=den_o*(up_tot*gr_p[0]/dp_i+uf_tot*gr_p[1]/df_i\
                          +0.*i_o*uq_tot*gr_p[2]/dq_i)
        gr_p=gradient(log(den_g))
        div_flux=den_g*(up_tot*gr_p[0]/dp_i+uf_tot*gr_p[1]/df_i\
                        +0.*i_o*uq_tot*gr_p[2]/dq_i)
        
        den_t=den_o-dt*(div_flux+div_flux_o)/2.
        # den_t[0,:,:]=den_o[0,:,:]
        delta_sor=(abs(den_g-den_t)/den_t).max()
        den_g=(den_g+den_t)/2.
        
        if delta_sor <1.e-05:
            print ('CONVERGENCE OF CONTINUITY',i_sor)
            break
    return (den_t)

#=============================================================================#
def drift(e_p,e_f,e_q):
    
    f=mobility()
    
    kappa=f[2]*mag_o
    mu_11=f[0];mu_12=-mu_11*kappa*cos(angle_I);mu_13=-mu_11*kappa*sin(angle_I)
    mu_21=-mu_12;mu_22=mu_11*(1.+kappa**2.*sin(angle_I)**2.);
    mu_23=-mu_11*kappa**2.*cos(angle_I)*sin(angle_I)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+kappa**2.*cos(angle_I)**2.)
    
    up_i=mu_11*e_p+mu_12*e_f+mu_13*e_q
    uf_i=mu_22*e_f+mu_21*e_p+mu_23*e_q
    uq_i=mu_33*e_q+mu_31*e_p+mu_32*e_f


    kappa=f[5]*mag_o
    mu_11=f[3];mu_12=-mu_11*kappa*cos(angle_I);mu_13=-mu_11*kappa*sin(angle_I)
    mu_21=-mu_12;mu_22=mu_11*(1.+kappa**2.*sin(angle_I)**2.);
    mu_23=-mu_11*kappa**2.*cos(angle_I)*sin(angle_I)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+kappa**2.*cos(angle_I)**2.)
    
    up_e=mu_11*e_p+mu_12*e_f+mu_13*e_q
    uf_e=0*mu_22*e_f+mu_21*e_p+0*mu_23*e_q
    uq_e=mu_33*e_q+mu_31*e_p+mu_32*e_f
    
    return (up_i,uf_i,uq_i,up_e,uf_e,uq_e)

#%%
data_ele=loadtxt('Deriva_SL_2015.dat')
dia=data_ele[:,0];mes=data_ele[:,1];ano=data_ele[:,2]
hora=data_ele[:,3];minutos=data_ele[:,4];
deriva=data_ele[:,5]
tempo=hora+minutos/60.
vel_mean=[]
for i in range (144):
    vel_mean.append((i*10./60.,deriva[tempo==i*10/60.].mean()))
vel_mean=array(vel_mean)
date_ele=18
x0, y0 = tempo[dia==date_ele], deriva[dia==date_ele]
x0,y0=vel_mean[:,0],vel_mean[:,1]
x,y=x0[x0>18],y0[x0>18]
t_ele=x*3600.;vel_amb=y;
t_ele=t_ele-t_ele[0]
f=wavelet(0,t_ele,vel_amb,32)
pd=f[0];pwr=f[1];emd=f[2];amp_sism=f[3][:,1][0]
fr_sism=1.e+00/(pd)
def campo_amb(t_e):
    vel_ana=0
    for iw in range (len(pd)):
        idx_peaks=find_peaks(abs(emd[iw,:]))[:,0]
        n_peaks=len(idx_peaks)
        t0=t_ele[idx_peaks]
        amp=emd[iw,idx_peaks]
        sigma_t=pd[iw]
        [tn,tn0]=meshgrid(t_e,t0)
        [tn,ampn]=meshgrid(t_e,amp)
        f_sism=skew(tn,tn0,sigma_t/2.,0)
        vel_ana=vel_ana+(f_sism*ampn).sum(0)
        # idx_peaks=find_peaks(abs(emd[iw,:]))[:,0]
        # for i_peak in range (len(idx_peaks)):
        #     t0=t[idx_peaks[i_peak]];
        #     # print (len(idx_peaks),t0)
        #     sigma_t=pd[iw]
        #     f_sism=skew(t-t_phase,t0,sigma_t/2.,0)
        #     f_sism=f_sism-f_sism[0]
        #     amp=emd[iw,idx_peaks[i_peak]]
        #     vel_ana=vel_ana+f_sism*amp
    #vel_ana=vel_amb.max()*vel_ana/emd.max()
    # fig=figure(figsize=(12,12),facecolor='w',edgecolor='k')
    # ax=subplot(111)
    # plot(t/60.,vel,'r',lw=1)
    # plot(t/60.,vel_ana,'b')
    return (vel_ana)
#=============================================================================#    

def campo_pol(t_e,wp_g,wf_g,wq_g,den_t,pot):
    global up_tot,uf_tot,uq_tot
    
#===
    f=conductivity(den_t)
    sigma_p=f[0];sigma_h=f[1];sigma_o=f[2];
#    sigma_o=1.e+03*sigma_p#+sigma_o.min()
    
    s_11=sigma_p;s_12=-sigma_h*cos(angle_I);s_13=-sigma_h*sin(angle_I)
    s_21=-s_12;s_22=sigma_p*cos(angle_I)**2.+sigma_o*sin(angle_I)**2.;
    s_23=(sigma_p-sigma_o)*sin(angle_I)*cos(angle_I)
    s_31=-s_13;s_32=s_23;s_33=sigma_o*cos(angle_I)**2.+sigma_p*sin(angle_I)**2.    
    
#===WIND DRIVEN DRIFs/CURRENTs
    
    e_p=wf_g*mag_o;e_f=-wp_g*mag_o;e_q=wq_g*mag_o
    
    f_d=drift(e_p,e_f,e_q)
    up_agw_i=f_d[0];uf_agw_i=f_d[1];uq_agw_i=f_d[2]
    up_agw_e=f_d[3];uf_agw_e=f_d[4];uq_agw_e=f_d[5]

    cur_wind_p=den_t*q_charge*(up_agw_i-up_agw_e)
    cur_wind_f=den_t*q_charge*(uf_agw_i-uf_agw_e)
    cur_wind_q=den_t*q_charge*(uq_agw_i-uq_agw_e)   
    
    
    div_cur_wind=gradient(cur_wind_p)[0]/dp_i+gradient(cur_wind_f)[1]/df_i+\
                gradient(cur_wind_q)[2]/dq_i
    source_wind=-div_cur_wind/sigma_p
    
#===GRAVITY DRIVEN DRIFTS/CURRENTS
    
    gr=i_amb*9.8;gr_p=-gr;gr_f=0;gr_q=0
    fac=mag_o/omega_i
    e_p=fac*gr_p;e_f=fac*gr_f;e_q=fac*gr_q;
    
    f_d=drift(e_p,e_f,e_q)
    up_gr_i=f_d[0];uf_gr_i=f_d[1];uq_gr_i=f_d[2]

    fac=mag_o/omega_e
    e_p=fac*gr_p;e_f=fac*gr_f;e_q=fac*gr_q;
    f_d=drift(e_p,e_f,e_q)
    up_gr_e=f_d[3];uf_gr_e=f_d[4];uq_gr_e=f_d[5]
    
    cur_gr_p=den_t*q_charge*(up_gr_i-up_gr_e)
    cur_gr_f=den_t*q_charge*(uf_gr_i-uf_gr_e)
    cur_gr_q=den_t*q_charge*(uq_gr_i-uq_gr_e)   
    
#===AMBIENT ELECTRIC FIELD DRIVEN CURRENTS
    
    t_o=8000.;sigma_t=4000.;
    ele_t=skew(t_e,t_o,sigma_t,-3)
    ele_amb=-1.5e-03*ele_t
    wl_alt=alt_3[0,iphi_m,iq_m]-alt_3[-1,iphi_m,iq_m]
    sigma=0.5*wl_alt;
    gauss_alt=1+0*skew(alt_i,300.,sigma/sqrt(2.),0)
    ele_amb=gauss_alt*ele_amb/2.
    #ele_amb=ele_amb+0.6e-03*skew(t_e,t_o/1.1,sigma_t/2.,-1)
    ele_amb=ele_amb+0.8e-03*skew(t_e,t_o/1.1,sigma_t/2.,-1)
    
    vel_ana=campo_amb(t_e)
    ele_amb=-vel_ana*mag_o
    print (vel_ana.max())
    ele_amb_p=i_amb*0;ele_amb_f=i_amb*ele_amb;ele_amb_q=i_amb*0
    f_d=drift(ele_amb_p,ele_amb_f,ele_amb_q)
    up_amb=f_d[3];uf_amb=f_d[4];uq_amb=f_d[5];

    cur_ele_p=s_11*ele_amb_p+s_12*ele_amb_f+s_13*ele_amb_q
    cur_ele_f=s_22*ele_amb_f+s_21*ele_amb_p+s_23*ele_amb_q
    cur_ele_q=s_33*ele_amb_q+s_31*ele_amb_p+s_32*ele_amb_f
    
#===TOTAL AMBIENT CURRENT/POTENTIAL SOURCE 
    
    cur_amb_p=cur_gr_p+cur_ele_p
    cur_amb_f=cur_gr_f+cur_ele_f
    cur_amb_q=cur_gr_q+cur_ele_q
            
    div_cur_amb=0*gradient(cur_amb_p)[0]/dp_i+gradient(cur_amb_f)[1]/df_i+\
                0*gradient(cur_amb_q)[2]/dq_i
    source_amb=-1.*div_cur_amb/s_11 #2.5
#    fac=1./(sigma_p*den_t)
#    cur_amb_f=sigma_p*ele_amb_f+den_t*q_charge*gr_p/omega_i
#    source_amb=-fac*cur_amb_f*gradient(den_t)[1]/df_i
    
#===>DIVERGENCE FREE CURRENT EQUATION
    
    df_map2=(s_11/s_22)*df_i**2.
    dq_map2=(s_11/s_33)*dq_i**2.
    
    a1=gradient(s_11)[0]/dp_i+gradient(s_21)[1]/df_i+i_o*gradient(s_31)[2]/dq_i
    a1=a1/s_11
    a2=gradient(s_12)[0]/dp_i+gradient(s_22)[1]/df_i+i_o*gradient(s_32)[2]/dq_i
    a2=a2/s_11
    a3=gradient(s_13)[0]/dp_i+gradient(s_23)[1]/df_i+i_o*gradient(s_33)[2]/dq_i
    a3=a3/s_11
   
    pot_g=pot
    w_iter=1.7
    for i_sor in range (0,n_sor):   
        #===
        gr_pot=gradient(pot_g)
        grad_pot_p=gr_pot[0]/dp_i;
        grad_pot_f=gr_pot[1]/df_i;
        grad_pot_q=gr_pot[2]/dq_i
        
        pot_1=a1*grad_pot_p;
        pot_2=a2*grad_pot_f;
        pot_3=a3*grad_pot_q
        
        gr_pot_pf=gradient(grad_pot_p)[1]/df_i
        gr_pot_pq=gradient(grad_pot_p)[2]/dq_i
        gr_pot_fp=gradient(grad_pot_f)[0]/dp_i
        gr_pot_fq=gradient(grad_pot_f)[2]/dq_i
        gr_pot_qp=gradient(grad_pot_q)[0]/dp_i
        gr_pot_qf=gradient(grad_pot_q)[1]/df_i
        
        pot_cross_1=s_12*gr_pot_fp+s_21*gr_pot_pf
        pot_cross_3=s_13*gr_pot_qp+s_31*gr_pot_pq
        pot_cross_2=s_23*gr_pot_qf+s_32*gr_pot_fq
        pot_cross=(pot_cross_1+pot_cross_2+pot_cross_3)/s_11
        
        a_p=(gradient(gr_pot[0])[0]+2.*pot_g)/dp_i**2.
        a_f=(gradient(gr_pot[1])[1]+2.*pot_g)/df_map2
        a_q=(gradient(gr_pot[2])[2]+2.*pot_g)/dq_map2
                        
        a=2.*(1./dp_i**2.+1./df_map2+i_o*1./dq_map2)
        
        pot=i_pot*(1./a)*(a_p+a_f+i_o*a_q+pot_1+pot_2+i_o*pot_3+\
                  i_o*pot_cross+0*source_wind+source_amb)
        
        delta_sor=abs(pot_g-pot).max()/pot.max()        
        pot_g=w_iter*pot+(1.-w_iter)*pot_g#(pot_g+pot)/2.        
        if delta_sor <1.e-02:
            print ('CONVERGENCE OF LAPLACE',i_sor)
            break
        
        w_iter=w_iter-0.1
        if w_iter<=1:
            w_iter=1.
                
#===POLARIZATION ELECTRIC FIELD
    
    ele_p=-1.*gradient(pot)[0]/dp_i;
    ele_f=-1.*gradient(pot)[1]/df_i;
    ele_q=-i_o*gradient(pot)[2]/dq_i
    
    # if t_e >=t_o/2.:
    #     ele_f=-1*gradient(pot)[1]/df_i;
        
#===TOTAL ELECTRIC FIELD
    
    ele_tot_p=ele_amb_p+ele_p;
    ele_tot_f=ele_amb_f+ele_f;
    ele_tot_q=ele_amb_q+ele_q;

    # if abs(up_tot).max()<2.e+01:
    #     ele_tot_p=ele_p;ele_tot_f=ele_f;ele_tot_q=ele_q;

#===TOTAL ELECTRON DIRFT    

    f_d=drift(ele_tot_p,ele_tot_f,ele_tot_q)
    up_ele=f_d[3];uf_ele=f_d[4];uq_ele=f_d[5];

#    f_d=drift(ele_tot_p,ele_f,ele_tot_q)
#    uf_ele=f_d[4]

#    up_ele=-ele_tot_f/mag_o
#    uf_ele=ele_p/mag_o
    
#===>TOTAL DRIFT+WIND DRIFT
    
    up_tot=up_ele;uf_tot=uf_ele;uq_tot=uq_ele;    
    if t_e==0:
        up_tot=up_ele+up_agw_i;uf_tot=uf_ele+uf_agw_i;uq_tot=uq_ele+uq_agw_i

    print ('AMBIENT, U_amb',round(up_amb.max(),2))
    print ('POTENTIAL, U_MAX', round(up_tot.max(),2),round(uf_tot.max(),2),round(uq_tot.max(),2))

    # fig=figure(2,figsize=(12,12),facecolor='khaki',edgecolor='k')  
    # plot(t_e,up_amb.max(),'ro')
    # plot(t_e,up_tot[:,:,iq_m].max(),'bo')
    # draw()
    return (pot,up_ele,up_amb)

#=============================================================================#
#==================================MAIN=======================================#

global dt,time
#nt=10000;t=zeros((nt));
time=0

f=sim_vol()

wp=zeros((np,nf,nq));wf=zeros((np,nf,nq));wq=zeros((np,nf,nq));
up_tot=zeros((np_i,nf,nq));uf_tot=zeros((np_i,nf,nq))

wl_full=lon_3[0,-1,iq_m]-lon_3[0,0,iq_m]
sigma=0.25*wl_full;
amp_gauss=ndf(lon_3,lon[iphi_m],sigma/sqrt(2.))
n_w=3.*pi/2.;wl=wl_full/n_w;wk=2.*pi/wl;
wind_pert=amp_gauss*cos(wk*(lon_3-lon[iphi_m]));
wl_alt=alt_3[0,iphi_m,iq_m]-alt_3[-1,iphi_m,iq_m]
sigma=0.25*wl_alt;
gauss_alt=1+0*ndf(alt_3,250.,sigma/sqrt(2.))
wp[:,:,:]=gauss_alt*wind_pert[:,:,:]

f=iono_amb()    
den_t=den_amb*(1.-0.1*wp[np-np_i:,:,:]);
#den_t[alt_i>500]=den_amb[alt_i>500]
pot=zeros((np_i,nf,nq));
wp_i=0*wp[np-np_i:,:,:];wf_i=wf[np-np_i:,:,:];wq_i=wq[np-np_i:,:,:]

data_evol=[]
t4=[];d4=[];u4_0=[];u4=[]
i_t=0    

while (up_tot[:,iphi_m,iq_m].max()<500 and time<3600*5.):

    up_max=up_tot[:,iphi_m,iq_m].max()
    dt=(0.005*dp_i.min()/max(up_max,30.))
    if dt <0.1:dt=0.1;
    
    #===========================IONOSPHERE====================================#
    
    f=campo_pol(time,wp_i,wf_i,wq_i,den_t,pot)    
    pot=f[0];up_ele=f[1];up_amb=f[2]
    
    den_t=den_ion(time,den_t)
    delta_n=100*abs(den_t[:,iphi_m,iq_m]-den_t[:,0,iq_m])/den_t[:,11,iq_m]
    delta_tec=100*(den_t.mean(0)-den_t[:,0,iq_m].mean(0))/den_t.mean(0)
    
    gr_rate=up_max*abs(gradient(log(den_amb))[0]/dp_i).max()
    gr_time=gr_rate*(time)
    
    #data_evol.append([time, gr_time, delta_n.max(), up_amb.max(), up_max])
    if remainder(i_t,20)==0:
        t4.append(time)
        d4.append(den_t)
        u4.append(up_tot)
        u4_0.append(up_amb)
    
    # t[i_t+1]=t[i_t]+dt;
    time=time+dt
    i_t=i_t+1
    
    print ('======================================================')
    print ("TIME=", i_t,dt,time)
    print ('Equatorial Growth time=',gr_time)
    print ('Equatorial Depletion velocity=',up_tot[:,iphi_m,iq_m].max())
    print ('PLOTTING')
    #=========================================================================#
    #------------------------------PLOTTING-----------------------------------#
    #=========================================================================#

#    fig = figure(1,figsize=(10,10),facecolor='khaki',edgecolor='k')
#    subplot(121)
#    plot(t[i_t],delta_n.max(),'o')
#    plot(0*arange(0,200)+1./gr_rate,arange(0,200))
#    axis((0,1.5e+04,0,2.e+02))
#    subplot(122)
#    plot(t[i_t],up_amb.max(),'ro');    
#    plot(t[i_t],up_max,'bo');
#    axis((0,1.5e+04,0,2.e+03))
#    draw();
##    
   
#     if remainder(i_t,10)==0 and up_tot.max()<1500:
#         fig=figure(3,figsize=(12,12),facecolor='khaki',edgecolor='k')   
#         clf()
#         vm=(up_tot[:,:,iq_m]-0*up_amb[:,:,iq_m]).max()
#         iq=iq_m-4
#         for i in range (1,4):
#             ax=subplot(1,3,i);subplots_adjust(wspace=0, hspace=0.)
#             lon_2=lon_3[np-np_i:,:,iq];alt_2=y_sph[np-np_i:,:,iq]-r_ea;
#             alt_2=alt_3[np-np_i:,:,iq];lat_2=lat_3[np-np_i:,:,iq];
#             data=den_t[:,:,iq];
#             contour(zoom(lon_2,2),zoom(alt_2,2),zoom(data,2),11,cmap=cm.jet);
#             data=up_tot[:,:,iq]-up_amb[:,:,iq]
#             im=pcolormesh(zoom(lon_2,2),zoom(alt_2,2),zoom(data,2),
#                           cmap=cm.seismic,vmax=vm,vmin=-vm)
#             axis((lon_2.min()/1.,lon_2.max()/1.,alt_2.min(),alt_2.max()))
#             tit_str=str.join('',[str(int(lat_2.max())),',',str(int(t[i_t])),',',str(round(data.max(),2))])
#             title(tit_str)
# #            title(str(degrees(angle_inc[0,101,iq])))
# #            if i > 1:
# #                gca().set_yticklabels([])
#             iq=iq+2
    
    
# #    divider=make_axes_locatable(ax);cax=divider.append_axes("right",size="2%",pad=0.1)
# #    colorbar(im,cax=cax);gca().set_title('m/s',fontsize=10)
#         draw();pause(0.1);

#         fig=figure(4,figsize=(8,8),facecolor='khaki',edgecolor='k') 
#         lon_2=lon_3[np-np_i+10,:,:];lat_2=lat_3[np-np_i+10,:,:];
#         data=delta_tec#up_tot[40,:,:]-up_amb[40,:,:]
#         vm=99#data.max()
#         im=pcolormesh(zoom(lon_2,2),zoom(lat_2,2),zoom(data,2),cmap=cm.seismic,vmax=vm,vmin=-vm)
#         draw();pause(0.1);
#    #===
        
#    fig=figure(3,figsize=(12,12),facecolor='w',edgecolor='k') 
#    ax=fig.gca(projection='3d')
#    vm=1.
#    i_alt=0
#    for i in range (0,5):
#        nalt=i_alt
#        data=pot[:,i_alt,:]
#        ax.contourf(lat_3[np-np_i:,nalt,:],alt_3[np-np_i:,nalt,:],
#                    lon_3[np-np_i:,nalt,:]+0.1*data,
#                    zdir='z',alpha=0.5)
#        i_alt=i_alt+50
#        
##    ax.set_zlim3d(0,500)
#    ax.view_init(10,45)
#    draw();pause(0.1);clf();
#====

t4=array(t4)
u4=array(u4)
u4_0=array(u4_0)
d4=array(d4)

save('alt_3.npy',alt_3)
save('lat_3.npy',lat_3)
save('lon_3.npy',lon_3)
save('t4_2015_quiet.npy',t4)
save('d4_2015_quiet.npy',d4)
save('u4_2015_quiet.npy',u4)
save('u4_0_2015_quiet.npy',u4_0)


fig=figure(figsize=(8,8),facecolor='w',edgecolor='k')
plot(t4,u4[:,:,iphi_m,iq_m].max(1),'ro') 
plot(t4,u4_0[:,:,iphi_m,iq_m].max(1),'bo') 

fig=figure(figsize=(8,8),facecolor='w',edgecolor='k')
i0=-2
vm=(u4[i0,:,:,iq_m]-0*u4_0[i0,:,:,iq_m]).max()
iq=iq_m-4
for i in range (1,4):
    ax=subplot(1,3,i);subplots_adjust(wspace=0, hspace=0.)
    lon_2=lon_3[np-np_i:,:,iq];alt_2=y_sph[np-np_i:,:,iq]-r_ea;
    alt_2=alt_3[np-np_i:,:,iq];lat_2=lat_3[np-np_i:,:,iq];
    data=d4[i0,:,:,iq];
    contour(zoom(lon_2,2),zoom(alt_2,2),zoom(data,2),11,cmap=cm.jet);
    data=u4[i0,:,:,iq]-u4_0[i0,:,:,iq]
    im=pcolormesh(zoom(lon_2,2),zoom(alt_2,2),zoom(data,2),
                  cmap=cm.seismic,vmax=vm,vmin=-vm)
    axis((lon_2.min()/1.,lon_2.max()/1.,alt_2.min(),600))
    tit_str=str.join('',[str(int(lat_2.max())),',',str(int(t4[i0])),',',str(round(data.max(),2))])
    title(tit_str)
#            title(str(degrees(angle_inc[0,101,iq])))
#            if i > 1:
#                gca().set_yticklabels([])
    iq=iq+2

# fig=figure(figsize=(8,8),facecolor='w',edgecolor='k')

# plot(z_sph[0:np:10,iphi_m,:].T,y_sph[0:np:10,iphi_m,:].T,'r')
# plot(z_sph[:,iphi_m,0:nq:1],y_sph[:,iphi_m,0:nq:1],'g')
# plot(z_f[0:np:10,iphi_m,:].T,y_f[0:np:10,iphi_m,:].T,'b')
# labels=around(y_f[0:np:10,iphi_m,iq_m]-r_ea,decimals=0)
# ii=0
# for i in range (0,np,10):
#     text(z_f[i,0,iq_m],y_f[i,0,iq_m],str(int(labels[ii])),fontsize=8)
#     ii=ii+1

# # gca().add_patch(Circle((0,0),radius=r_ea,fc='gray',fill=False,alpha=1))
# # gca().add_patch(Circle((0,0),radius=r_ea-100,fc='w',fill=False,alpha=1))
# axis('off')        
# data_evol_arr=array(data_evol)    
# savetxt('Time_evol_trial.out',data_evol_arr)    
# save('den_trial.npy',den_t[:,:,iq_m])
# save('vel_trial.npy',up_tot[:,:,iq_m]-up_amb[:,:,iq_m])
show()


#=============================================================================#
def pois_equ_lhs(data):
    #===
    f=conductivity(den_t)
    sigma_p=f[0];sigma_h=f[1];sigma_o=f[2];
    
    s_11=sigma_p;s_12=-sigma_h*cos(angle_I);s_13=-sigma_h*sin(angle_I)
    s_21=-s_12;s_22=sigma_p*cos(angle_I)**2.+sigma_o*sin(angle_I)**2.;
    s_23=(sigma_p-sigma_o)*sin(angle_I)*cos(angle_I)
    s_31=-s_13;s_32=s_23;s_33=sigma_o*cos(angle_I)**2.+sigma_p*sin(angle_I)**2.    
    
    df_map2=(s_11/s_22)*df_i**2.
    dq_map2=(s_11/s_33)*dq_i**2.
    
    a1=gradient(s_11)[0]/dp_i+gradient(s_21)[1]/df_i+i_o*gradient(s_31)[2]/dq_i
    a1=a1/s_11
    a2=gradient(s_12)[0]/dp_i+gradient(s_22)[1]/df_i+i_o*gradient(s_32)[2]/dq_i
    a2=a2/s_11
    a3=gradient(s_13)[0]/dp_i+gradient(s_23)[1]/df_i+i_o*gradient(s_33)[2]/dq_i
    a3=a3/s_11
   
    gr_pot=gradient(data)
    grad_pot_p=gr_pot[0]/dp_i;
    grad_pot_f=gr_pot[1]/df_i;
    grad_pot_q=gr_pot[2]/dq_i
    
    pot_1=a1*grad_pot_p;
    pot_2=a2*grad_pot_f;
    pot_3=a3*grad_pot_q
    
    gr_pot_pf=gradient(grad_pot_p)[1]/df_i
    gr_pot_pq=gradient(grad_pot_p)[2]/dq_i
    gr_pot_fp=gradient(grad_pot_f)[0]/dp_i
    gr_pot_fq=gradient(grad_pot_f)[2]/dq_i
    gr_pot_qp=gradient(grad_pot_q)[0]/dp_i
    gr_pot_qf=gradient(grad_pot_q)[1]/df_i
    
    pot_cross_1=s_12*gr_pot_fp+s_21*gr_pot_pf
    pot_cross_3=s_13*gr_pot_qp+s_31*gr_pot_pq
    pot_cross_2=s_23*gr_pot_qf+s_32*gr_pot_fq
    pot_cross=(pot_cross_1+pot_cross_2+pot_cross_3)/s_11
    
    a_p=(gradient(gr_pot[0])[0]+2.*data)/dp_i**2.
    a_f=(gradient(gr_pot[1])[1]+2.*data)/df_map2
    a_q=(gradient(gr_pot[2])[2]+2.*data)/dq_map2
                    
    a=2.*(1./dp_i**2.+1./df_map2+i_o*1./dq_map2)
    
    pois_lhs_II=i_pot*(a_p+a_f+i_o*a_q+pot_1+pot_2+i_o*pot_3+i_o*pot_cross)
    
    pois_eq=a*data-i_pot*(a_p+a_f+i_o*a_q+pot_1+pot_2+i_o*pot_3+i_o*pot_cross)
        
    return (a, pois_lhs_II, pois_eq)
 
def campo_pol_desc():   
    pot_g=pot
    i=0
    imax=10
    eps=0.01
    b=source_amb;
    f=pois_eq_lhs(pot_g);
    r=b-f[2]
    r_tr=transpose(r,(0,2,1))
    delta=dot(r_tr,r)
    delta0=delta
    while i<imax:
        f=pois_eq_lhs(r);
        alpha=float(delta/dot(r_tr,f[2]))
        
        pot_g= pot_g+alpha*r
        
        f=pois_eq_lhs(pot_g);
        r=b-f[2]
        r_tr=transpose(r,(0,2,1))
        delta=dot(r_tr,r)
        
        i +=1
        
    pot=pot_g;
    
    return pot

#i_hemi=float(raw_input('HEMISPHERE, +1 FOR SOUTHERN '))   #+1 for S, -1 for N
#i_mag=float(raw_input('MAGNETIC OR SPHERICAL, 0 FOR SPHERICAL '))   #1 for magnetic, 0 for spherical
#i_inert=float(raw_input('INERTIA OR COLLISIONAL, 0 FOR COLLISIONAL '))   #1 for INERTIA, 0 for COLLISIONAL
#i_amb=float(raw_input('AMBIENT FORCE, 0 FOR NO AMBIENT '))
#i_pot=float(raw_input('POLARIZATION POTENTIAL, 1 FOR SOLVING POTENTIAL '))    #
#i_o=float(raw_input('PARALLEL DYNAMICS, 1 FOR YES '))
