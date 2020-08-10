from numpy import *
from pylab import *
from scipy.signal import *
from scipy.interpolate import *

def extrema (ns,t,data):
    grad_data=gradient(data);
    peak_pos=[];peak_neg=[];i_max=0;i_min=0
    for i in range (0,ns,1):
        if i==0 or i==ns-1:
            if data[i]>0:peak_pos.append([t[i],data[i]])
            if data[i]<0:peak_neg.append([t[i],data[i]])
        else:
            temp=grad_data[i-1]*grad_data[i+1]
            if temp < 0 and grad_data[i-1]>=0:
                i_max=i_max+1
                peak_pos.append([t[i],data[i]])
            if temp < 0 and grad_data[i-1]<0:
                i_min=i_min+1
                peak_neg.append([t[i],data[i]])
#    print "NUMEROS DAS EXTREMAS", i_max,i_min
    t_max=t;t_min=t;data_max=data;data_min=data
    if i_max>3:
#        print "EXISTE MAXIMA"
        peak_pos=array(peak_pos);t_max=peak_pos[:,0];data_max=peak_pos[:,1]
        t_new=linspace(t_max[0],t_max[-1],ns);
        data_max=interp1d(t_max,data_max,kind='cubic')(t_new);t_max=t_new
    if i_min>3:
#        print "EXISTE MINIMA"
        peak_neg=array(peak_neg);t_min=peak_neg[:,0];data_min=peak_neg[:,1]
        t_new=linspace(t_min[0],t_min[-1],ns);
        data_min=interp1d(t_min,data_min,kind='cubic')(t_new);t_min=t_new

    return (t_max,data_max,t_min,data_min,min(i_max,i_min))    

def hht(ns,dt,t_l,t_u,t,data):
    ns_h=int(ns/2.);
    freqs=linspace(1./t.max(),1/dt,ns);
    data_filt=0*data
    data_g=data;
    modes=zeros((ns,ns_h))

    for i_mode in range (0,ns_h):
        data_p=data_g
        for i_shift in range (0,11):
            f=extrema(ns,t,data_g)
            if f[4]<4:
    #            print 'SHIFTING IS SATISFIED',i_shift,i_mode
                break
            else:
    #            print 'SHIFTING IS NOT SATISFIED',i_shift,i_mode
                t_max=f[0];data_max=f[1];t_min=f[2];data_min=f[3]
                t_mean=(t_max+t_min)/2;data_mean=(data_max+data_min)/2.
                data_g=data_g-data_mean;
    #            print data.max(),data.min()            
        modes[:,i_mode]=data_g
        data_g=data_p-data_g;
        error=abs(data_g-data_p).max()/data_p.max()
        if error<5.e-06 or i_shift==0:
            print ("TOTAL NUMBER OF MODES=",i_mode+1)
            break
    n_mode=i_mode+1
    print ('Number of modes=',n_mode)
    modes=modes[:,0:n_mode];

    for i_mode in range (0,n_mode):    
        w=abs(fft(modes[:,i_mode]));
        w_half=w[0:ns_h];i_p=w_half.argmax();t_p=around(1./freqs[i_p],2)
        if t_p >t_l and t_p < t_u:
            data_filt=data_filt+modes[:,i_mode]
        
    return (modes,data_filt)
    
def fourier(ns,dt,t_l,t_u,t,data):
    ns_h=int(ns/2.);
    freqs=linspace(1./t.max(),1/dt,ns);
    data_filt=0*data
    w=abs(fft(data));#w[w <1.e-01*w.max()]=1.e-06
    w_half=w[0:ns_h+1]
    
    data_peak=[];freq_p=0
    for i in range (0,ns_h):
        amp=w_half[i];phase=angle(w_half[i])
    #    data_peak.append([i,freqs[i],amp,phase])
        i_p=w_half.argmax()
        w_half[i_p]=-1;ratio=(w[i_p]-w[i_p-1])/(w[i_p]-w[i_p+1])
        if ratio>0.:
            w_ip=abs(fft(data))
            for j in range (0,1):
                i_pp=i_p-int(1/2.)+j
                amp=w_ip[i_pp];phase=angle(w_ip[i_pp])
                data_peak.append([i_pp,freqs[i_pp],amp,phase])

    peak_data=array(data_peak)
    n_mode=len(peak_data[:,0])
    #=====
    error_m=0*t_mirror;
    modes=zeros((ns,n_mode));
    
    for i_ter in range (0,21):
        for i in range (0,n_mode):
            omega=peak_data[i,1];
            amp=(1.+error_m)*peak_data[i,2];phase=(1.+error_m)*peak_data[i,3]
            modes_sine=amp*cos(omega*t+phase)
            amp_modes=(modes_sine/modes_sine.max()-data/data.max())
            amp_modes=error_m-abs(amp_modes)
            modes[:,i]=amp_modes*modes_sine/amp
           
        data_new=modes.sum(1)
        data_new=data.max()*data_new/data_new.max()
        error_p=error_m.mean()  
        error=abs(data_new-data)  
        error_m=error/error.max()
        print (error_m.mean())
        if error_m.mean()<error_p/2.:
            break

    for i_mode in range (0,n_mode):    
        w=abs(fft(modes[:,i_mode]));
        w_half=w[0:ns_h];i_p=w_half.argmax();t_p=around(1./freqs[i_p],2)
        if t_p >t_l and t_p < t_u:
            data_filt=data_filt+modes[:,i_mode]

    return (modes,data_filt)

    
#dt=1.;t=arange(0,500,dt);ns=len(t);
#data=0*t
#n_mode=11
#for i_mode in range (0,n_mode):
#    tau=2.*dt*(1+i_mode);omega=2.*pi/tau
#    amp=(1+i_mode)**0.
#    data=data+amp*cos(omega*t)
#    
#t_l=120;t_u=140
#
#f=hht(ns,t,t_l,t_u,data)
#modes=f[0];data_filt=f[1]
#        
#fig = figure(1,figsize=(16,8),facecolor='w',edgecolor='k') 
#plot(t,data,'r');#plot(t_max,data_max,'g');plot(t_min,data_min,'b')
#plot(t,modes.sum(1),'b')
#plot(t,data_filt,'g')
#
#fig = figure(2,figsize=(16,16),facecolor='w',edgecolor='k') 
#for i_mode in range (0,n_mode):
#    subplot(n_mode,1,i_mode+1)
#    plot(t,modes[:,i_mode])
#    
#freqs=linspace(1./t.max(),1/dt,ns);freqs=freqs[0:ns_h]
#fig = figure(3,figsize=(16,16),facecolor='w',edgecolor='k') 
#for i_mode in range (0,n_mode):    
#    w=abs(fft(modes[:,i_mode]));
#    w_half=w[0:ns_h];i_p=w_half.argmax();period=around(1./freqs[i_p],2)
#    subplot(n_mode,1,i_mode+1)
#    plot(1./freqs,0.5*w_half/w_half.max(),label=str(period))
#    legend()
#
show()
