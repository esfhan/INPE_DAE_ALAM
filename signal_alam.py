from scipy.ndimage import *
import numpy as np
from pylab import *
from scipy import fftpack
from scipy.signal import *

#%%
def shift(data,n):
    e = empty_like(data)
    if n >= 0:
        e[:n] = data[n]
        e[n:] = data[:-n]
    else:
        e[n:] = data[n]
        e[:n] = data[-n:]
    return e

#%%
def find_peaks(data):
    ns=len(data)
    gr_d=gradient(data);
    a=shift(gr_d,1)*shift(gr_d,-1)
    peak_values=a[a<0]
    peak_number=len(peak_values)
    peak_pos=argwhere(a<0)[::2]
    return peak_pos

    
#%%
def cross_correlate(t_all,y_all,data_all):
#t_all is in hours
#y_all is in degrees
#i_smooth=1 does filtering and smoothning of data

    ny=len(t_all[:,0]);s=[];y=[];t=[]
    for i in range (ny):
        t_i=t_all[i];data_i=data_all[i];y_i=y_all[i]
#        i_1=abs(t_i-14.55).argmin();i_2=abs(t_i-19.55).argmin()
#        t_i=t_i[i_1:i_2];data_i=data_i[i_1:i_2];
#        y_i=y_i[i_1:i_2];
        s.append(data_i)
        y.append(y_i)
        t.append(t_i)

    s=array(s);t=array(t);y=array(y)

    nt=len(t[0,:]);
    print (nt,ny)
    nt2=int(nt/2);ny2=int(ny/2)

    tmx=[];ymn=[];cmx=[];ymn0=[];smx=[]
    for i in range (ny):
        for j in range (ny):
            if j!=i:
                s_mx=max(abs(s[i]).max(),abs(s[j]).max())
                cin=convolve(s[i],s[j],mode='same')#[nt-nt2:nt+nt2]
                imx=argmax(cin);imn=argmin(cin);
                tmx.append(t[j][imx])
                ymn.append(y[j][imx]-y[i][imx]);
                ymn0.append((y[j][imx]+y[i][imx])/2.);
                cmx.append(cin[imx])
                smx.append(s[i][imx])
                
    tmx=array(tmx);ymn=array(ymn);ymn0=array(ymn0);cmx=array(cmx)
    smx=array(smx)
    return (t,y,s,tmx,ymn,ymn0,cmx,smx)

#%%
def cross_correlate_atual(t,y,data):
    #all inputs are two dimensions numpy array with first and second dimensions as for space and time resepctively
    #Todas entradas estão bi-dimensional com 1 e 2 representam espação e tempo respectivamente.
    # tempo e Espaçõ está em segundos e kilo-metros
    t2=t;y2=y
    if len(shape(t))==1 and len(shape(y))==1:
        [t2,y2]=meshgrid(t,y)
    nt=len(t2[0,:]);ny=len(y2[:,0])

    tmx=[];ymx=[];cmx=[];vmx=[];wl_mx=[];tau=[];vel=[]
    cin_p=zeros((nt))
    tmx2=zeros((ny,nt));ymx2=zeros((ny,nt));cmx2=zeros((ny,nt));
    wl2=zeros((ny,nt));tau2=zeros((ny,nt));vel2=zeros((ny,nt));
    for i in range (1,ny-1):
        cin=0*data[i,:]
        for j in range (1,ny-1):
            if j!=i:
                for it in range (nt):
                    cin=cin+data[i,:]*data[j,it]
        if cin.any() != 0:
#            cin=cin/cin.max()
            wl=0.5*(y2[i,:]-y2[i-1,:])*(cin+cin_p)/(cin-cin_p)
            imx=find_peaks(cin)[:,0]
            tmx.append(t2[i,imx])
            tmx2[i,imx]=t2[i,imx];ymx2[i,imx]=y2[i,imx];cmx2[i,imx]=abs(cin[imx])
            wl2[i,imx]=wl[imx];tau2[i,imx]=gradient(t2[i,imx])/2.;
            vel2[i,imx]=2*wl[imx]/gradient(t2[i,imx])
        cin_p=cin
    
    wl_max=y2.max()-y2.min()
    tau_min=(gradient(t2)[1]).min()/2
    vmx=wl_max/tau_min
    cmx2=cmx2/cmx2.max()
#    wl2[abs(wl2)>wl_max]=0
#    tau2[abs(tau2)<tau_min]=0
#    vel2[abs(vel2)>vmx]=0
        
    return (tmx2,ymx2,cmx2,wl2,tau2,vel2)

#%%
def cross_correlate_einsum(t,y,data):
    #all inputs are two dimensions numpy array with first and second dimensions as for space and time resepctively
    #Todas entradas estão bi-dimensional com 1 e 2 representam espação e tempo respectivamente.
    # tempo e Espaçõ está em segundos e kilo-metros
    t2=t;y2=y
    if len(shape(t))==1 and len(shape(y))==1:
        [t2,y2]=meshgrid(t,y)
    nt=len(t2[0,:]);ny=len(y2[:,0])

    tmx=[];ymx=[];cmx=[];vmx=[];wl_mx=[];tau=[];vel=[]
    cin_p=zeros((nt))
    tmx2=zeros((ny,nt));ymx2=zeros((ny,nt));cmx2=zeros((ny,nt));
    wl2=zeros((ny,nt));tau2=zeros((ny,nt));vel2=zeros((ny,nt));
    cin=einsum('ij,kl->ij', data, data)
    wl=cin*gradient(y2)[0]/gradient(cin)[0]
    for i in range (ny):
#        if cin.any() != 0:
#            cin=cin/cin.max()
        imx=find_peaks(cin[i,:])[:,0]
        if len(imx)>1:
            tmx.append(t2[i,imx])
            tmx2[i,imx]=t2[i,imx];ymx2[i,imx]=y2[i,imx];cmx2[i,imx]=abs(cin[i,imx])
            wl2[i,imx]=wl[i,imx];tau2[i,imx]=gradient(t2[i,imx])/2.;
            vel2[i,imx]=2*wl[i,imx]/gradient(t2[i,imx])
        
    
    wl_max=y2.max()-y2.min()
    tau_min=(gradient(t2)[1]).min()/2
    vmx=wl_max/tau_min
    cmx2=cmx2/cmx2.max()
#    wl2[abs(wl2)>wl_max]=0
#    tau2[abs(tau2)<tau_min]=0
#    vel2[abs(vel2)>vmx]=0
        
    return (tmx2,ymx2,cmx2,wl2,tau2,vel2)

#%%
def convolve_al(data,f):
    nd=len(data);nf=len(f);
    data_n=zeros((nd))
    if nd==nf:
        for i in range (nd):
            for j in range (i,nf):
                data_n[i]=data_n[i]+data[j]*f[i-j]
    else:
        for i in range (nd-nf):
            for j in range (nf):
                data_n[i]=data_n[i]+data[i+j]*f[j]

    #%%
    nf2=int(nf/2)
    data_n=shift(data_n,nf2)
       
    if nf % 2 !=0:
        nf2=nf2+1

    x=[0,nf2];y=[data[0],data_n[nf2]]
    xvals=arange(0,nf2)
    data_n[0:nf2]=interp(xvals,x,y)
##    
    x=[-nf2-1,-1];y=[data_n[-nf2-1],data[-1]]
    xvals=linspace(x[0],x[-1],nf2)
    data_n[-nf2:]=interp(xvals,x,y)
        
#        print (nf2,x,y,xvals,data_n[-nf2],data_n[-nf2:])
    return data_n

#%%
def wave_fft(t,data):
    nt=len(t);dt=(t[1]-t[0]);dt_s=(t[-1]-t[0])
    fn=1./dt;fs=1./(1.*dt_s)
    data_sine=cos(2*pi*(t-t[0])*60./10.)
    pwr=abs(fftpack.fft(data))
    freqs=fftpack.fftfreq(len(data))*fn
    pd=1./freqs;n_fft=int(len(pd)/2.)
    pwr=pwr[:n_fft];pd=pd[:n_fft]
    pwr=abs(data).max()*pwr/pwr.max()
    return (pd,pwr)

#%%
def wave_fft_alam(t,data,n_mode):
    nt=len(t);dt=(t[1]-t[0]);dt_s=(t[-1]-t[0])
    fn=1./dt;fs=1./(1.*dt_s)
    modes=logspace(log10(2./fn),log10(nt/(2.*fn)),num=n_mode);#modes=arange(2./fn,nt/(fn*2.),1./fn);#
    
#    n_mode=len(modes);
    pd=[];pwr=[]
    for i in range (n_mode):
        pd_o=modes[i]
        win=cos(2.*pi*t/pd_o)
        pwr_cos=convolve(data,win)#,mode='same')
        win=sin(2.*pi*t/pd_o)
        pwr_sin=convolve(data,win)#,mode='same')
        pwr_abs=(sqrt(pwr_cos**2.+pwr_sin**2.)).mean()
        pwr.append(pwr_abs)
        pd.append(pd_o)
    pwr=array(pwr);pd=array(pd);
    pwr=abs(data).max()*pwr/abs(pwr).max()
    pwr[pwr<abs(data).max()/100.]=0
    return (pd,pwr,modes)

#%%
def wavelet(iw,t,data,n_mode): # t is in minutes
    def wave_ones():
        win=ones((int(modes[i]),))/int(modes[i]);
#        pwr=(data-real(np.convolve(data,win,mode='same')))
        pwr=(data-convolve_al(data,win))
        return pwr
    def wave_morlet():
        wo=5.;so=1
        win=morlet(modes[i],w=wo,s=so,complete=True);#pd[i]=pd[i]/(2.*so*wo)
        pwr=real(np.convolve(data,win,mode='same'))
        return pwr
    def wave_hat():
        win=ricker(nt,modes[i])
        pwr=real(np.convolve(data,win,mode='same'))
        return pwr
    
    data_0=data
    f=wave_fft_alam(t,data,n_mode);pd_fft=f[0];pwr_fft=f[1];modes_fft=f[2]
    data_fft=[]
    data_fft.append([f[0],f[1]])
    
    nt=len(t);dt=(t[1]-t[0]);dt_s=(t[-1]-t[0])
    fn=1./dt;fs=1./(1.*dt_s)
    modes=modes_fft/dt#logspace(log10(2),log10(nt/2),num=n_mode);#modes=arange(int(2*dt*fn),int(dt_s*fn/2.),int(1*dt*fn));#
#    n_mode=len(modes);n_mode=min(n_mode,1024)
    
    pd_all=[];pwr_all=[];emd_all=[];
    
    for i in range (n_mode):
        if iw==0:pwr=wave_ones();data=data-pwr
        if iw==1:pwr=wave_morlet()
        if iw==2:pwr=wave_hat()
        
        if iw==0:
            peaks=find_peaks(pwr);
            imx=peaks[-1];imn=peaks[0];
            pd=pd_fft[i]#(t[imx]-t[imn])/(len(peaks)/2.)
            if imx==imn:
                pd=pd_prev
                #print ('NO MORE HARMONICS AFTER PERIOD, MINUTES=',i, pd)
                break
            if pd >=nt*dt/2.:
                #print ('NO MORE HARMONICS AFTER PERIOD, MINUTES=',i, pd)
                break
        if iw !=0:
            pd=2.*dt*modes[i];
            
        i_fft=abs(pd_fft-pd).argmin();
        emd=(modes_fft.mean()/pd)*pwr*pwr_fft[i_fft]/abs(pwr_fft).mean()
        pd_prev=pd
        pd_all.append(pd)
        pwr_all.append(pwr)
        emd_all.append(emd)
    #print ('Modes=',i, 'period, Minutes=',array(pd_all).min(),array(pd_all).max())  
    return(array(pd_all),array(pwr_all),array(emd_all),array(data_fft))

#%%
def data_filt(t,data,tl,tu):
    dt=t[1]-t[0]
    nm1=int(tl/dt);nm2=int(tu/dt)
    if nm1==0:
        data_filt=data-np.convolve(data,ones((nm2,))/nm2,mode='same')
    else:
        data_filt=np.convolve(data,ones((nm1,))/nm1,mode='same')\
                -np.convolve(data,ones((nm2,))/nm2,mode='same')
    data_filt[:nm2]=data_filt[nm2]
    data_filt[-nm2:]=data_filt[-nm2]
    return data_filt

#%%

#dt=0.01
#t=arange(0,2.,dt);nt=len(t)
#data=0*t
#for i in range (11):
#    pd=(i+1)*2.5*dt;omega=2.*pi/pd
#    data=data+cos(omega*t)/(0+1)
#
#f=wavelet(0,t,data,32)
#pd=f[0];pwr=f[1];emd=f[2];data_fft=f[3]
#fr=1./pd;#Hz
#
##%%    
#fig = figure(1,figsize=(12,12))
#subplot(211)
#plot(t,data)
#i=find_peaks(data)
##plot(t[i],data[i],'ro')
#
#subplot(212)
#plot(t,data,'r',lw=4)
#plot(t,pwr[:,:].sum(0),'b',lw=2)
#plot(t,emd[:,:].mean(0),'b--')
#
##%%
#figure(2,figsize=(12,12))
#subplot(212)
#for i in range (len(pd)):
#    plot(t,fr[i]+1.*emd[i,:])
#    
#figure(3,figsize=(12,12))
#f=wave_fft_alam(t*60.,data,32)
#pd=f[0];pwr=f[1]
#semilogx(1.e+03/(pd*60.),pwr,'r')
#f=wave_fft(t*60,data)
#pd=f[0];pwr=f[1]
#semilogx(1.e+03/(pd*60.),pwr,'b')
#
#figure(5,figsize=(12,12))
#plot(t,data,'r-o')
#n_smooth=2
#win=ones((n_smooth,))/n_smooth#cos(2.*pi*t/(10.*dt))#ones((5,))/5
#f=convolve(data,win,mode='same')
#plot(t,f,'b-')
#f=convolve_al(data,win)
#plot(t,f,'b--')
