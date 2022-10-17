import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# plotting function written by Dr. McNames at PSU
def PlotSystem(sos,Rs=None,Rp=None,wp=None,ws=None):
    w, h = signal.sosfreqz(sos,2000)

    (zeros,poles,k) = signal.sos2zpk(sos)

    #fig = plt.figure()
    figure = plt.figure(num=1,figsize=(10,6),dpi=100)
    figure.clf()
   
    axes = figure.add_subplot(3,2,1)
    for zeroAngle in np.angle(zeros):
        axes.axvline(zeroAngle,color='b',linestyle=':',linewidth=0.5)
    axes.plot(w, 20 * np.log10(np.maximum(abs(h), 1e-5)),color='r')
    if Rs:
        yMin = -Rs-20
    else: 
        yMin = -60-20
    yMax = 5.0
    if Rs is not None and ws is not None:
        axes.fill_between([0,ws[0]*np.pi],[-Rs,-Rs],[yMax,yMax],linewidth=0.0,facecolor='k',alpha=0.2)        
        axes.fill_between([ws[1]*np.pi,np.pi],[-Rs,-Rs],[yMax,yMax],linewidth=0.0,facecolor='k',alpha=0.2)
    if Rp is not None and wp is not None:
        ap = 10.0**(Rp/(-20.0))
        axes.fill_between(wp*np.pi,[yMin,yMin],[ap,ap],linewidth=0.0,facecolor='k',alpha=0.2)
        axes.fill_between(wp*np.pi,[1.0,1.0],[yMax,yMax],linewidth=0.0,facecolor='k',alpha=0.2)
    axes.set_ylabel('Amplitude (dB)')
    axes.set_ylim([yMin,yMax])
    axes.set_xlim([0.0,np.pi])

    axes = figure.add_subplot(3,2,2)
    axes.plot(w,abs(h),color='r')
    yMin = 0.0
    yMax = 1.05
    for zeroAngle in np.angle(zeros):
        axes.axvline(zeroAngle,color='b',linestyle=':',linewidth=0.5)
    if Rs is not None and ws is not None:
        axes.fill_between([0,ws[0]*np.pi],[-Rs,-Rs],[yMax,yMax],linewidth=0.0,facecolor='k',alpha=0.2)
        axes.fill_between([ws[1]*np.pi,np.pi],[-Rs,-Rs],[yMax,yMax],linewidth=0.0,facecolor='k',alpha=0.2)
    if Rp is not None and wp is not None:
        ap = 10.0**(Rp/(-20.0))
        axes.fill_between(wp*np.pi,[yMin,yMin],[ap,ap],linewidth=0.0,facecolor='k',alpha=0.2)
        axes.fill_between(wp*np.pi,[1.0,1.0],[yMax,yMax],linewidth=0.0,facecolor='k',alpha=0.2)
    axes.set_ylabel('Amplitude')
    axes.set_ylim([0.0,1.05])
    axes.set_xlim([0.0,np.pi])

    axes = figure.add_subplot(3,2,3)
    for angle in np.arange(-360.0,360.0+1,180.0):
        axes.axhline(angle,color='k',linestyle=':',linewidth=0.5)
    for zeroAngle in np.angle(zeros):
        axes.axvline(zeroAngle,color='b',linestyle=':',linewidth=0.5)    
    axes.plot(w, np.angle(h)*(180.0/np.pi),color='g')
    axes.set_ylabel('Phase ($^o$)')
    axes.set_ylim([-190.0,190.0])
    axes.set_xlim([0.0,np.pi])

    axes = figure.add_subplot(3,2,4)
    phaseUnwrapped = np.unwrap(np.angle(h))*(180.0/np.pi)
    for angle in np.arange(-360.0*10,360.0*10+1,180.0):
        axes.axhline(angle,color='k',linestyle=':',linewidth=0.5)
    for zeroAngle in np.angle(zeros):
        axes.axvline(zeroAngle,color='b',linestyle=':',linewidth=0.5)
    axes.plot(w,phaseUnwrapped,color='g')
    axes.set_ylabel('Phase ($^o$)')
    axes.set_xlim([0.0,np.pi])
    axes.set_ylim(phaseUnwrapped.min(),phaseUnwrapped.max())

    (num,den) = signal.sos2tf(sos)
    wgd,groupDelay = signal.group_delay((num,den), w=2**13)
    groupDelay[0] = 0.0
    axes = figure.add_subplot(3,2,5)
    for zeroAngle in np.angle(zeros):
        axes.axvline(zeroAngle,color='b',linestyle=':',linewidth=0.5)
    axes.axhline(0.0,color='k',linestyle=':',linewidth=0.5)
    axes.plot(wgd, groupDelay)
    axes.set_ylabel('GD ($n$)')
    yMin = np.percentile(groupDelay,0.05)-1.0
    yMax = np.percentile(groupDelay,99.5)+1.0
    axes.set_ylim([yMin,yMax])
    axes.set_xlim([0.0,np.pi])

    axes = figure.add_subplot(3,2,6)
    (num,den) = signal.sos2tf(sos)
    ni = np.arange(200)
    xi = signal.unit_impulse(ni.shape)
    hi = signal.lfilter(num,den,xi)
    axes.axhline(0.0,color='k',linestyle=':',linewidth=0.5)
    axes.vlines(ni,0,hi,color='b')
    axes.set_ylabel('h[n]')
    axes.set_xlim([0.0,ni[-1]+0.5])
    
    figure.tight_layout(pad=3.0)

    figure = plt.figure(num=2,figsize=(4,4),dpi=100)
    figure.clf()

    axes = figure.add_subplot(1,1,1)
    t = np.linspace(0,np.pi*2,100)
    axes.plot(np.cos(t), np.sin(t),color='k',linewidth=0.8,linestyle=':')
    axes.axhline(0.0,color='k',linewidth=0.8,linestyle=':')
    axes.axvline(0.0,color='k',linewidth=0.8,linestyle=':')
    axes.plot(np.real(zeros),np.imag(zeros),marker='o',linestyle='none',markerfacecolor='None',color='b',linewidth=0.1)
    axes.plot(np.real(poles),np.imag(poles),marker='x',linestyle='none',color='r',linewidth=0.1)
    axes.set_xlim([-1.2,1.2])
    axes.set_ylim([-1.2,1.2])
    axes.set_aspect('equal')
    axes.set_xlabel('Real')
    axes.set_ylabel('Imaginary')
    axes.axis('off')

    return groupDelay