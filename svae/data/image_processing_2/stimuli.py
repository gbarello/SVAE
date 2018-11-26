import numpy as np
import test_gratings as test
import image_processing as proc
import math

def make_SS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,get_grat = False):

    sizes = range(0,int(1*(fdist + freq)),1)
#    sizes = [10,20]

    grats = np.array([test.GRATC(1,0,freq,s,tot + 2*fdist + 1) for s in sizes])

    if get_grat:
        return grats

    filt = np.array([proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot) for g in grats])

    filt = np.array([proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in filt])
    
    return np.array([c*filt for c in con])#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_grating(con,angle,freq,rad,tot,phase = "c"):

    if phase == "c":
        return test.GRATC(con,angle,freq,rad,tot)
    elif phase == "s":
        return test.GRATS(con,angle,freq,rad,tot)
        

def make_OTUNE_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,R):

    grats = np.array([[test.GRATC(c,a,freq,freq,tot + 2*fdist + 1) for a in np.linspace(0,2*np.pi,32)] for c in [.05,.1,.25,.5]])

    filt = np.array([[proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot,R) for g in C] for C in grats])

    filt = np.array([[proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in C] for C in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_full_field_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,R):
    
    grats = np.array([test.GRATC(c,0,freq,2*(tot + 2*fdist + 1),tot + 2*fdist + 1) for c in con])

    filt = np.array([proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot,R) for g in grats])

    filt = np.array([proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_full_field_COS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,nda,R):
    
    grats = np.array([test.GRATC(c1,0,freq,2*(tot + 2*fdist + 1),tot + 2*fdist + 1) + test.GRATC(c2,d,freq,2*(tot + 2*fdist + 1),tot + 2*fdist + 1) for k in range(nda) for c1 in con for c2 in con for d in [k*np.pi/(2*(nda - 1))]])

    filt = np.array([proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot,R) for g in grats])

    filt = np.array([proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_att_COS_filters(con,p2,nfilt,nang,npha,freq,scale,tot,fdist):

    if p2 == 0:
        grats = np.array([test.GRATC(c,0,freq,fdist/2,tot + 2*fdist + 1) + test.GRATC(c,np.pi/2,freq,2*(tot + 2*fdist + 1),tot + 2*fdist + 1) for c in con])
    elif p2 == 1:
        grats = np.array([test.GRATC(c,0,freq,fdist/2,tot + 2*fdist + 1) for c in con])
    else:
        grats = np.array([test.GRATC(c,np.pi/2,freq,fdist/2,tot + 2*fdist + 1) for c in con])

    filt = np.array([proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot) for g in grats])

    filt = np.array([proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_BSS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,nda):
   
    grats = np.array([[[
        test.GRATC(c,0.,freq,fdist,tot + 2*fdist + 1),
        test.GRATC(c,0.,freq,fdist,tot + 2*fdist + 1,surr = 0)
        +
        test.s_GRATC(1,da,freq,fdist,tot+2*fdist + 1,surr = 0.)
    ] for c in con] for da in [i * math.pi/(2 * ((nda - 1) if nda > 1 else 1)) for i in range(nda)]])

    filt = np.array([[[proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot,R) for g in C] for C in A] for A in grats])

    filt = np.array([[[proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in C] for C in A] for A in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_COS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,nda,R,GRID = True):

    if GRID:
        grats = np.array([[[test.GRATS(c1,0,freq,fdist/2,tot + 2*fdist + 1),
                            test.GRATS(c2,da,freq,fdist/2,tot + 2*fdist + 1),
                            (
                            test.GRATS(c1,0,freq,fdist/2,tot + 2*fdist + 1)
                            + 
                            test.GRATS(c2,da,freq,fdist/2,tot + 2*fdist + 1)
                            )
                            ] for c1 in con for c2 in con] for da in [i * math.pi/(2 * (nda - 1)) for i in range(0,nda)]])

    else:
        grats = np.array([[[test.GRATS(c1,0,freq,fdist/2,tot + 2*fdist + 1),
                            test.GRATS(c1,da,freq,fdist/2,tot + 2*fdist + 1),
                            (
                            test.GRATS(c1,0,freq,fdist/2,tot + 2*fdist + 1)
                            + 
                            test.GRATS(c1,da,freq,fdist/2,tot + 2*fdist + 1)
                            )
                            ] for c1 in con] for da in [i * math.pi/(2 * (nda - 1)) for i in range(0,nda)]])

    filt = np.array([[[proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot,R) for g in C] for C in A] for A in grats])

    filt = np.array([[[proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in C] for C in A] for A in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_TI_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,R,npts = 20):
   
    grats = np.array([[
                    test.GRATC(c,0,freq,fdist/2,tot + 2*fdist + 1,surr = 0)
                    + 
                    test.s_GRATC(c,a,freq,fdist/2,tot + 2*fdist + 1,surr = 0)
                    for a in [i*math.pi/npts for i in range(npts)]] for c in con])
    
    filt = np.array([[proc.get_phased_filter_coefficients(g,nang,npha,freq,scale,tot,R) for g in C] for C in grats])

    filt = np.array([[proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for I in C] for C in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

def make_WTA_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,R,npts = 32):
   
    grats = np.array([[
        [
        test.GRATC(0,a,freq,tot + 2*fdist + 1,tot + 2*fdist + 1)
        + 
        test.GRATC(c,a + (np.pi/2),freq,tot + 2*fdist + 1,tot + 2*fdist+1)
        ,
        test.GRATC(1,a,freq,tot + 2*fdist + 1,tot + 2*fdist + 1)
        + 
        test.GRATC(c,a + (np.pi/2),freq,tot + 2*fdist + 1,tot + 2*fdist + 1)
        ]
        for a in [i*math.pi/npts for i in range(npts)]] for c in con])
    
    filt = np.array([[[proc.get_phased_filter_coefficients(x,nang,npha,freq,scale,tot,R) for x in g] for g in C] for C in grats])

    filt = np.array([[[proc.sample_coef(x,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],nfilt,fdist) for x in I] for I in C] for C in filt])
    
    return filt#(filt-np.array([[[dif]]]))/np.array([[[fac]]])

if __name__ == "__main__":
    import math

    "make_BSS_filters(con,nfilt,nang,npha,freq,scale,tot,fdist,nda)"
    A = make_BSS_filters([0.,.01,.02,.03,.04,.1,.2,.3,.4,.5],8,4,2,2*.7*2*math.pi,1,10,10,1)

#    print("A[0,:,0,0,0,0]")
#    print(A[0,:,:,0,0,0,0])
    print(A.shape)
    for a in A[0]:
        print(a[0,0,0,:,0])
        print(a[0,0,1,:,0])
        print(a[1,0,0,:,0])
        print(a[1,0,1,:,0])
