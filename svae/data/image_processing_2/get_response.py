import argparse

import test_gratings as test
import stimuli as stim
import image_processing as proc
import numpy as np

def main(f_scale,con,lam,phase,size,angle,n_filter,f_dist,file_name):


    MGSM = False
    if f_dist > 0:
        MGSM = True
        if n_filter != 4:
            print("MGSM is only implemented for n_filter = 4. Setting n_filter = 4.")
            
        n_filter = 4

    if file_name == "":
        file_name = "{}_{}_{}_{}_{}_{}_{}".format(f_scale,con,lam,phase,size,angle,n_filter,f_dist)

    print("Saving file to file name: {}".format(file_name))

    npha = 2
    f_freq = f_scale/(2*np.pi)
    tot = int(3*f_scale)
    
    if phase == 0:
        gfunc = test.GRATC
    else:
        gfunc = test.GRATS

    grat = gfunc(con,angle,lam,size,int(tot + 2*f_dist + 1))
    I = proc.get_phased_filter_coefficients(grat,n_filter,npha,f_freq,f_scale,tot)
    filt = proc.sample_coef(I,[[(len(I) - 1)/2,(len(I[0]) - 1)/2]],8,f_dist)

    filt = np.reshape(filt[0],[-1,n_filter*2])
    
    if MGSM:
        out = filt
    else:
        out = filt[0]

    np.savetxt(file_name + ".csv",out)

    print("Done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("f_scale", help="scale of filter RF.",
                        type = float)
    
    parser.add_argument("lam", help="wavelength (in px.) of the grating",
                        type = float)
    parser.add_argument("phase", help="phase of grating. Either 0 (cos phase) or 1 (sin phase)",
                        type = int)
    parser.add_argument("size", help="Radius in px. of the grating.",
                        type = float)
    parser.add_argument("angle", help="Angle of grating from horizontal in degrees",
                        type = float)

    parser.add_argument("con", help="Contrast of grating between 0 and 1.",
                        type = float)

    parser.add_argument("--n_filter",help="number of angles to sample",type = int,default = 4)
    parser.add_argument("--f_dist",help="distance in pixels between center and surround filter (for MGSM only)",type = float,default = 0)

    parser.add_argument("--file_name", help="name of file to save. If not given, save file with the list of arguments as the name.",
                        type = str,default = "")    
    

    args = vars(parser.parse_args())
        
    main(**args)
