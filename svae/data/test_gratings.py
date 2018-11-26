from PIL import Image
import numpy as np
import math
from scipy import signal

##these functions make gratings (s and c)
def GRATS(con,a,k,s,t,surr = "mean"):
    
    c = float(con)

    if surr == "mean":
        sur = 1.
    else:
        sur = 0.

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2
    sin = np.array([[1. + c*np.sin(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k) if (float(x) - t2)**2 + (float(y) - t2)**2 < s**2 else sur for x in range(t)] for y in range(t)])

#    sin = sin/sin.max()

    return sin/2

def GRATC(con,a,k,s,t,surr = "mean"):
    
    c = float(con)

    if surr == "mean":
        sur = 1.
    else:
        sur = surr

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2
    sin = np.array([[1. + c*np.cos(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k) if (float(x) - t2)**2 + (float(y) - t2)**2 < s**2 else sur for x in range(t)] for y in range(t)])

#    sin = sin/sin.max()

    return sin/2

#these function make "surround" gratings

def s_GRATS(con,a,k,s,t,surr = "mean"):
    
    c = float(con)

    if surr == "mean":
        sur = 1.
    else:
        sur = surr

    v1 = np.sin(a)
    v2 = np.cos(a)

    t2 = float(t)/2
    sin = np.array([[1. + c*np.sin(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k) if (float(x) - t2)**2 + (float(y) - t2)**2 >= s**2 else sur for x in range(t)] for y in range(t)])

#    sin = sin/sin.max()

    return sin/2

def s_GRATC(con,a,k,s,t,surr = "mean"):
    
    c = float(con)

    if surr == "mean":
        sur = 1.
    else:
        sur = 0.

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t)/2
    sin = np.array([[1. + c*np.cos(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k) if (float(x) - t2)**2 + (float(y) - t2)**2 >= s**2 else sur for x in range(t)] for y in range(t)])

#    sin = sin/sin.max()

    return sin/2

##
