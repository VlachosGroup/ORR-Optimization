# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:36:33 2017

@author: lansf
"""
Coverage111 = [6.14313809e-06,3.56958665e-12, 1.93164910e-01]
popt111 = [1.06582547,1.71404262,1.54934696,1.59707713,0.8094801,1]
CoverageCavity = [3.92759996e-01, 4.69470761e-04, 7.72335855e-07
            , 8.80315264e-13, 3.81904686e-04+1.31812086e-11]
poptCavity = [  7.55711383e-01,   3.98024255e-37,   3.06518803e-01,
         7.74202453e-01,   1.72808698e+00]
def dGdOH():
    s,t,u,x,y,z = popt111
    OHcov, OOHcov, Ocov = Coverage111
    dGval = u*s*(t*Ocov+OHcov)**(u-1) + z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
    return dGval
    
def dGdO():
    OHcov, OOHcov, Ocov = Coverage111
    s,t,u,x,y,z = popt111
    dGval = t*u*s*(t*Ocov+OHcov)**(u-1)+y*z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
    return dGval
    
def dGdOOH():
    OHcov, OOHcov, Ocov = Coverage111
    s,t,u,x,y,z = popt111
    dGval = x*(y*Ocov+OHcov)**z
    return dGval

def dGdOHedge():
    OHedge, OHcav, OOHedge, OOHcav, Oedge = CoverageCavity
    x,x2,x3,y,z = poptCavity
    dGval = y*x*z*(y*OHedge+OOHedge+y*Oedge)**(z-1) + x2*OHcav + x3*OOHcav
    return dGval

def dGdOHcav():
    OHedge, OHcav, OOHedge, OOHcav, Oedge = CoverageCavity
    x,x2,x3,y,z = poptCavity
    dGval = x2*(OHedge+OOHedge+Oedge)
    return dGval

def dGdOOHedge():
    OHedge, OHcav, OOHedge, OOHcav, Oedge = CoverageCavity
    x,x2,x3,y,z = poptCavity
    dGval = x*z*(y*OHedge+OOHedge+y*Oedge)**(z-1) + x2*OHcav + x3*OOHcav
    return dGval

def dGdOOHcav():
    OHedge, OHcav, OOHedge, OOHcav,Oedge = CoverageCavity
    x,x2,x3,y,z = poptCavity
    dGval = x3*(OHedge + OOHedge+Oedge)
    return dGval

def dGdOedge():
    OHedge, OHcav, OOHedge, OOHcav, Oedge = CoverageCavity
    Ocov = Oedge; OHcov = OHedge+OHcav; OOHcov = OOHedge+OOHcav
    s,t,u,x,y,z = popt111
    dGval = t*u*s*(t*Ocov+OHcov)**(u-1)+y*z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
    return dGval