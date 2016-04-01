## This example shows how to generate spherically symmetric random vectors.

import matplotlib.pyplot as plt
import numpy as np
import randSS

## Parameters
N = 2  # Length of each random vector
M = 10000  # Number of random vectors to generate

## Data generation
# ATTENTION : quand mettre 1.0 ou 1 ?
# Example 1 : multivariate spherical Gaussian 
options = dict(type='gauss',scale=1)
Xgauss = randSS:randSS(N,M,options) 

# Example 2 : multivariate spherical Student 
options = dict(type='t_mg',df = 5)
Xstudent = randSS:randSS(N,M,options) 

# Example 3 : multivariate spherical Kotz 
options = dict(type='kotz',N_kotz = 2,r = .5,s = 1)
Xkotz = randSS:randSS(N,M,options) 

# Example 4 : multivariate spherical Laplace 
options = dict(type='laplace',scale=1)
Xlaplace = randSS:randSS(N,M,options) 

# Example 5 : multivariate spherical (continuous) Bessel 
options = dict(type='bessel',q = 1,r = 1,scale = 1)
Xbessel = randSS:randSS(N,M,options) 

# Example 6 : multivariate spherical exponential power 
options = dict(type='exp_power',power = 3)
Xpower = randSS:randSS(N,M,options) 



## Visualization



plt.figure(1)

# Gaussian 
plt.subplot(321)
plt.hist2d(Xgauss.T,[30 30])
# ou np.histogramdd(Xgauss.T,[30 30])
# ou np.histogram2d(Xgauss.T,[30 30])
plt.axis([-3 3,-3 3,0 200])
plt.title("Gaussian")

# Student 
plt.subplot(322)
plt.hist2d(Xstudent.T,[70 70])
plt.axis([-3 3,-3 3,0 200])
plt.title("Student")

# Kotz 
plt.subplot(323)
plt.hist2d(Xkotz.T,[20 20])
plt.axis([-3 3,-3 3,0 200])
plt.title("Kotz")

# Exponential power 
plt.subplot(324)
plt.hist2d(Xpower.T,[12 12])
plt.axis([-3 3,-3 3,0 200])
plt.title("Exp. power")

# Laplace
plt.subplot(325)
plt.hist2d(Xlaplace.T,[50 50])
plt.axis([-3 3,-3 3,0 200])
plt.title("Laplace")

# Bessel
plt.subplot(326)
plt.hist2d(Xbessel.T,[50 50])
plt.axis([-3 3,-3 3,0 200])
plt.title("Bessel")

plt.show()

'''
from rootpy.interactive import wait
from rootpy.plotting import Canvas, Hist, Hist2D, Hist3D
from rootpy.plotting.style import set_style
import numpy as np

set_style('ATLAS')

c1 = Canvas()
a = Hist(1000, -5, 5)
a.fill_array(np.random.randn(1000000))
a.Draw('hist')

c2 = Canvas()
c2.SetRightMargin(0.1)
b = Hist2D(100, -5, 5, 100, -5, 5)
b.fill_array(np.random.randn(1000000, 2))
b.Draw('LEGO20')

c3 = Canvas()
c3.SetRightMargin(0.1)
c = Hist3D(10, -5, 5, 10, -5, 5, 10, -5, 5)
c.markersize = .3
c.fill_array(np.random.randn(10000, 3))
c.Draw('SCAT')
wait(True)
'''