''' Function generating spherically symmetric random vectors 

Input: 
  N = size of the random vectors we wish to generate
  M = number of random vectors to generate
  options = structure with the following elements:
      -   options.type = name of the distribution ('gauss' for Gaussian,
      't_mg' or 't_fisher' for Student, 'kotz' for Kotz, 'mg' for Gaussian 
      mixture, 'exp_power' for Exponential power, 'laplace' for multivariate
      spherical Laplace, 'unifSS' for uniform on sphere, 'pearsonII' for 
      Pearson type II.
      -   other elements of options correspond to parameters of each law.

Output: 
  X = matrix of size NxM containing the M random vectors in column.

By A. Boisbunon, April 2011.
Modified in August 2012.
Modified in November 2012.

TODO: 
'''

import math
import sys
import numpy as np
import random

def randSS(N, M, options)
# sys.argv
	# A VERIFIER et a comparer avec 
	options = dict.setdefault('type', 'gauss')
	options = dict.setdefault('scale', 1)
	options = dict.setdefault('df', 100)
	options = dict.setdefault('rate', 1)
	options = dict.setdefault('power', 1)
	options = dict.setdefault('N_kotz', 2)
	options = dict.setdefault('r', .5)
	options = dict.setdefault('shape', 1)
	options = dict.setdefault('s', 1)
	options = dict.setdefault('q', 1)
	options = dict.setdefault('p2', 1)
	'''
	# A VERIFIER et a comparer avec 
    if type not in options:
        options['type']='gauss'
    
    if scale not in options:
        options['scale']=1
    
    if df not in options:
        options['df']=100
    
    if rate not in options:
        options['rate']=1
    
    if power not in options:
        options['power']=1
    
    if N_kotz not in options:
        options['N_kotz']=2
        
    if r not in options:
        options['r']=.5
    
    if s not in options:
        options['s']=1
    
    if shape not in options:
        options['shape']=1
        
    if q not in options:
        options['q']=1
        
    if p2 not in options:
        options['p2']=1
       '''
	   
    # Gaussian distribution from chi2 radius with N degrees of freedom
    if options['type']=='gauss':
        R = math.sqrt(np.random.chisquare(N, M, 1))     
        bool_rayon = 1 
	# multivariate student (with Fisher radius)
	# OK POUR LA MULTIPLICATION D'UN SCALAIRE PAR UN VECTEUR ???
    if options['type']=='t_fisher':
		R = math.sqrt(N*np.random.f(N, options['df'], M, 1))
		bool_rayon = 1 
    # multivariate student via gaussian mixture
	# VERIFIER LA TRANSPOSITION ET LA REPETITION
	# SI C'EST OK, FAIRE PAREIL POUR LAPLACE, BESSEL ET UNIFORME 
    if options['type']=='t_mg':
		v = 1./np.random.gamma(options['df']/2, 2/options['df'], M, 1)
		V = np.repeat(v.T, [N, 1], axis=0)
		X = np.dot(np.random.randn(N,M),math.sqrt(V))
		bool_rayon = 0 
	if options['type']=='exp':  # exponential radius
		R = np.random.exponential(options['rate'], M, 1)
		bool_rayon = 1 
	if options['type']=='wbl':  
		R = options['shape']*np.random.weibull(options['scale'], M, 1)  
		bool_rayon = 1 
	if options['type']=='kotz': # Multivariate Kotz with gamma radius
		R = math.sqrt(np.random.gamma(options['N_kotz']+N/2+1,options['r']/options['s']**2,M,1))
		bool_rayon = 1 
	if options['type']=='exp_power':# Multivariate exponential power with gamma radius
		# AR PUISSANCE D'UN VECTEUR (voir math.pow(x,y) si ca ne marche pas)
		R = np.random.gamma(N/(2*options['power']),2*options['scale']**(2*options['power']),M,1).**(1/(options['power']*2))
		bool_rayon = 1 
	if options['type']=='ep_kotz':# Multivariate exponential power through special case of Kotz
		R = math.sqrt(np.random.gamma(N/2+2,options['power']/options['scale']**2,M,1))
		bool_rayon = 1 
	if options['type']=='laplace': # Multivariate Laplace through Gaussian mixture
		v = np.random.gamma(N/2,1/options['scale']**2,M,1) 
		bool_rayon = 0 
		X = np.random.randn(N,M).*repmat(math.sqrt(v'),N,1)
	if options['type']=='bessel':
		v = np.random.gamma(options['q']+N/2,1/(2*options['scale']**2*options['r']),M,1) 
		bool_rayon = 0 
		X = np.random.randn(N,M).*repmat(math.sqrt(v'),N,1)            
    # Uniform distribution on a sphere of radius R
    if options['type']=='unifSS': 
		R = repmat(options['scale'],M,1) 
		bool_rayon = 1 
	if options['type']=='pearsonII':
		R = math.sqrt(np.random.beta(N/2,options['p2'],M,1)) 
		bool_rayon = 1 
    

    # If what we have generated sofar corresponds to the radius of the
    # vector, we now have to generate its direction.
    if bool_rayon == 1:
        Y = np.random.randn(N,M) 
        normY = math.sqrt(diag(np.dot(Y.T,Y))) 
        matNormY = np.repeat(normY.T, [N, 1], axis=0)
        U = np.divide(Y,matNormY)
        X = np.multiply(U,np.repeat(R.T, [N, 1], axis=0))
		# ou bien : X = np.multiply(U,R) --> A TESTER !!!
'''        normY = math.sqrt(diag(Y'*Y)) 
        matNormY = repmat(normY',N,1)
        U = Y./matNormY
        X = U.*repmat(R',N,1)
''' 
	print X


