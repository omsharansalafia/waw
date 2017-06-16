IO_specifications = """
# The emcounterparts module provides functions that produce EM counterpart
# lightcurves taking binary parameters as inputs, with uniform input/output formats.
#
# The input parameters of the functions are:
# binary_params: a dictionary containing arrays of compact binary parameters.
# [Expected keys in binary_params are e.g.:
#     m1, m2: component masses in Msun
#     dL: luminosity distance in Mpc
#     iota: orbital plane inclination w.r.t. the line of sight in rad
#     chi_eff: effective spin]
# nu_obs: observing frequency in Hz
# t: observing times in days
# The optional "verbose" keyword can be set to True to get information on
# the lightcurve computation/retrieval status.
#
# The functions must return a (N,M) ndarray containing N lightcurves,
# each defined by M flux densities (@nu_obs) in millijansky corresponding 
# to the input observing times t, where M = len(t). 
# The number N of lightcurves is equal to the length of the m1,m2,dL,
# iota,chi_eff arrays contained in the binary_params dictionary (they must 
# all be of the same length and they must represent the parameters of a 
# sample of N binaries).
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo #use Planck+15 cosmological parameters
from .aftab import aftab
import os

# convenience function to get the redshift corresponding to a given luminosity distance
z0 = np.linspace(0.,10.,10000)
dL0 = cosmo.luminosity_distance(z0).to('Mpc').value
z_of_dL = lambda dL: np.interp(dL,dL0,z0)

# sgrb afterglow lightcurve
def sgrb_afterglow(binary_params,nu_obs,t,verbose=False):
	"""
	Short Gamma-Ray Burst afterglow lightcurve associated to a binary merger.
	
	"""
	
	# FIXME: this module must be replaced a much more flexible one 
	# (i.e. make use of all binary parameters, allow for sampling the
	# non-merger-related parameters from prior ditributions, allow
	# for computing the lightcurve at any desired observing frequency,
	# and so on...)
	
	# initialize the aftab module loading the table used in Salafia+17
	bdir = os.path.dirname(__file__)
	filename = os.path.join(bdir, 'aftab/data/eb0.01thj0.2E50n0.01')
	my_aftab = aftab.LoadTable(filename)
	
	# some afterglow parameters are fixed in this simple model
	E = 1e50 #erg -- jet isotropic equivalent kinetic energy
	n = 0.01 #cm-3 -- ISM density
	# note also that we're not using m1, m2 and chi_eff
	
	# the viewing angle is derived from the inclination
	# simply assuming that two opposite jets are launched 
	# perpendicular to the orbital plane
	if np.isscalar(binary_params['iota']): # scalar version
		if binary_params['iota']<np.pi/2.:
			thv = binary_params['iota']
		else:
			thv = np.pi-binary_params['iota']
	else: # array version
		thv = binary_params['iota']
		thv[binary_params['iota']>np.pi/2.] = np.pi - binary_params['iota'][binary_params['iota']>np.pi/2.]
	
	# associate redshifts to lum distances
	z = z_of_dL(binary_params['dL'])
	
	# compute the lightcurves
	flux = np.empty([len(binary_params['dL']),len(t)])
	if verbose:
		print("")
	for i in range(len(binary_params['dL'])):
		if verbose:
			print(" #### computing lightcurve {0} of {1} ({2:.1f} percent completed) ####".format(i+1,len(binary_params['dL']),i/len(binary_params['dL'])*100.),end="\r")
		T,F = my_aftab.lightcurve(E,n,thv[i],nu_obs,z[i],binary_params['dL'][i]*3.08e24,60)
		flux[i] = np.interp(t,T,F)
	
	if verbose:
		print(" #### {} lightcurves computed (100 percent completed) ####".format(len(binary_params['dL'])))
	
	return flux
	
if __name__=="__main__":
	
	# example
	
	# create mock binary parameters
	m1 = np.random.normal(1.33,0.05,4)
	m2 = np.random.normal(1.33,0.05,4)
	chi_eff = np.zeros(4)
	
	dL = np.random.uniform(30.,100.,4)
	iota = np.random.uniform(0.,np.pi,4)
	
	
	# choose obs frequency
	nu_obs = 4.6e14 #Hz -- optical r band
	
	# compute lightcurves
	t = np.logspace(0.,2.,30)
	data = sgrb_afterglow({'dL':dL,'iota':iota},nu_obs,t,verbose=True)
	
	# plot them!
	import matplotlib.pyplot as plt
	plt.rcParams['figure.autolayout']=True
	plt.rcParams['font.size']=12
	
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.loglog(t,data[i],lw=3)
		plt.xlabel('time after merger [days]')
		plt.ylabel('flux density [mJy]')
		plt.title('dL:{2:.0f}Mpc iota:{3:.1f}deg'.format(m1[i],m2[i],dL[i],iota[i]/np.pi*180.))
	
	plt.show()
		
	
	
	
