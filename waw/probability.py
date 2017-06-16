import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# this core module should do the basic stuff: take posterior sample parameters, call an external
# function to produce the related lightcurves and produce the time-dependent detectability maps.
# - we want the maps to be healpix-based, for simplicity;
# - we want the lightcurves to be produced by an external tool/script for flexibility;
# - the KDE to get the detectabilty maps out of the lightcurves must be an independent module
#   (so that it can be developed independently)
# - the detectability maps then need to be converted into an observing strategy: a different
#   tool has to be used (don't mix up the two stages so that they can be independently developed)
# - the maps depend on sky position (1D thanks to healpix), time (1D) and frequency (1D), so they 
#   could be in principle 3D numpy arrays (maybe stored as hdf5 structures? Or fits is better?)

def approximate_skymap(ra,dec,nside=64):
	"""
	Approximate the sky-position posterior probability density by constructing
	a healpix histogram of the posterior samples and smoothing it.
	Useful when no LVC skymap is given. No 3D KDE + marginalization has
	to be done. The result is only an approximation of the true skymap.
	"""
	
	#compute histogram
	N = len(ra)
	th = np.pi/2. - dec
	phi = ra
	npix = hp.nside2npix(nside)
	
	h = np.zeros(npix)
	for i in range(len(th)):
		h[hp.ang2pix(nside,th[i],phi[i])]+=1.
	
	#refine histogram resolution by 4
	H = hp.ud_grade(h,nside*4)
	
	#smooth refined histogram
	H = hp.smoothing(H,sigma=2.*hp.nside2resol(nside),verbose=False)
	H = hp.smoothing(H,sigma=2.*hp.nside2resol(nside),verbose=False) #smooth it twice?
	
	#normalize
	H = H/np.sum(H)*16 #times 16 because the resolution will be degraded!
	
	return hp.ud_grade(H,nside)

def integrated_probability(skymap):
	"""
	Take a (healpix) skymap and return the corresponding integrated probability skymap.
	The result can be then used to compute the confidence regions.
	"""
	
	sort_idx = np.argsort(skymap)[::-1]
	csm = np.empty(len(skymap))
	csm[sort_idx] = np.cumsum(skymap[sort_idx])
	
	return csm


# sky_pos_cond_prob_gt, i.e. sky-position-conditional probability that x is greater than x0. 
# In a more compact form: P(x>x0|ra,dec)
def sky_pos_cond_prob_gt(x,x0,ra,dec,nside=32,limit_to_region=None):
	"""
	
	Return a numpy array which represents the healpix projection of the 
	sky-position-conditional probabilities that x>x0, i.e. P(x>x0|sky pos,...),
	constructed from a set of posterior samples as described in Salafia+17
	
	
	Parameters
	----------
	x: numpy 1D array
	    Value of x for each posterior sample.
	x0: float
	    Comparison value.
	ra, dec: numpy arrays of the same length as x
	    Sky positions of the posterior samples.
	nside: int, default=32
	    Resolution of the output healpix map.
	limit_to_region: 1D numpy array of booleans
	    A healpix map of booleans representing the region where the probability has to be computed.
	    Typically, one wants to limit the computation to the 90% confidence area not to waste
	    computational resources. If its nside is different from that given, it will be converted.
	    Default: None (i.e. all-sky).
	
	Returns
	-------
	m: a masked 1D numpy array of length 12*nside**2, which represents the output healpix map.
	"""
	
	# assert that x, ra and dec lengths match
	assert len(x) == len(ra)," The x and ra arrays must have the same lengths"
	assert len(x) == len(dec)," The x and dec arrays must have the same lengths"
	
	# compute the cartesian components of the vectors pointing at the positions
	# of the posterior samples
	npsamp = len(x)
	vx = np.cos(dec)*np.cos(ra)
	vy = np.cos(dec)*np.sin(ra)
	vz = np.sin(dec)
	
	# compute the cartesian components of the vectors pointing at the positions
	# of the skymap pixels
	npix = hp.nside2npix(nside)
	px,py,pz = hp.pix2vec(nside,np.arange(npix))
	
	# precompute mask that selects x_i>x0
	gt_x0 = x>x0
	
	# create an empty map
	m = np.ones(npix)*np.nan
	
	# if no limit_to_region given, create a mask with True everywhere,
	# otherwise adapt the given region resolution to ours
	if limit_to_region is None:
		region = np.ones(npix,dtype=bool)
	else:
		region = hp.ud_grade(limit_to_region,nside)
	
	# to compute the angular distances of all posterior samples to all pixels of the
	# skymap is quite memory intensive, so we better do it one skymap pixel at a time
	for p in np.arange(npix)[region]:
		# compute angular distances
		d = np.arccos(px[p]*vx + py[p]*vy + pz[p]*vz)
		
		#compute IDW weigths and normalize them
		bandwidth = np.std(d)*npsamp**(-1./5.) #Silverman's rule (FIXME: use a better estimate?)
		w = np.exp(-0.5*(d/bandwidth)**2)
		
		if np.sum(w)>0.: #do not normalize if the sum is 0
			w = w/np.sum(w)
		
		#compute the probability as prescribed in Salafia+17
		m[p] = np.sum(w[gt_x0])
	
	#return the map, masking invalid values
	return np.ma.masked_invalid(m)
	
if __name__=="__main__":
	# this is an example use of the above functions
	
	# generate mock posterior samples
	ra = np.random.normal(np.pi,np.pi/8.,10000)
	dec = np.random.normal(0.,np.pi/32.,10000)
	dL = np.empty(10000)
	for i in range(10000):
		dL[i] = np.random.normal((ra[i]-np.pi)/(np.pi/8.)*50. + 50.,10.)
		
	# compute the skymap
	sm = approximate_skymap(ra,dec,nside=128)
	cm = integrated_probability(sm)
	cr90 = cm<0.9
	
	# compute the contours containing 90 and 50 percent of the sky pos probability
	from contours import hpix_contours
	cntrs = hpix_contours(cm)
	
	# compute the sky-position-conditional probability that dL>50
	m = sky_pos_cond_prob_gt(dL,50,ra,dec,nside=64,limit_to_region=cr90)
	
	# plot the result
	import skyplot
	ax,cb = skyplot.plot_healpix_map(m)
	cb.set_label('P(dL>50 Mpc|ra,dec)')
	
	# plot the 50% and 90% confidence region contours
	for c in cntrs:
		skyplot.plot_ra_dec(c[0],c[1],ax=ax,rotate=0.,linestyle='-',color='white')
	
	ax.set_xlim([0.,2.*np.pi])
	ax.set_ylim([-np.pi/2.,np.pi/2.])
	ax.grid()
	
	skyplot.hourangle_ticks(ax)
	skyplot.degree_ticks(ax)
	
	plt.show()
