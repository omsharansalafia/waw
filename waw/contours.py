import numpy as np
import healpy as hp
from scipy.interpolate import LinearNDInterpolator
from skimage import measure


def hpix_contours(m,levels=[0.5,0.9],nest=False):
	
	nside = hp.npix2nside(len(m))
	
	# define a grid over which to evaluate the density
	lat = np.linspace(-np.pi/2.,np.pi/2.,300)
	lon = np.linspace(0.,2.*np.pi,300)
	
	# evaluate the map and keep track of both coordinates and indices where
	# the evaluation is done
	values = np.zeros([300*300,2])
	points = np.zeros([300*300,2])
	MAP = np.zeros([300,300])
	
	k=0
	for i in range(300):
		for j in range(300):
			p = hp.ang2pix(nside,np.pi/2.-lat[j],lon[i],nest=nest)
			MAP[i,j] = m[p]
			values[k] = np.array([lon[i],lat[j]])
			points[k,0] = i
			points[k,1] = j
			k = k+1
	
	#construct a linear interpolator to get interpolated
	#coordinates from fractional indices
	lint = LinearNDInterpolator(points,values)
	
	#use skimage find_contours method to find fractional indices
	#defining the contours, then compute corresponding coordinates
	#by using the linear interpolator
	contours = []
	for l in levels:
		contour_components = measure.find_contours(MAP,l)
		#the find_contours function above returns a list of the
		#connected components of the contour. We unpack it,
		#compute the interpolated coordinates for each, and
		#store them in one array, separating them with nans
		
		whole_contour = np.array([[np.nan,np.nan]])
		
		for contour in contour_components:
			cont_coords = lint(contour)
			whole_contour = np.concatenate((whole_contour,cont_coords),axis=0)
			whole_contour = np.concatenate((whole_contour,[[np.nan,np.nan]]),axis=0)
		
		#then we mask the nans
		C = whole_contour.transpose()
		contours.append(np.ma.masked_invalid(C))
	
	return contours
