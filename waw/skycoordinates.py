import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
import warnings

# location of the Greenwich Royal Observatory
Greenwich = EarthLocation(3980608.9024681724, -102.47522910648239, 4966861.273100675, unit='m')

#a date and time at which GMST = 0
time_gmst_0 = Time('2016-10-13 22:28:22',location=Greenwich)

def modulo_2pi(x):
	
	X = x
	
	while X>=np.pi:
		X=X-2.*np.pi
	while X<-np.pi:
		X=X+2.*np.pi
	
	return X

mod2pi = np.vectorize(modulo_2pi)

def RA2lon(RA,gpstime):
	"""Converts right ascension (in radians) to longitude (in radians), given a gps time."""
	gmst = Time(gpstime,format='gps',location=Greenwich).sidereal_time('mean')
	alpha = gmst.to('rad').value
	lon = RA-alpha
	
	if np.isscalar(lon):
		if lon>np.pi:
			lon = lon-2.*np.pi
	else:
		lon[lon>np.pi] = lon[lon>np.pi]-2.*np.pi
	
	return lon

def lon2RA(lon,gpstime):
	"""Converts longitude (in radians) to right ascension (in radians), given a gps time."""
	gmst = Time(gpstime,format='gps',location=Greenwich).sidereal_time('mean')
	
	if np.isscalar(lon):
		if lon<0.:
			Lon = lon+2.*np.pi
		else:
			Lon = lon
	
		RA = modulo_2pi(Lon+float(gmst/u.rad))
	else:
		
		RA = lon + gmst.to('rad').value
		
		RA[np.ma.less(RA,0.)]=RA[np.ma.less(RA,0.)]+2.*np.pi
		RA[np.ma.greater_equal(RA,2.*np.pi)]=RA[np.ma.greater_equal(RA,2.*np.pi)]-2.*np.pi
	
	return RA

def gpstime(mjd):
	"""Converts MJD in gpstime."""
	return Time(mjd,format='mjd').gps
	
def radians2hourangle(angle):
	"""Convert radians to hourangle (hours, minutes, seconds)."""
	hours = int(np.floor(angle/np.pi*12.))
	minutes = int(np.floor((angle/np.pi*12. - hours)*60.))
	seconds = ((angle/np.pi*12. - hours)*60. - minutes)*60.
	
	return '{0:d}h{1:d}m{2:.3f}s'.format(hours,minutes,seconds)

def radians2arc(angle):
	"""Convert radians to arc (degrees, minutes, seconds)."""
	degrees = int(np.floor(angle/np.pi*180.))
	minutes = int(np.floor((angle/np.pi*180. - degrees)*60.))
	seconds = ((angle/np.pi*180. - degrees)*60. - minutes)*60.
	
	return '{0:d}d{1:d}m{2:.3f}s'.format(degrees,minutes,seconds)

def arc2radians(arc):
	"""Convert arc (degrees, minutes, seconds) to radians. The arc can
	   be given as a string (e.g. '30d15m27.21s') or as a tuple
	   (e.g. (30,15,27.21))."""
	
	#if the arc is given as a string, parse it
	if type(arc) is str:
		degrees,min_sec = arc.split("d")
		minutes,seconds = min_sec.split("m")
		seconds = seconds.strip("s")
		
		degrees = int(degrees)
		minutes = int(minutes)
		seconds = float(seconds)
	else: #if the arc is given as a tuple, unpack it
		degrees,minutes,seconds = arc
	
	#compute the corresponding angle in radians
	radians = (degrees + minutes/60. + seconds/3600.)/180.*np.pi
	
	return radians

def hourangle2radians(hourangle):
	"""Convert hourangle (hours, minutes, seconds) to radians. 
	   The hourangle can be given as a string (e.g. '12h15m27.21s')
	   or as a tuple (e.g. (12,15,27.21))."""
	
	#if the angle is given as a string, parse it
	if type(hourangle) is str:
		hours,min_sec = hourangle.split("h")
		minutes,seconds = min_sec.split("m")
		seconds = seconds.strip("s")
		
		hours = int(hours)
		minutes = int(minutes)
		seconds = float(seconds)
	else: #if the angle is given as a tuple, unpack it
		hours,minutes,seconds = arc
	
	#compute the corresponding angle in radians
	radians = (hours + minutes/60. + seconds/3600.)/12.*np.pi
	
	return radians
		
