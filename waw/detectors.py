import astropy.coordinates as apcoords
import astropy.time as aptime
import astropy.units as u
import vectors as vec
import numpy as np
from skimage import measure
from scipy.interpolate import LinearNDInterpolator
import healpy as hp
import skycoordinates

#WGS-84 parameters
a = 6.378137e6 #m
b = 6.356752314e6 #m

def Rwgs84(lat):
	"""The radius of curvature of the wgs84 oblate spheroid earth model,
	as a function of the local latitude."""
	return a**2/np.sqrt(a**2*np.cos(lat)**2+b**2*np.sin(lat)**2)

def local_north(lon,lat):
	"""The local north-pointing unit vector on the wgs84 surface."""
	return np.array([-np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)])

def local_east(lon):
	"""The local east-pointing unit vector on the wgs84 surface."""
	return np.array([-np.sin(lon),np.cos(lon),0.])


class Detector:
	"""This class stores detector location and other parameters. It is
	initialized providing the location 'loc' as an astropy.coordinates.Earthlocation
	object, and the x_arm and y_arm unit vectors as 3D numpy arrays."""
	
	def __init__(s, loc, xarm, yarm, BNSrange=200.):
		s.loc = loc
		s.lon = float(s.loc.longitude/u.rad)
		s.lat = float(s.loc.latitude/u.rad)
		s.x_arm = xarm
		s.y_arm = yarm
		s.z_arm = vec.CrossP(xarm,yarm)
		s.BNSrange=BNSrange
		
		#for Schutz+11 arbitrary position Fplus&Fcross computation
		s.chi = np.arccos(vec.DotP(0.5*(s.x_arm+s.y_arm),local_east(s.lon)))
		s.sinb = np.sin(s.lat)
		s.cosb = np.cos(s.lat)
		s.sin2b = np.sin(2.*s.lat)
		s.cos2b = np.cos(2.*s.lat)
		s.sin2chi = np.sin(2.*s.chi)
		s.cos2chi = np.cos(2.*s.chi)
		s.sineta = np.sin(np.arccos(vec.DotP(s.x_arm,s.y_arm)))
	
	def a_sch(s,th,phi):
		three_cos2th = 3.-np.cos(2.*th)
		sin2th = np.sin(2.*th)
		return (1./16.)*s.sin2chi*(3.-s.cos2b)*three_cos2th*np.cos(2.*(phi+s.lon)) +\
		0.25*s.cos2chi*s.sinb*three_cos2th*np.sin(2.*(phi+s.lon)) +\
		0.25*s.sin2chi*s.sin2b*sin2th*np.cos(phi+s.lon) +\
		0.50*s.cos2chi*s.cosb*sin2th*np.sin(phi+s.lon) +\
		0.75*s.sin2chi*s.cosb**2*np.sin(th)**2
	
	def b_sch(s,th,phi):
		costh = np.cos(th)
		sinth = np.sin(th)
		cos2pl = np.cos(2.*(phi+s.lon))
		cospl = np.cos(1.*(phi+s.lon))
		sin2pl = np.sin(2.*(phi+s.lon))
		sinpl = np.sin(1.*(phi+s.lon))
		return s.cos2chi*s.sinb*costh*cos2pl - 0.25*s.sin2chi*(3.-s.cos2b)*costh*sin2pl +\
		s.cos2chi*s.cosb*sinth*cospl - 0.5*s.sin2chi*s.sin2b*sinth*sinpl
	
	def Fplus_sch(s,lon,lat,psi):
		return s.sineta*(s.a_sch(np.pi/2.-lat,-lon)*np.cos(2.*psi) + s.b_sch(np.pi/2.-lat,-lon)*np.sin(2.*psi))
	
	def Fcross_sch(s,lon,lat,psi):
		return s.sineta*(s.b_sch(np.pi/2.-lat,-lon)*np.cos(2.*psi) - s.a_sch(np.pi/2.-lat,-lon)*np.sin(2.*psi))
	
	def local_spherical(s,lon,lat):
		"""Returns the spherical coordinates (in the frame where the
		detector is at the origin and the arms are aligned with the axes)
		of the unit vector pointing to lon,lat."""
		
		#construct the unit vector pointing at lon,lat
		x = np.cos(lat)*np.cos(lon)
		y = np.cos(lat)*np.sin(lon)
		z = np.sin(lat)
		
		v = np.array([x,y,z])
		
		#compute the components in the detector system
		X = vec.DotP(v,s.x_arm)
		Y = vec.DotP(v,s.y_arm)
		Z = vec.DotP(v,s.z_arm)
		
		#compute the corresponding spherical coordinates
		phi = np.arctan2(Y,X)
		theta = np.arccos(Z)
		
		return theta,phi
		
	
	def Fplus(s,lon,lat,psi):
		"""Returns the Fplus antenna pattern component as defined in Schutz+11,
		corresponding to geographic coordinates lon,lat. To turn these in sky
		coordinates, the latitude can be set equal to the declination; the
		longitude must be computed as the right ascension - the 'greenwich
		mean sidereal time' at the time of observation."""
		
		#compute local spherical coordinates corresponding to lon,lat
		theta,phi = s.local_spherical(lon,lat)
		
		#compute trigonometric functions
		cth = np.cos(theta)
		c2phi = np.cos(2.*phi)
		c2psi = np.cos(2.*psi)
		s2phi = np.sin(2.*phi)
		s2psi = np.sin(2.*psi)
		
		#compute antenna pattern
		return 0.5*(1. + cth**2) * c2phi * c2psi - cth * s2phi * s2psi
		
	def Fcross(s,lon,lat,psi):
		"""Returns the Fplus antenna pattern component as defined in Schutz+11,
		corresponding to geographic coordinates lon,lat. To turn these in sky
		coordinates, the latitude can be set equal to the declination; the
		longitude must be computed as the right ascension - the 'greenwich
		mean sidereal time' at the time of observation."""
		
		#compute local spherical coordinates corresponding to lon,lat
		theta,phi = s.local_spherical(lon,lat)
		
		#compute trigonometric functions
		cth = np.cos(theta)
		c2phi = np.cos(2.*phi)
		c2psi = np.cos(2.*psi)
		s2phi = np.sin(2.*phi)
		s2psi = np.sin(2.*psi)
		
		#compute antenna pattern
		return 0.5*(1. + cth**2) * c2phi * s2psi + cth * s2phi * c2psi
	
	def Faverage(s,lon,lat):
		"""Returns the average antenna pattern F_av = sqrt((Fplus^2+Fcross^2)/2),
		corresponding to geographic coordinates lon,lat. To turn these in sky
		coordinates, the latitude can be set equal to the declination; the
		longitude must be computed as the right ascension - the 'greenwich
		mean sidereal time' at the time of observation."""
		
		#compute local spherical coordinates corresponding to lon,lat
		theta,phi = s.local_spherical(lon,lat)
		
		#compute trigonometric functions
		cth = np.cos(theta)
		c2phi = np.cos(2.*phi)
		s2phi = np.sin(2.*phi)
		
		#compute antenna pattern
		return np.sqrt(0.5*(0.25*(1. + cth**2)**2 * c2phi**2 + cth**2 * s2phi**2))
	
	def Faverage_contours(s,nside=128,levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
		"""Returns the contour lines at the given levels of the average antenna pattern"""
		
		#define a grid over which to evaluate the antenna pattern
		lat = np.linspace(-np.pi/2.,np.pi/2.,100)
		lon = np.linspace(0.,2.*np.pi,100)
		
		#evaluate and keep track of both coordinates and indices where
		#the evaluation is done
		values = np.zeros([100*100,2])
		points = np.zeros([100*100,2])
		MAP = np.zeros([100,100])
		
		k=0
		for i in range(100):
			for j in range(100):
				MAP[i,j] = s.Faverage(lon[i],lat[j])
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
			
			contours.append(whole_contour.transpose())
		
		return contours

class DetectorNetwork:
	
	def __init__(s, detectors):
		s.dets = detectors
	
	def time_delay(s,RA,Dec,gpstime,det1,det2):
		"""Returns the delay time between detectors det1 and det2 for a 
		   signal coming from Ra,Dec at gpstime."""
		
		#compute the distance between det1 and det2 and derive the maximum Dt
		v0 = np.array(s.dets[det2].loc.to(u.m).value)-np.array(s.dets[det1].loc.to(u.m).value)
		d = np.sqrt(vec.DotP(v0,v0))
		Dtmax = d/3e8 #d in meters, c in meters/seconds --> Dtmax in seconds
		v0 = v0/d #make it unit
		
		#convert RA,Dec into lon,lat
		lat = Dec
		lon = skycoordinates.RA2lon(RA,gpstime)
		
		#compute the unit vector pointing at lon,lat
		clat = np.cos(lat)
		slat = np.sin(lat)
		clon = np.cos(lon)
		slon = np.sin(lon)
		
		v = np.array([clat*clon,clat*slon,slat])
		
		#compute the angle between v0 and v
		ctheta = vec.DotP(v,v0)/np.sqrt(vec.DotP(v,v)*vec.DotP(v0,v0))
		
		#return the resulting delay time
		return Dtmax*ctheta
		
	
	def Constant_time_delay_line(s, Dt, det1, det2):
		"""Returns the line of constant time delay (Dt) between detectors 
		   det1 and det2 line in lon,lat coordinates."""
		
		#compute the distance between det1 and det2 and derive the maximum Dt
		v0 = np.array(s.dets[det2].loc.to(u.m).value)-np.array(s.dets[det1].loc.to(u.m).value)
		d = np.sqrt(vec.DotP(v0,v0))
		Dtmax = d/3e8 #Dtmax in seconds
		
		#if Dt>Dtmax raise error:
		if abs(Dt)>abs(Dtmax):
			print('The required Dt is larger than the max possible Dt '+\
			'for a signal traveling at light speed between detectors {} '.format(det1)+\
			'and {}, namely Dtmax = {} milliseconds'.format(det2,Dtmax*1000.))
			print('Using Dtmax instead.')
			Dt = Dtmax
		
		#compute the distance vector between det1 and det2
		v0 = v0/d #make it unit
		
		#create a perpendicular vector
		v1 = np.random.uniform(-1.,1.,3) #take a random vector
		while vec.DotP(v1,v0)**2 == vec.DotP(v1,v1): #take another one if it is parallel to v0
			v1 = np.random.uniform(-1.,1.,3)
		#orthogonalize a' la Gram-Schmidt
		v1 = v1 - (vec.DotP(v1,v0)/np.sqrt(vec.DotP(v0,v0)))*v0
		v1 = v1/np.sqrt(vec.DotP(v1,v1)) #make unit
		#compute the third member of the basis using the cross product
		v2 = vec.CrossP(v0,v1)
		
		#now that we have a orthogonal basis, we can work out the constant time delay line
		
		#compute the angle between v0 and the arrival direction of GW that gives the required
		#time delay
		theta = np.arccos(Dt/Dtmax)
		ctheta = np.cos(theta)
		stheta = np.sin(theta)
		
		#compute the unit vectors pointing at the possible arrival directions
		th = np.linspace(0.,2.*np.pi+0.0001,100) #create a grid of angles
		vpx = np.sin(th)*v1[0] + np.cos(th)*v2[0] #compute the components of the perpendicular vectors making an angle th with the v2 vector
		vpy = np.sin(th)*v1[1] + np.cos(th)*v2[1]
		vpz = np.sin(th)*v1[2] + np.cos(th)*v2[2]
		#the above vectors define the 0 time delay line. Now we want to obtain the Dt time delay line
		vtheta = np.zeros([len(th),3])
		for th_i in range(len(th)):
			vtheta[th_i,0] = vpx[th_i]*stheta + v0[0]*ctheta
			vtheta[th_i,1] = vpy[th_i]*stheta + v0[1]*ctheta
			vtheta[th_i,2] = vpz[th_i]*stheta + v0[2]*ctheta
		#now we want the lon,lat coordinates corresponding to this family of vectors
		lon = np.zeros(len(th))
		lat = np.zeros(len(th))
		
		for th_i in range(len(th)):
			lon[th_i] = np.arctan2(vtheta[th_i,1],vtheta[th_i,0])
			lat[th_i] = np.arcsin(vtheta[th_i,2])
		
		return np.array([lon,lat])
			
	
	def Faverage(s,lon,lat):
		F = 0.
		norm = 0.
		for det in s.dets:
			F += (det.BNSrange**2*det.Faverage(lon,lat))**2
			norm += det.BNSrange**4
		
		return np.sqrt(F/norm)
	
	def Faverage_contours(s,nside=128,levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
		"""Returns the contour lines at the given levels of the average antenna pattern"""
		
		#define a grid over which to evaluate the antenna pattern
		lat = np.linspace(-np.pi/2.,np.pi/2.,100)
		lon = np.linspace(0.,2.*np.pi,100)
		
		#evaluate and keep track of both coordinates and indices where
		#the evaluation is done
		values = np.zeros([100*100,2])
		points = np.zeros([100*100,2])
		MAP = np.zeros([100,100])
		
		k=0
		for i in range(100):
			for j in range(100):
				MAP[i,j] = s.Faverage(lon[i],lat[j])
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


#----- LIGO Hanford ----------
#Earth location
H1loc = apcoords.EarthLocation(-2.16141492636e+06*u.m,-3.83469517889e+06*u.m,4.60035022664e+06*u.m)
#x-arm and y-arm unit vectors
H1xarm = np.array([ -0.22389266154,0.79983062746,0.55690487831])
H1yarm = np.array([-0.91397818574,0.02609403989,-0.40492342125])

H1 = Detector(H1loc,H1xarm,H1yarm)

H1_early = Detector(H1loc,H1xarm,H1yarm,BNSrange=55.)
H1_mid = Detector(H1loc,H1xarm,H1yarm,BNSrange=100.)

#----- LIGO Livingston -------
#Earth location
L1loc = apcoords.EarthLocation(-7.42760447238e+04*u.m,-5.49628371971e+06*u.m,3.22425701744e+06*u.m)
#x-arm and y-arm unit vectors
L1xarm = np.array([-0.95457412153,-0.14158077340,-0.26218911324])
L1yarm = np.array([0.29774156894,-0.48791033647,-0.82054461286])

L1 = Detector(L1loc,L1xarm,L1yarm)

L1_early = Detector(L1loc,L1xarm,L1yarm,BNSrange=55.)
L1_mid = Detector(L1loc,L1xarm,L1yarm,BNSrange=55.)

#----- Virgo -------
#Earth location
V1loc = apcoords.EarthLocation(4.54637409900e+06*u.m,8.42989697626e+05*u.m,4.37857696241e+06*u.m)
#x-arm and y-arm unit vectors
V1xarm = np.array([-0.70045821479,0.20848948619,0.68256166277])
V1yarm = np.array([-0.05379255368,-0.96908180549,0.24080451708])

Virgo = Detector(V1loc,V1xarm,V1yarm)

Virgo_early = Detector(V1loc,V1xarm,V1yarm,BNSrange=30.)

#----- H1L1 network
HL = DetectorNetwork([H1,L1])
HL_early = DetectorNetwork([H1_early,L1_early])
#----- H1L1Virgo network
HLV = DetectorNetwork([H1,L1,Virgo])
HLV_early = DetectorNetwork([H1_mid,L1_mid,Virgo_early])
