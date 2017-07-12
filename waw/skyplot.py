import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

def ha_ticks(rotate=0.):
    """Returns tick positions and tick labels for 2-hourangle-spaced meridians in sky plot."""
    xticks = np.linspace(0,2.*np.pi,13,endpoint=True)
    
    #apply rotation
    xticks += rotate
    xticks[xticks>2.*np.pi] = xticks[xticks>2.*np.pi] - 2.*np.pi
    xticks[xticks<0.] = xticks[xticks<0.] + 2.*np.pi
    
    xlabels = ['{}h'.format(2*i) for i in range(12)]
    #xlabels[0] = ''
    
    return xticks,xlabels

def deg_ticks():
    """Returns tick positions and tick labels for 15-degrees-spaced parallels in sky plot."""
    yticks = np.linspace(-np.pi/2.,np.pi/2.,13,endpoint=True)
    ylabels = [r'{0:d}$^\circ$'.format(-90 + 15*i) for i in range(13)]
    ylabels[0] = ''
    ylabels[-1] = ''
    
    return yticks,ylabels

def plot_healpix_map(m,nest=False, ax=None, projection='rectilinear', rotate=0., **kwargs):
    """Project a healpix map 'm' on the given axes 'ax' using pcolormesh. If no 'ax'
    is given, then create a new one with the given projection. Keyword arguments
    of pcolormesh can be given using **kwargs."""
    
    #if no axis is given, create one
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection=projection)
    
    #find map resolution
    nside = hp.npix2nside(len(m))
    res = hp.nside2resol(nside)
    
    #create a theta,phi grid of (eight times) the map average resolution
    theta,phi = np.meshgrid(np.arange(0.,np.pi,res/8.),np.arange(0.,2.*np.pi,res/8.))
    
    #create rotated phi angle and wrap over periodicity interval
    phi_rot = phi + rotate
    phi_rot[phi_rot>2.*np.pi] = phi_rot[phi_rot>2.*np.pi] - 2.*np.pi
    phi_rot[phi_rot<0.] = phi_rot[phi_rot<0.] + 2.*np.pi
    
    #evaluate the map on the grid
    M = m[hp.ang2pix(nside,theta,phi)]
    #mask invalid pixels
    M = np.ma.masked_invalid(M)
    
    #create rotated phi angle and wrap over periodicity interval
    phi_rot = phi + rotate
    phi_rot[phi_rot>2.*np.pi] = phi_rot[phi_rot>2.*np.pi] - 2.*np.pi
    phi_rot[phi_rot<0.] = phi_rot[phi_rot<0.] + 2.*np.pi
    
    #plot the map using pcolormesh
    lon = phi_rot
    lat = np.pi/2. - theta
    cm = ax.pcolormesh(lon,lat,M,**kwargs)
    ax.set_facecolor('gray')
    cb = plt.colorbar(cm)
    
    return ax,cb
    

def plot_ra_dec(ra, dec, ax=None, projection='rectilinear', rotate=0., **kwargs):
    """Plot a line, whose points are defined in RA,Dec coordinates, on the axis ax.
    If no axis is given, the a new one is defined, using the given projection.
    The rotate keyword can be used to rotate the projection by a given angle (in radians).
    If the line crosses the ra+rot=2pi line, points on opposite sides are disconnected.
    The input RA and Dec *must* anyway be in radians."""
    
    #create rotated right ascension
    ra_rot = ra+rotate
    
    if np.isscalar(ra_rot):
        if ra_rot>2.*np.pi:
            ra_rot -= 2.*np.pi
        if ra_rot<0.:
            ra_rot += 2.*np.pi
    else:
        ra_rot[np.ma.greater(ra_rot,2.*np.pi)] = ra_rot[np.ma.greater(ra_rot,2.*np.pi)] - 2.*np.pi
        ra_rot[np.ma.less(ra_rot,0.)] = ra_rot[np.ma.less(ra_rot,0.)] + 2.*np.pi
    
    # if no axis given create one
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection=projection)
    
    #find if the line crosses the ra_rot=2pi line and divide the line
    radiff = np.ma.diff(ra_rot)
    transitions = np.argwhere(np.ma.greater(radiff*radiff,np.pi**2))
    
    if len(transitions)>0:
        transitions = np.concatenate(([[-1]],transitions))      
        transitions = np.concatenate((transitions,[[len(ra)-2]]))
    else:
        transitions=[[-1],[len(ra)-2]]
    
    lines = []
    
    for j in range(len(transitions)-1):
        rg = [i for i in range(transitions[j][0]+1,transitions[j+1][0]+1)]
        lines.append(ax.plot(ra_rot[rg],dec[rg],**kwargs))
        
    return lines


def hourangle_ticks(ax, axis='x', invert=True, rotate=0.):
    """Set hourangle ticks on the RA axis, which can be the 'x' axis (default) 
    or the 'y' axis of the axes 'ax' (set the axis keyword, e.g. axis='y').
    If the 'invert' keyword is set, then invert the axis. The ticks are set 
    assuming that the actual RA units are radians."""
    
    haticks,halabels = ha_ticks(rotate)
    
    if axis=='x':
        ax.set_xticks(haticks)
        ax.set_xticklabels(halabels)
        if invert:
            ax.invert_xaxis()
    elif axis=='y':
        ax.set_yticks(haticks)
        ax.set_yticklabels(halabels)
        if invert:
            ax.invert_yaxis()

def degree_ticks(ax, axis='y'):
    """Set degree ticks on the Dec axis, which can be the 'y' axis (default)
    or the 'y' axis of the axes 'ax' (set the axis keyword, e.g. axis='x')."""
    degticks,deglabels = deg_ticks()
    
    if axis=='y':
        ax.set_yticks(degticks)
        ax.set_yticklabels(deglabels)
    else:
        ax.set_xticks(degticks)
        ax.set_xticklabels(deglabels)


def scatter_ra_dec(ra, dec, ax=None, projection='rectilinear', rotate=0., **kwargs):
    """Plot a series of points, defined in RA,Dec coordinates, on the axis ax.
    If no axis is given, the a new one is defined, using the given projection.
    If the hourangle_ticks keyword is set to True, then hourangle ticks and labels are drawn on the plot.
    If the deg_ticks keyword is set to True, then degrees ticks and labels are drawn on the y axis.
    The input RA and Dec *must* anyway be in radians."""
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection=projection)
    
    #create rotated right ascension
    ra_rot = ra+rotate
    
    if np.isscalar(ra_rot):
        if ra_rot>2.*np.pi:
            ra_rot -= 2.*np.pi
        if ra_rot<0.:
            ra_rot += 2.*np.pi
    else:
        ra_rot[ra_rot>2.*np.pi] = ra_rot[ra_rot>2.*np.pi] - 2.*np.pi
        ra_rot[ra_rot<0.] = ra_rot[ra_rot<0.] + 2.*np.pi
        
    points = ax.scatter(ra_rot,dec,**kwargs)
    
    return ax,points


def plot_lon_lat(lon, lat, ax=None, projection='mollweide', **kwargs):
    """Plot a line, whose points are defined in lon,lat coordinates, on the axis ax.
    If no axis is given, the a new one is defined, using the given projection.
    If the line crosses the lon=2pi line, points on opposite sides are disconnected."""
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection=projection)
    
    #find if the line crosses the ra=2pi line and divide the line
    transitions = np.argwhere(np.diff(lon)*np.diff(lon)>np.pi**2)
    
    if len(transitions)>0:
        transitions = np.concatenate(([[-1]],transitions))      
        transitions = np.concatenate((transitions,[[len(lon)-2]]))
    else:
        transitions=[[-1],[len(lon)-2]]
    
    lines = []
    
    for j in range(len(transitions)-1):
        rg = [i for i in range(transitions[j][0]+1,transitions[j+1][0]+1)]
        lines.append(ax.plot(lon[rg]-np.pi,lat[rg],**kwargs))
    
    return lines

        
    
