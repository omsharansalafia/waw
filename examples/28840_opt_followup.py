from context import waw
import numpy as np
import healpy as hp
from waw import probability
from waw import contours
from waw import skyplot
from waw.strategy import detectability_maps
from waw.strategy import construct_followup_strategy


# in the following example, the posterior samples of event 28840 are
# loaded, and detectability maps for optical follow up of sgrb afterglow
# are computed. These are used to construct a follow-up strategy.
# The result should reproduce Fig. 7 from Salafia+17.

# load psamples
ra,dec,dL,iota = np.load('psamples_F2Y28840.npy')

# compute approximate skymap and 90% confidence region
print(" Computing approximate skymap for event 28840...")
sm = probability.approximate_skymap(ra, dec, nside=128)
cm = probability.integrated_probability(sm)
cr90 = cm < 0.9

# find contours of 50% and 90% c.r.
print(" Computing 50 and 90 percent confidence region contours...")
cnt = contours.hpix_contours(cm)

# choose obs frequency and limiting flux
nu_obs = 4.6e14  # Hz -- optical r band
Flim = 6e-4  # mJy (~ 24.5 AB magnitude)

# compute detmaps
t = np.logspace(0., np.log10(5.125), 10)
detmaps = detectability_maps(nu_obs, Flim, t, {'ra': ra, 'dec': dec, 'dL': dL, 'iota': iota}, verbose=True,limit_to_region=cr90)
np.save("dm.npy",detmaps)
#detmaps = np.load("dm.npy")

# create strategy
T_available = (1.,1.125,2.,2.125,3.,3.125,4.,4.125,5.,5.125) # 3h per night, starting 24h after merger
strategy = construct_followup_strategy(sm,detmaps,t,1.,1000.,T_available,limit_to_region=cr90)

# plot the strategy
skyplot.plt.rcParams['font.family']='Liberation Serif'
skyplot.plt.rcParams['font.size']='11'
skyplot.plt.rcParams['figure.figsize']='10.,5.'
skyplot.plt.rcParams['figure.autolayout']=True

rot = -0.5
ax,cb=skyplot.plot_healpix_map(strategy,rotate=rot,cmap=skyplot.plt.cm.Reds)

for c in cnt: # plot the 50% and 90% confidence contours
    skyplot.plot_ra_dec(c[0],c[1],ax=ax,rotate=rot,linestyle='-',color='red')

cb.set_label("obs time [days after merger]")

# set zoom and axes ticks
skyplot.hourangle_ticks(ax,rotate=rot)
skyplot.degree_ticks(ax)

xmin = 6.
xmax = 2.135
ymin = -0.8
ymax = 0.5

ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])

ax.set_xlabel('RA')
ax.set_ylabel('Dec')

ax.grid()

skyplot.plt.show()

