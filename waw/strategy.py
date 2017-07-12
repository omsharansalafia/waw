import numpy as np
import healpy as hp
from .probability import sky_pos_cond_prob_gt
from .emcounterparts import sgrb_afterglow


def detectability_maps(nu_obs, Flim, t, binary_params, counterpart=sgrb_afterglow, nside=64, verbose=False,
                       limit_to_region=None):
    """
    Compute the detectability maps P(F(t)>Flim|ra,dec,...) for a given EM
    counterpart, at a given obs frequency, with a given flux limit, at
    given times after the merger.


    Parameters:
    -----------
    nu_obs: scalar
        Observing frequency in Hz.
    Flim: scalar
        Limiting flux density of the search in mJy.
    t: numpy 1D array
        The times [in days] at which the detectability maps must be computed.
    binary_params: dictionary
        Dictionary containing the parameters of the posterior samples. The
        dictionary is passed to the counterpart function that computes/retrieves
        the lightcurves. The dictionary *must* contain at least the keys 'ra'
        and 'dec' that contain the sky positions of the posterior samples.
    counterpart: function
        The function that is used to compute/retrieve the counterpart lightcurves.
        It must comply with the input/output specifications defined in
        emcounterparts.IO_specifications.
    nside: int
        The nside parameter that defines the healpix resolution of the output
        detectability maps.
    verbose: boolean
        If True, print the advancement status of the lightcurve computation/retrieval.
    limit_to_region: numpy 1D array
        A healpix map of booleans. The detectability maps will be computed only
        at sky positions that correspond to True pixels in this map. If None,
        the computation will be all-sky.

    Returns:
    --------
    detmaps: ndarray
        The shape of the output ndarray is (nt,npix), where nt is the length
        of the input t array, and npix = 12*nside**2. In practice, detmaps[i]
        is the detectability map at time t[i]. Conversely, detmaps[:,p] is
        the time-dependent detectability at sky position p (p is the healpix
        index! The actual coordinates are given by hp.pix2ang(nside,p)).
    """

    # compute the lightcurve of the counterpart for each posterior sample
    F = counterpart(binary_params, nu_obs, t, verbose)

    # compute the detectability maps
    detmaps = np.empty([len(t), hp.nside2npix(nside)])

    if verbose:
        print("")
    for i in range(len(t)):
        if verbose:
            print(" #### computing detmap {0:d} of {1:d} ({2:.1f} percent completed) ####".format(i + 1, len(t),i / len(t) * 100.),end="\r")
        detmaps[i] = sky_pos_cond_prob_gt(F[:, i], Flim, binary_params['ra'], binary_params['dec'], nside,limit_to_region)

    if verbose:
        print(" #### {} detmaps computed (100 percent completed) ####".format(len(t)))

    return detmaps


def construct_followup_strategy(skymap, detmaps, t_detmaps, Afov, T_int, T_available, min_detectability=0.01, limit_to_region=None):
    """
    Construct the EM follow-up as in Salafia+17.

    Parameters:
    -----------
    skymap: 1D numpy array
        The healpix map of GW posterior sky position probability per unit area.
    detmaps: ndarray
        The detectability maps as computed by ::function::detectability_maps.
    t_detmaps: 1D array
        The times (in days) corresponding to the detectability maps provided.
    Afov: scalar
        The field of view of the observing instrument in square degrees.
    T_int: scalar
        The exposure/integration time (in seconds) that corresponds to the
        search limiting flux.
    T_available: tuple
        The starting and ending times (in days after the merger) of the
        available time windows. Must be an even number! (each window has
        a starting time and an ending time).
    min_detectability: scalar
        The minimum value of the detectability for which an observation
        can be scheduled.
    ...

    Return:
    -------
    strategy: 1D masked numpy array
        A healpix map containing the observation times corresponding
        to the follow-up strategy

    """

    # make sure that the detectability is above the minimum at some point
    if np.all(detmaps[np.isfinite(detmaps)] < min_detectability):
        print("No point in the detmaps is above the minumum required detectability.")
        return None

    # determine the nside that makes the pixel area closest to the fov
    # area, but not larger (FIXME: a MOC based representation of the fov
    # would do much better!)
    nsides = 2. ** np.arange(2, 10)
    areas = hp.nside2pixarea(nsides, degrees=True)
    ratio = Afov / areas
    nside = int(np.min(nsides[ratio > 1.]))

    # rescale the integration time to the effective one corresponding to
    # that pixel area
    Apixel = hp.nside2pixarea(nside, degrees=True)
    T = T_int * Apixel / Afov  # this is the time needed to cover one healpix pixel

    # bring the limit_to_region to the correct resolution
    if limit_to_region is None:
        region = np.ones(hp.nside2npix(nside),dtype=bool)
    else:
        region = hp.ud_grade(limit_to_region.astype(int), nside).astype(bool)

    # limit the detmaps to the region (to save memory and computation time)
    # after degrading/upgrading to the correct resolution
    ridx = np.arange(hp.nside2npix(nside))[region] # this keeps track of the original healpix indices corresponding to the region

    dm = np.empty([len(detmaps), len(region[region])])

    for i in range(len(detmaps)):
        dm[i] = hp.ud_grade(detmaps[i], nside)[region]

    # bring the skymap to the same resolution, and take only the region
    sm = hp.ud_grade(skymap,nside)[region]
    # also, find the descending probability sorted indices of the skymap
    descending_prob_idx = np.argsort(sm)[::-1]

    # how much total observing time is available? Just sum the differences
    # between ending and starting times of the available time windows,
    # then convert to seconds
    n_windows = len(T_available) // 2
    tot_obs_time = np.sum(np.diff(T_available)[::2]) * 86400.
    # compute the number of time slots, and their starting times in days.
    # Note that the number of slots per window must be an integer, thus
    # a the end of each window there might be some remainder time unused
    slots_in_window = (np.diff(T_available)[::2] * 86400./T).astype(int)
    n_slots = np.sum(slots_in_window)
    t0_slot = np.empty(n_slots)
    k = 0
    for i in range(n_windows):
        for j in range(slots_in_window[i]):
            t0_slot[k] = T_available[2 * i] + j * T/86400.
            k = k + 1
    # mark all slots as available
    available_slots = np.ones(n_slots,dtype=bool)

    # the strategy will be a healpix map of observation times
    strategy = np.ones(hp.nside2npix(nside))*np.nan

    # assign the available time slots to the skymap pixels in order
    # of descending sky position probability. Each pixel is assigned
    # the available time slot where the detectability is highest
    for p in descending_prob_idx:
        try: # find the time of best detectability
            detp = np.interp(t0_slot[available_slots],t_detmaps,dm[:,p])
            i_best = np.argmax(detp)
            i0_best = np.arange(len(available_slots))[available_slots][i_best] #original index of the best time slot
        except:
            continue

        # do not assign observation if the best detectability is below the requested limit
        if detp[i_best]<min_detectability:
            continue
        else:
            strategy[ridx[p]]=t0_slot[i0_best] # assign the observation time
            available_slots[i0_best]=False # mark the assigned slot as not available anymore

    return np.ma.masked_invalid(strategy)


