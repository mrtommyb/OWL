#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the HLM project.
Copyright 2014 Dan Foreman-Mackey and David W. Hogg.

Bugs:
-----
* Things called "flux" should be called "intensities".
* Global variables with `API()` and `XGRID, YGRID, NINEBYSIX`.
"""

import numpy as np
import astropy.io.fits as pyfits
import kplr
client = kplr.API()

# make shit for least squares
# BUG: I am sure there is a one-liner for this
XGRID, YGRID = np.meshgrid(np.arange(3) - 1., np.arange(3) - 1.)
XGRID, YGRID = XGRID.flatten(), YGRID.flatten()
print XGRID, YGRID
NINEBYSIX = np.zeros((9,6))
NINEBYSIX[:,0] = 1.
NINEBYSIX[:,1] = XGRID
NINEBYSIX[:,2] = YGRID
NINEBYSIX[:,3] = XGRID * XGRID
NINEBYSIX[:,4] = XGRID * YGRID
NINEBYSIX[:,5] = YGRID * YGRID
print NINEBYSIX

def get_target_pixel_file(kicid, quarter):
    kic = client.star(kicid)
    tpfs = kic.get_target_pixel_files(short_cadence=False)
    tpfs = filter(lambda t: t.sci_data_quarter == quarter, tpfs)
    if not len(tpfs):
        raise ValueError("No dataset for that quarter")
    return tpfs[0]

def get_max_pixel(flux):
    """
    input:
    * `flux` - `np.array` shape `(nt, ny, nx)` pixel intensities (not fluxes!)

    output:
    * `xc, yc` - integer max pixel location

    comments:
    * Asserts that the max pixel is not on an edge; might be dumb.
    """
    nt, ny, nx = flux.shape
    max_indx = np.argmax(np.mean(flux, axis=0))
    xc, yc = max_indx % nx, max_indx / nx
    assert (xc > 0)
    assert (yc > 0)
    assert ((xc + 1) < nx)
    assert ((yc + 1) < ny)
    return xc, yc

def get_one_centroid(one_flux, invvar, xc, yc):
    """
    input:
    * `one_flux` - `np.array` shape `(ny, nx)` pixel intensities (not fluxes!)
    * `invvar` - mask or weight image same shape as `one_flux`
    * `xc, yc` - max or "central" pixel.

    output:
    * `xc, yc` - floating-point centroid based on quadratic fit

    comments:
    * does (A.T C.inv A).inv A.T C.inv b

    bugs:
    * Currently does NOTHING.
    * Ought to do a solve operation, not an inverse operation.
    * Ought to do 9-fold leave-one-out set of operations for robustness.
    * Ought to take in a prior / regularization that draws the answer towards a default (floating point) centroid.
    """
    ninebyone = (one_flux[yc + YGRID, xc + XGRID]).flatten()
    iv = (invvar[yc + YGRID, xc + XGRID]).flatten()
    if np.sum(iv > 0.) < 7:
        print "get_one_centroid(): not enough data to support fit"
        print "get_one_centroid():", iv
        print "get_one_centroid(): returning integer centroid"
        return xc, yc
    numerator = np.dot(np.transpose(NINEBYSIX), iv * ninebyone)
    denominator = np.dot(np.transpose(NINEBYSIX), iv[:, None] * NINEBYSIX)
    print numerator.shape, denominator.shape
    pars = np.dot(np.inverse(denominator), numerator)
    if (pars[3] > 0.) or (pars[5] > 0.) or (pars[4] * pars[4] > pars[3] * pars[5]): # is this correct?
        print "get_one_centroid(): fit bad"
        print "get_one_centroid():", pars
        print "get_one_centroid(): returning integer centroid"
        return xc, yc
    # now magic happens...
    return xc, yc

def get_all_centroids(flux, mask):
    """
    input:
    * `flux` - pixel intensities (not fluxes!)

    output:
    * `centroids` - some crazy object of centroids

    bugs:
    * Is `map()` dumb?
    * Should I be using a lambda function or something smarter?
    """
    xc, yc = get_max_pixel(flux)
    iv = zeros_like(flux[0])
    iv[(mask > 0)] = 1.
    def goc(c):
        return get_one_centroid(c, iv, xc, yc)
    return np.array(map(goc, flux))

if __name__ == "__main__":
    kicid = 3335426
    prefix = "kic_%08d" % (kicid, )
    tpf = get_target_pixel_file(kicid, 5)
    if False:
        fig = tpf.plot()
        fig.savefig(prefix + ".png")
    with tpf.open() as hdu:
        table = hdu[1].data
        mask = hdu[2].data
    time_in_kbjd = table["TIME"]
    # raw_cnts = table["RAW_CNTS"]
    bkg_sub_flux = table["FLUX"]
    xc, yc = get_max_pixel(bkg_sub_flux)
    centroids = get_all_centroids(bkg_sub_flux, mask)
    print np.median(bkg_sub_flux, axis=0)
    print mask
    print centroids.shape
