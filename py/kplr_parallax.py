#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the HLM project.
Copyright 2014 Dan Foreman-Mackey and David W. Hogg.

Bugs:
-----
* Contains useless photometry stuff that has nothing to do with astrometry.
* No testing framework; testing would rock this.
* Things called "flux" should be called "intensities".
* Global variables with `API()` and `XGRID, YGRID, NINEBYSIX`.
* Barely tested, and there are copious x <-> y issues possible.
"""

import numpy as np
import scipy.optimize as op
import kplr
client = kplr.API()

if False:
    from multiprocessing import Pool
    p = Pool(16)
    pmap = p.map
else:
    pmap = map

# make shit for least squares
# BUG: I am sure there is a one-liner for this
XGRID, YGRID = np.meshgrid(np.arange(3) - 1, np.arange(3) - 1)
XGRID, YGRID = XGRID.flatten(), YGRID.flatten()
NINEBYSIX = np.zeros((9,6))
NINEBYSIX[:,0] = 1.
NINEBYSIX[:,1] = XGRID
NINEBYSIX[:,2] = YGRID
NINEBYSIX[:,3] = XGRID * XGRID
NINEBYSIX[:,4] = XGRID * YGRID
NINEBYSIX[:,5] = YGRID * YGRID

def get_target_pixel_file(kicid, quarter):
    kic = client.star(kicid)
    tpfs = kic.get_target_pixel_files(short_cadence=False)
    tpfs = filter(lambda t: t.sci_data_quarter == quarter, tpfs)
    if not len(tpfs):
        raise ValueError("No dataset for that quarter")
    return tpfs[0]

def get_max_pixel(flux, mask):
    """
    input:
    * `flux` - `np.array` shape `(nt, ny, nx)` pixel intensities (not fluxes!)

    output:
    * `xc, yc` - integer max pixel location

    comments:
    * Asserts that the max pixel is not on an edge; might be dumb.
    """
    nt, ny, nx = flux.shape
    median_image = np.median(flux, axis=0)
    median_image[mask < 1] = 0.
    max_indx = np.argmax(median_image)
    xc, yc = max_indx % nx, max_indx / nx
    if (xc < 1) or (yc < 1) or ((xc + 2) > nx) or ((yc + 2) > ny):
        print "hlm.get_max_pixel(): ERROR: max pixel at image edge; failing!"
        print "hlm.get_max_pixel():", mask, np.median(flux, axis=0), xc, yc
        assert False
    return xc, yc

def get_one_centroid(one_flux, invvar, xc, yc):
    """
    input:
    * `one_flux` - `np.array` shape `(ny, nx)` pixel intensities (not fluxes!)
    * `invvar` - mask or weight image same shape as `one_flux`
    * `xc, yc` - max or "central" pixel; integer centroid

    output:
    * `[dxc, dyc, m]` - floating-point centroid offset and mask value based on quadratic fit

    comments:
    * does (A.T C.inv A).inv A.T C.inv b.
    * then does `sdenominator` and `snumerator` magic.

    bugs:
    * Ought to take in a prior / regularization that draws the answer towards a default (floating point) centroid.
    * Magic algebra ought to be checked.
    """
    ninebyone = (one_flux[yc + YGRID, xc + XGRID]).flatten()
    iv = (invvar[yc + YGRID, xc + XGRID]).flatten()
    intcentroid = np.array([xc, yc, 0.])
    if np.sum(iv > 0.) < 7:
        # print "hlm.get_one_centroid(): WARNING: not enough data to support fit"
        # print "hlm.get_one_centroid(): returning integer centroid"
        return intcentroid
    numerator = np.dot(np.transpose(NINEBYSIX), iv * ninebyone)
    denominator = np.dot(np.transpose(NINEBYSIX), iv[:, None] * NINEBYSIX)
    pars = np.linalg.solve(denominator, numerator)
    if (pars[3] > 0.) or (pars[5] > 0.) or (pars[4] * pars[4] > pars[3] * pars[5]): # is this correct?
        # print "hlm.get_one_centroid(): WARNING: fit bad"
        # print "hlm.get_one_centroid(): returning integer centroid"
        return intcentroid
    # now magic happens...
    sdenominator = 4. * pars[3] * pars[5] - pars[4] * pars[4]
    snumerator = np.array([pars[2] * pars[4] - 2. * pars[1] * pars[5],
                          pars[1] * pars[4] - 2. * pars[2] * pars[3]])
    if (sdenominator <= 0.) or not np.all(np.isfinite(snumerator)):
        # print "hlm.get_one_centroid(): WARNING: pars bad"
        # print "hlm.get_one_centroid(): returning integer centroid"
        return intcentroid
    centroid = intcentroid
    centroid[:2] += snumerator / sdenominator
    centroid[2] = 1.
    return centroid

def get_one_robust_centroid(one_flux, invvar, xc, yc):
    """
    input:
    * `one_flux` - `np.array` shape `(ny, nx)` pixel intensities (not fluxes!)
    * `invvar` - mask or weight image same shape as `one_flux`
    * `xc, yc` - max or "central" pixel.

    output:
    * `centroid` - `np.array` shape `(3, )` with floating-point centroid and mask value

    comments:
    * does a median of leave-one-out trials of `get_one_centroid()`.

    bugs:
    * Ought to take in a prior / regularization that draws the answer towards a default (floating point) centroid.
    """
    loo_iv = invvar[None, :, :] * (np.ones((9, )))[:, None, None]
    loo_iv[range(9), yc + YGRID, xc + XGRID] = 0. # create leave-one-out invvars
    def goc(iv):
        return get_one_centroid(one_flux, iv, xc, yc)
    centroids = np.array(pmap(goc, loo_iv))
    centroid = np.median(centroids, axis=0)
    centroid[2] = np.min(centroids[:, 2])
    return centroid

def get_all_centroids(flux, mask):
    """
    input:
    * `flux` - `np.array` shape `(nt, ny, nx)` of pixel intensities (not fluxes!)
    * `mask` - Kepler-generated pixel mask.

    output:
    * `centroids` - some crazy object of centroids

    bugs:
    * Is `map()` dumb?
    * Should I be using a lambda function or something smarter?
    """
    xc, yc = get_max_pixel(flux, mask)
    iv = np.zeros_like(flux[0])
    iv[mask > 0] = 1.
    def gorc(c):
        return get_one_robust_centroid(c, iv, xc, yc)
    return np.array(pmap(gorc, flux))

def get_centroid_derivatives(flux, mask):
    """
    input:
    * `flux` - `np.array` shape `(nt, ny, nx)` of pixel intensities (not fluxes!).
    * `mask` - Kepler-generated pixel mask.

    output:
    * `derivatives` - `np.array` shape `(ny, nx, 2)` of derivatives wrt centroid position.

    bugs:
    * Ought to be a map not a pair of nested for loops!
    """
    nt, ny, nx = flux.shape
    centroids = get_all_centroids(flux, mask)
    iv = centroids[:, 2]
    centroids = centroids[:, :2]
    centroids -= np.median(centroids, axis=0)
    A = np.hstack((np.ones(nt)[:, None], centroids))
    bkg_sub_flux[iv <= 0., :, :] = 0.
    ATA = np.dot(np.transpose(A), iv[:, None] * A)
    centroid_derivatives = np.zeros((ny, nx, 2))
    for yp in range(ny):
        for xp in range(nx):
            if mask[yp, xp] > 0:
                ATb = np.dot(np.transpose(A), iv * bkg_sub_flux[:, yp, xp])
                centroid_derivatives[yp, xp, :] = np.linalg.solve(ATA, ATb)[1:]
    return centroid_derivatives

def get_orthogonal_partial_basis(derivs, mask):
    """
    inputs:
    * `derivs` - `np.array` shape `(ny, nx, 2)` of positional derivatives from `get_centroid_derivatives`.
    * `mask` - Kepler mask, shape of `derivs`, with `nd` nonzero elements.

    outputs:
    * `v` - `np.array` shape `(nd-2, nd)` containing orthonormal `nd`-vectors `v[d]`.

    comments:
    * Vectors are all orthonormal and all orthogonal to the input `derivs`.
    * Roughly Gram-Schmidt orthonormalization.

    bugs:
    * Obtains numerical robustness by repeating all operations 3 times!
    """
    nd = np.sum(mask > 0)
    v = np.zeros((nd, nd))
    for dd in range(2):
        v[dd] = (derivs[:, :, dd])[mask > 0]
    # indx insanity
    indx = (np.argsort(v[0] * v[0] + v[1] * v[1]))[::-1]
    for dd in range(2, nd):
        v[dd, indx[dd-2]] = 1.
    for rep in range(3): # MAGIC 3
        for dd in range(nd):
            for ddd in range(dd):
                v[dd] -= v[ddd] * np.dot(v[dd], v[ddd])
            v[dd] /= np.sqrt(np.dot(v[dd], v[dd]))
    # for dd in range(nd):
    #     for ddd in range(dd):
    #         print dd, ddd, np.dot(v[dd], v[ddd])
    return v[2:]

def get_objective_function(ln_new_weights, derivs, mask, factors):
    """
    bugs:
    * Needs proper comment header here.
    * Repeats tons of operations every time.
    """
    new_weights = np.exp(ln_new_weights)
    objfn = 0.

    objfn1 = np.abs(np.sum(new_weights) - np.sum(mask == 3))
    print "abs total weight", objfn1
    objfn += factors[0] * objfn1

    objfn2 = np.abs(np.dot(new_weights, (derivs[:, :, 0])[mask > 0]))
    print "abs dot against derivative 0", objfn2
    objfn += factors[1] * objfn2

    objfn3 = np.abs(np.dot(new_weights, (derivs[:, :, 1])[mask > 0]))
    print "abs dot against derivative 1", objfn3
    objfn += factors[1] * objfn3

    old_weights = np.zeros_like(mask)
    old_weights[mask == 3] = 1.
    old_weights = old_weights[mask > 0]
    objfn4 = np.sum((new_weights - old_weights) ** 2)
    print "squared difference from old weights:", objfn4
    objfn += factors[3] * objfn4

    return objfn

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
    derivs = get_centroid_derivatives(bkg_sub_flux, mask)
    # make first guess at ln_weights
    ln_weights = np.zeros_like(bkg_sub_flux[0]) - 100.
    ln_weights[mask > 0] = -10.
    ln_weights[mask == 3] = 0.
    ln_weights = ln_weights[mask > 0]
    factors = [1e8, 1e8, 1e4, 1e0]
    ln_weights = op.fmin(get_objective_function, ln_weights, (derivs, mask, factors), maxfun=np.Inf, maxiter=np.Inf)
    assert False
    factors = [1e0, 1e0, 1e0, 1e0]
    ln_weights = op.fmin(get_objective_function, ln_weights, (derivs, mask, factors), ftol=1e-10, xtol=1e-10)
    new_weights = np.zeros_like(bkg_sub_flux[0])
    new_weights[mask > 0] = np.exp(ln_weights)
    print new_weights
