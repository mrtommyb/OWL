#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the HLM project.
Copyright 2014 Dan Foreman-Mackey and David W. Hogg.

Bugs:
-----
* No testing framework; testing would rock this.
* Things called "flux" should be called "intensities".
* Global variable with `API()`.
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

def get_target_pixel_file(kicid, quarter):
    """
    bugs:
    * needs comment header.
    """
    kic = client.star(kicid)
    tpfs = kic.get_target_pixel_files(short_cadence=False)
    tpfs = filter(lambda t: t.sci_data_quarter == quarter, tpfs)
    if not len(tpfs):
        raise ValueError("No dataset for that quarter")
    return tpfs[0]

def evaluate_circular_two_d_gaussian(dx, dy, sigma2):
    return (1. / (2. * np.pi * sigma2)) * np.exp(-0.5 * (dx * dx + dy * dy) / sigma2)

def get_fake_data(nt, ny, nx):
    """
    bugs:
    * Needs comment header.
    * Many magic numbers.
    """
    fake_sky_noise = 1. * np.random.normal(size = (nt, ny, nx))
    xc, yc = nx - 2.65, ny - 2.15 # MAGIC NUMBERS
    xc = nx - 2.65 + 5e-5 * np.arange(nt) # MAGIC
    yc = ny - 2.15 - 2e-5 * np.arange(nt) # MAGIC
    xg, yg = np.meshgrid(range(nx), range(ny))
    flux = 10000. # MAGIC
    fake_mean = flux * evaluate_circular_two_d_gaussian(xg[None, :, :] - xc[:, None, None], yg[None, :, :] - yc[:, None, None], 1.) # MAGIC NUMBER
    fake_obj_noise = 0.1 * np.sqrt(fake_mean / flux) * np.random.normal(size = (nt, ny, nx)) # MAGIC FORMULA
    fake_mask = np.ones((ny, nx))
    fake_mask[0, 0] = 0
    mean_fake_mean = np.mean(fake_mean, axis=0)
    fake_mask[mean_fake_mean > np.percentile(mean_fake_mean, 87.5)] = 3 # MORE MAGIC
    return fake_mean + fake_sky_noise + fake_obj_noise, fake_mask

def get_pixel_mask(intensities, kplr_mask):
    """
    bugs:
    * needs comment header.
    """
    pixel_mask = np.zeros(intensities.shape)
    pixel_mask[np.isfinite(intensities)] = 1 # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0 # unless masked by kplr
    return pixel_mask

def get_epoch_mask(pixel_mask, kplr_mask):
    """
    bugs:
    * needs comment header.
    """
    foo = np.sum((pixel_mask > 0), axis=1)
    foo = np.sum(foo, axis=1)
    epoch_mask = np.zeros_like(foo)
    bar = np.sum(kplr_mask > 0)
    epoch_mask[(foo == bar)] = 1
    return epoch_mask

def get_means_and_covariances(intensities, kplr_mask, clip_mask=None):
    """
    inputs:
    * `intensities` - what `kplr` calls `FLUX` from the `target_pixel_file`
    * `kplr_mask` - what `kplr` calls `hdu[2].data` from the same
    * `clip_mask` [optional] - read the source, Luke

    outputs:
    * `means` - one-d array of means for `kplr_mask > 0` pixels
    * `covars` - two-d array of covariances for same

    bugs:
    * Only deals with unit and zero weights, nothing else.
    * Uses for loops!
    * Needs more information in this comment header.
    """
    pixel_mask = get_pixel_mask(intensities, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask, kplr_mask)
    if clip_mask is not None:
        epoch_mask *= clip_mask
    means = np.mean(intensities[epoch_mask > 0, :, :], axis=0)
    nt, ny, nx = intensities.shape
    covars = np.zeros((ny, nx, ny, nx))
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nx):
                for ll in range(ny):
                    if (kplr_mask[jj, ii] > 0) and (kplr_mask[ll, kk] > 0) and (covars[jj, ii, ll, kk] == 0):
                        mask = epoch_mask * pixel_mask[:, jj, ii] * pixel_mask[:, ll, kk]
                        data = (intensities[:, jj, ii] - means[jj, ii]) * (intensities[:, ll, kk] - means[ll, kk])
                        cc = np.mean(data[(mask > 0)])
                        covars[jj, ii, ll, kk] = cc
                        covars[ll, kk, jj, ii] = cc
    means = means[(kplr_mask > 0)]
    covars = covars[(kplr_mask > 0)]
    covars = covars[:, (kplr_mask > 0)]
    print "get_means_and_covariancess():", means
    print "get_means_and_covariancess():", np.trace(covars), np.linalg.det(covars), np.sum(epoch_mask)
    return means, covars

def get_objective_function(weights, means, covars):
    """
    bugs:
    * Needs more information in this comment header.
    """
    wm = np.dot(weights, means)
    return 1.e6 * np.dot(weights, np.dot(covars, weights)) / (wm * wm)

def get_chi_squareds(intensities, means, covars, kplr_mask):
    """
    bugs:
    * Needs more information in this comment header.
    """
    resids = intensities[:, kplr_mask > 0] - means[None, :]
    invcov = np.linalg.inv(covars)
    return np.sum(resids * np.dot(resids, invcov), axis=1)

def get_sigma_clip_mask(intensities, means, covars, kplr_mask, nsigma=4.0):
    """
    bugs:
    * Needs more information in this comment header.
    """
    ndof = np.sum(kplr_mask > 0)
    chi_squareds = get_chi_squareds(intensities, means, covars, kplr_mask)
    mask = np.zeros_like(chi_squareds)
    mask[chi_squareds < ndof + nsigma * np.sqrt(2. * ndof)] = 1.
    return mask

if __name__ == "__main__":
    Fake = False
    if Fake:
        intensities, kplr_mask = get_fake_data(4700, 4, 5)
    else:
        kicid = 3335426
        prefix = "kic_%08d" % (kicid, )
        tpf = get_target_pixel_file(kicid, 5)
        if False:
            fig = tpf.plot()
            fig.savefig(prefix + ".png")
        with tpf.open() as hdu:
            table = hdu[1].data
            kplr_mask = hdu[2].data
        time_in_kbjd = table["TIME"]
        raw_cnts = table["RAW_CNTS"]
        intensities = table["FLUX"]
    means, covars = get_means_and_covariances(intensities, kplr_mask)
    for i in range(5):
        clip_mask = get_sigma_clip_mask(intensities, means, covars, kplr_mask)
        means, covars = get_means_and_covariances(intensities, kplr_mask, clip_mask)
    eig = np.linalg.eig(covars)
    eigval = eig[0]
    eigvec = eig[1]
    II = (np.argsort(eigval))[::-1]
    print II
    print eigval[II]
    foo = np.zeros_like(intensities[0])
    foo[kplr_mask > 0] = eigvec[II[0]]
    print foo
    sap_weights = np.zeros(kplr_mask.shape)
    sap_weights[kplr_mask == 3] = 1
    sap_weights = sap_weights[kplr_mask > 0]
    start_weights = np.ones(means.shape)
    hlm_weights = op.fmin(get_objective_function, start_weights, args=(means, covars), maxfun=np.Inf, maxiter=np.Inf, xtol=1e-9, ftol=1e-9)
    print "SAP", get_objective_function(sap_weights, means, covars)
    print "start", get_objective_function(start_weights, means, covars)
    print "HLM", get_objective_function(hlm_weights, means, covars)
    sap_weights = np.zeros_like(intensities[0])
    sap_weights[kplr_mask == 3] = 1.
    print "SAP weights:", sap_weights
    foo = np.zeros_like(intensities[0])
    foo[kplr_mask > 0] = means
    print "means:", foo
    foo = np.zeros_like(intensities[0])
    foo[kplr_mask > 0] = np.diag(covars)
    mean_img = foo
    print "diag(covars):", foo
    foo = np.zeros_like(intensities[0])
    foo[kplr_mask > 0] = hlm_weights
    weight_img = foo
    print "HLM weights:", foo
    print "frac pixel contribs:", mean_img * weight_img / np.sum(mean_img * weight_img)
