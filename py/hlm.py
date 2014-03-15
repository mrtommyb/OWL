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

def get_intensity_means_and_covariances(intensities, kplr_mask):
    """
    inputs:
    * `intensities` - what `kplr` calls `FLUX` from the `target_pixel_file`
    * `mask` - what `kplr` calls `hdu[2].data` from the same

    outputs:
    * `means` - one-d array of means for `kplr_mask > 0` pixels
    * `covars` - two-d array of covariances for same

    bugs:
    * Uses for loops!
    * Needs more information in this comment header.
    """
    pixel_mask = get_pixel_mask(intensities, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask, kplr_mask)
    means = np.mean(intensities[epoch_mask > 0, :, :], axis=0)
    nt, ny, nx = intensities.shape
    covars = np.zeros((ny, nx, ny, nx))
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nx):
                for ll in range(ny):
                    if (kplr_mask[jj, ii] > 0) and (kplr_mask[ll, kk] > 0) and (covars[jj, ii, ll, kk] == 0):
                        mask = pixel_mask[:, jj, ii] * pixel_mask[:, ll, kk]
                        data = (intensities[:, jj, ii] - means[jj, ii]) * (intensities[:, ll, kk] - means[ll, kk])
                        cc = np.mean(data[(mask > 0)])
                        covars[jj, ii, ll, kk] = cc
                        covars[ll, kk, jj, ii] = cc
    means = means[(kplr_mask > 0)]
    covars = covars[(kplr_mask > 0)]
    covars = covars[:, (kplr_mask > 0)]
    return means, covars

def get_objective_function(weights, means, covars):
    """
    bugs:
    * Needs more information in this comment header.
    """
    wm = np.dot(weights, means)
    return np.dot(weights, np.dot(covars, weights)) / (wm * wm)

if __name__ == "__main__":
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
    # raw_cnts = table["RAW_CNTS"]
    intensities = table["FLUX"]
    means, covars = get_intensity_means_and_covariances(intensities, kplr_mask)
    eig = np.linalg.eig(covars)
    eigval = eig[0]
    eigvec = eig[1]
    II = (np.argsort(eigval))[::-1]
    print II
    print eigval[II]
    print eigvec[II[0]]
    sap_weights = np.zeros(kplr_mask.shape)
    sap_weights[kplr_mask == 3] = 1
    sap_weights = sap_weights[kplr_mask > 0]
    start_weights = np.ones(means.shape)
    hlm_weights = op.fmin(get_objective_function, start_weights, args = (means, covars))
    print "SAP", get_objective_function(sap_weights, means, covars)
    print "start", get_objective_function(start_weights, means, covars)
    print "HLM", get_objective_function(hlm_weights, means, covars)
    foo = np.zeros_like(intensities[0])
    foo[kplr_mask > 0] = hlm_weights
    hlm_weights = foo
    print hlm_weights
    sap_weights = np.zeros_like(intensities[0])
    sap_weights[kplr_mask == 3] = 1.
    print sap_weights
