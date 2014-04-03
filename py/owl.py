#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the OWL project.
Copyright 2014 Dan Foreman-Mackey and David W. Hogg.

## bugs:
* No testing framework; testing would rock this.
* Things called "flux" should be called "intensities".
* Global variable with `API()`.
* Barely tested, and there are copious x <-> y issues possible.
"""

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
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
    ## bugs:
    * needs comment header.
    """
    kic = client.star(kicid)
    tpfs = kic.get_target_pixel_files(short_cadence=False)
    tpfs = filter(lambda t: t.sci_data_quarter == quarter, tpfs)
    if not len(tpfs):
        raise ValueError("No dataset for that quarter")
    return tpfs[0]

def evaluate_circular_two_d_gaussian(dx, dy, sigma2):
    """
    Only used in `get_fake_data()`.
    """
    return (1. / (2. * np.pi * sigma2)) * np.exp(-0.5 * (dx * dx + dy * dy) / sigma2)

def get_fake_data(nt, ny=5, nx=7):
    """
    Make a moving, varying, PSF-changing source plus noise.

    ## bugs:
    * Many magic numbers.
    * Needs comment header.
    """
    xc, yc = 3. + 1. / 7., 2. + 4. / 9. # MAGIC NUMBERS
    psf_sigma2 = 1.1 * 1.1 # MAGIC NUMBER (in pixels * pixels)
    psf_sigma2 = psf_sigma2 + 0.01 * np.arange(nt) / nt
    flux = 100000. # MAGIC NUMBER (in ADU per image)
    flux = flux + 0.0 * flux * np.sin(np.arange(nt) / 50. )
    gain = 0.001 # MAGIC NUMBER (in electrons per ADU)
    fake_sky_noise = np.sqrt(1.) * np.random.normal(size = (nt, ny, nx)) # MAGIC NUMBER in ADU per pixel per image
    xc = xc + 0.01 * np.arange(nt) / nt
    yc = yc + np.zeros(nt)
    xg, yg = np.meshgrid(range(nx), range(ny))
    fake_mean = flux[:, None, None] * evaluate_circular_two_d_gaussian(xg[None, :, :] - xc[:, None, None],
                                                                       yg[None, :, :] - yc[:, None, None],
                                                                       psf_sigma2[:, None, None])
    fake_obj_noise = np.sqrt(gain * fake_mean) * np.random.normal(size = (nt, ny, nx))
    fake_mask = np.ones((ny, nx))
    fake_mask[0, 0] = 0
    mean_fake_mean = np.mean(fake_mean, axis=0)
    fake_bg = 0.01 * np.random.normal(size = nt) # MAGIC
    fake_mask[mean_fake_mean > np.percentile(mean_fake_mean, 87.5)] = 3 # MORE MAGIC
    return fake_mean + fake_sky_noise + fake_obj_noise + fake_bg[:, None, None], fake_mask

def get_pixel_mask(intensities, kplr_mask):
    """
    ## bugs:
    * needs comment header.
    """
    pixel_mask = np.zeros(intensities.shape)
    pixel_mask[np.isfinite(intensities)] = 1 # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0 # unless masked by kplr
    return pixel_mask

def get_epoch_mask(pixel_mask, kplr_mask):
    """
    ## bugs:
    * needs comment header.
    """
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = np.zeros_like(foo)
    bar = np.sum(kplr_mask > 0)
    epoch_mask[(foo == bar)] = 1
    return epoch_mask

def get_means_and_covariances(intensities, kplr_mask, clip_mask=None):
    """
    ## inputs:
    * `intensities` - what `kplr` calls `FLUX` from the `target_pixel_file`
    * `kplr_mask` - what `kplr` calls `hdu[2].data` from the same
    * `clip_mask` [optional] - read the source, Luke

    ## outputs:
    * `means` - one-d array of means for `kplr_mask > 0` pixels
    * `covars` - two-d array of covariances for same

    ## bugs:
    * Only deals with unit and zero weights, nothing else.
    * Uses `for` loops!
    * Needs more information in this comment header.
    """
    nt, ny, nx = intensities.shape
    pixel_mask = get_pixel_mask(intensities, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask, kplr_mask)
    if clip_mask is not None:
        epoch_mask *= clip_mask
    means = np.mean(intensities[epoch_mask > 0, :, :], axis=0)
    covars = np.zeros((ny, nx, ny, nx))
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nx):
                for ll in range(ny):
                    if (kplr_mask[jj, ii] > 0) and (kplr_mask[ll, kk] > 0) and (covars[jj, ii, ll, kk] == 0):
                        mask = epoch_mask * pixel_mask[:, jj, ii] * pixel_mask[:, ll, kk]
                        data = (intensities[:, jj, ii] - means[jj, ii]) * (intensities[:, ll, kk] - means[ll, kk])
                        cc = np.mean(data[(mask > 0)])
                        assert np.isfinite(cc)
                        covars[jj, ii, ll, kk] = cc
                        covars[ll, kk, jj, ii] = cc
    means = means[(kplr_mask > 0)]
    covars = covars[(kplr_mask > 0)]
    covars = covars[:, (kplr_mask > 0)]
    return means, covars

def get_chi_squareds(intensities, means, covars, kplr_mask):
    """
    ## bugs:
    * Needs more information in this comment header.
    """
    resids = intensities[:, kplr_mask > 0] - means[None, :]
    invcov = np.linalg.inv(covars)
    return np.sum(resids * np.dot(resids, invcov), axis=1)

def get_sigma_clip_mask(intensities, means, covars, kplr_mask, nsigma=4.0):
    """
    Sigma-clipper making use of the chi-squared distribution.

    ## bugs:
    * Not properly audited or tested.
    * Needs more information in this comment header.
    """
    ndof = np.sum(kplr_mask > 0)
    chi_squareds = get_chi_squareds(intensities, means, covars, kplr_mask)
    mask = np.zeros_like(chi_squareds)
    mask[chi_squareds < ndof + nsigma * np.sqrt(2. * ndof)] = 1.
    return mask

def get_robust_means_and_covariances(intensities, kplr_mask, clip_mask=None):
    """
    Iterative sigma-clipping version of `get_means_and_covariances()`.

    ## bugs:
    * Magic number 5 hard-coded.
    * Needs more information in this comment header.
    """
    means, covars = get_means_and_covariances(intensities, kplr_mask)
    for i in range(5): # MAGIC
        clip_mask = get_sigma_clip_mask(intensities, means, covars, kplr_mask)
        means, covars = get_means_and_covariances(intensities, kplr_mask, clip_mask)
    return means, covars

def get_owl_weights(means, covars):
    """
    needs no comment
    """
    return np.linalg.solve(covars, means)

def get_objective_function(weights, means, covars, ln=False):
    """
    ## bugs:
    * Needs more information in this comment header.
    """
    if ln:
        ws = np.exp(weights)
    else:
        ws = weights
    wm = np.dot(ws, means)
    return 1.e6 * np.dot(ws, np.dot(covars, ws)) / (wm * wm)

def get_opw_weights(means, covars, owl_weights=None):
    """
    needs comment
    """
    if owl_weights is None:
        owl_weights = get_owl_weights(means, covars)
    start_ln_weights = np.log(np.abs(owl_weights))
    ln_weights = op.fmin(get_objective_function, start_ln_weights, args=(means, covars, True))
    return np.exp(ln_weights)

def get_tsa_intensities_and_mask(intensities, kplr_mask):
    tsa_intensities = np.hstack((np.sum(intensities[:, kplr_mask == 3], axis=1)[:, None],
                                 intensities[:, kplr_mask == 1]))
    tsa_mask = np.ones_like(tsa_intensities[0])
    tsa_mask[0] = 3
    return tsa_intensities[:,:,None], tsa_mask[:, None]

def savefig(fn):
    print "writing file " + fn
    plt.savefig(fn)

def photometer_and_plot(kicid, quarter, fake=False, makeplots=True):
    """
    ## inputs:
    - `kicid` - KIC number
    - `quarter` - Kepler observing quarter (or really place in list of files)

    ## outputs:
    - [some plots]
    - `time` - times in KBJD
    - `sap_photometry` - home-built SAP equivalent photometry
    - `owl_photometry` - OWL photometry

    #3 bugs:
    - Does unnecessary reformatting galore.
    - Should split off plotting to a separate function.
    - Comment header not complete.
    """
    fsf = 2.5 # MAGIC number used to stretch plots
    if fake:
        prefix = "fake"
        title = "fake data"
        intensities, kplr_mask = get_fake_data(4700)
        nt, ny, nx = intensities.shape
        time_in_kbjd = np.arange(len(intensities)) / 24. / 2.
    else:
        prefix = "kic_%08d" % (kicid, )
        title = "KIC %08d" % (kicid, )
        tpf = get_target_pixel_file(kicid, quarter)
        with tpf.open() as hdu:
            table = hdu[1].data
            kplr_mask = hdu[2].data
        time_in_kbjd = table["TIME"]
        # raw_cnts = table["RAW_CNTS"]
        intensities = table["FLUX"]
        nt, ny, nx = intensities.shape
        if makeplots:
            fig = tpf.plot()
            fig.suptitle(title)
            savefig("%s_pixels.png" % prefix)

    # get SAP weights and photometry
    sap_weights = np.zeros(kplr_mask.shape)
    sap_weights[kplr_mask == 3] = 1
    sap_weights = sap_weights[kplr_mask > 0]
    def reformat_as_image(bar):
        foo = np.zeros_like(intensities[0])
        foo[kplr_mask > 0] = bar
        return foo
    sap_weight_img = reformat_as_image(sap_weights)
    pixel_mask = get_pixel_mask(intensities, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask, kplr_mask)
    fubar_intensities = intensities
    fubar_intensities[pixel_mask == 0] = 0.
    sap_lightcurve = np.sum(np.sum(fubar_intensities * sap_weight_img[None, :, :], axis=2), axis=1)

    # get OWL weights and photometry
    means, covars = get_robust_means_and_covariances(intensities, kplr_mask)
    owl_weights = get_owl_weights(means, covars)
    owl_weights *= np.sum(sap_weights * means) / np.sum(owl_weights * means)
    owl_weight_img = reformat_as_image(owl_weights)
    owl_lightcurve = np.sum(np.sum(fubar_intensities * owl_weight_img[None, :, :], axis=2), axis=1)

    # get OPW weights and photometry
    opw_weights = get_opw_weights(means, covars, owl_weights=owl_weights)
    opw_weights *= np.sum(sap_weights * means) / np.sum(opw_weights * means)
    opw_weight_img = reformat_as_image(opw_weights)
    opw_lightcurve = np.sum(np.sum(fubar_intensities * opw_weight_img[None, :, :], axis=2), axis=1)
    print "SAP", np.min(sap_lightcurve), np.max(sap_lightcurve)
    print "OWL", np.min(owl_lightcurve), np.max(owl_lightcurve)
    print "OPW", np.min(opw_lightcurve), np.max(opw_lightcurve)

    # fire up the TSA
    tsa_intensities, tsa_mask = get_tsa_intensities_and_mask(intensities, kplr_mask)
    tsa_means, tsa_covars = get_robust_means_and_covariances(tsa_intensities, tsa_mask)
    tsa_weights = get_opw_weights(tsa_means, tsa_covars)
    tsa_weights *= np.sum(sap_weights * means) / np.sum(tsa_weights * tsa_means)
    tsa_weight_img = np.zeros_like(intensities[0])
    tsa_weight_img[kplr_mask == 3] = tsa_weights[0]
    tsa_weight_img[kplr_mask == 1] = tsa_weights[1:]
    tsa_lightcurve = np.sum(np.sum(fubar_intensities * tsa_weight_img[None, :, :], axis=2), axis=1)
    print "TSA", np.min(tsa_lightcurve), np.max(tsa_lightcurve)

    if not makeplots:
        return time_in_kbjd, sap_lightcurve, owl_lightcurve, opw_lightcurve

    # get two eigenvectors (for plotting)
    eig = np.linalg.eig(covars)
    eigval = eig[0]
    eigvec = eig[1]
    II = (np.argsort(eigval))[::-1]
    eigvec0 = eigvec[II[0]]
    eigvec1 = eigvec[II[1]]

    # more reformatting
    mean_img = reformat_as_image(means)
    covar_diag_img = reformat_as_image(np.diag(covars))
    eigvec0_img = reformat_as_image(eigvec0)
    eigvec1_img = reformat_as_image(eigvec1)
    sap_frac_contribs_img = sap_weight_img * mean_img
    owl_frac_contribs_img = owl_weight_img * mean_img
    opw_frac_contribs_img = opw_weight_img * mean_img

    # make images plot
    plt.gray()
    plt.figure(figsize=(fsf * nx, fsf * ny)) # MAGIC
    plt.clf()
    plt.title(title)
    vmax = np.percentile(intensities[:, kplr_mask > 0], 99.)
    vmin = -1. * vmax
    for ii, sp in [(0, 331), (nt / 2, 332), (nt-1, 333)]:
        plt.subplot(sp)
        plt.imshow(intensities[ii], interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.title("exposure %d" % ii)
        plt.colorbar()
    plt.subplot(334)
    plt.imshow(mean_img, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(r"mean $\hat{\mu}$")
    plt.colorbar()
    plt.subplot(335)
    plt.imshow(np.log10(covar_diag_img), interpolation="nearest")
    plt.title(r"log diag($\hat{C})$")
    plt.colorbar()
    plt.subplot(336)
    plt.imshow(eigvec0_img, interpolation="nearest")
    plt.title(r"dominant $\hat{C}$ eigenvector")
    plt.colorbar()
    vmax = 1.2 * np.max(sap_weight_img)
    vmin = -1. * vmax
    plt.subplot(337)
    plt.imshow(sap_weight_img, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(r"SAP weights")
    plt.colorbar()
    plt.subplot(338)
    plt.imshow(owl_weight_img, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(r"OWL weights")
    plt.colorbar()
    plt.subplot(339)
    plt.imshow(owl_weight_img * mean_img, interpolation="nearest", vmin=-np.max(owl_weight_img * mean_img))
    plt.title(r"OWL mean contribs")
    plt.colorbar()
    savefig("%s_images_owl.png" % prefix)

    # make OPW plot
    plt.figure(figsize=(fsf * nx, fsf * ny / 3.)) # MAGIC
    plt.clf()
    plt.title(title)
    plt.subplot(131)
    plt.imshow(sap_weight_img, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(r"SAP weights")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(opw_weight_img, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(r"OPW weights")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(opw_weight_img * mean_img, interpolation="nearest", vmin=-np.max(opw_weight_img * mean_img))
    plt.title(r"OPW mean contribs")
    plt.colorbar()
    savefig("%s_images_opw.png" % prefix)

    # make OPW plot REPEATED CODE
    plt.figure(figsize=(fsf * nx, fsf * ny / 3.)) # MAGIC
    plt.clf()
    plt.title(title)
    plt.subplot(131)
    plt.imshow(sap_weight_img, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(r"SAP weights")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(tsa_weight_img, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(r"TSA weights")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(tsa_weight_img * mean_img, interpolation="nearest", vmin=-np.max(tsa_weight_img * mean_img))
    plt.title(r"TSA mean contribs")
    plt.colorbar()
    savefig("%s_images_tsa.png" % prefix)

    # make photometry plot
    plt.figure(figsize=(fsf * nx, 0.5 * fsf * nx))
    plt.clf()
    plt.title(title)
    clip_mask = get_sigma_clip_mask(intensities, means, covars, kplr_mask)
    I = (epoch_mask > 0) * (clip_mask > 0)
    plt.plot(time_in_kbjd[I], sap_lightcurve[I], "k-", alpha=0.5)
    plt.text(time_in_kbjd[0], sap_lightcurve[0], "SAP-", alpha=0.5, ha="right")
    plt.text(time_in_kbjd[-1], sap_lightcurve[-1], "-SAP", alpha=0.5)
    shift1 = 0.
    plt.plot(time_in_kbjd[I], shift1 + owl_lightcurve[I], "k-")
    plt.text(time_in_kbjd[0], shift1 + owl_lightcurve[0], "OWL-", ha="right")
    plt.text(time_in_kbjd[-1], shift1 + owl_lightcurve[-1], "-OWL")
    shift2 = 0.25 * (np.min(sap_lightcurve[I]) - np.median(sap_lightcurve[I]))
    plt.plot(time_in_kbjd[I], shift2 + opw_lightcurve[I], "k-")
    plt.text(time_in_kbjd[0], shift2 + opw_lightcurve[0], "OPW-", ha="right")
    plt.text(time_in_kbjd[-1], shift2 + opw_lightcurve[-1], "-OPW")
    shift3 = 0.5 * (np.min(sap_lightcurve[I]) - np.median(sap_lightcurve[I]))
    plt.plot(time_in_kbjd[I], shift3 + tsa_lightcurve[I], "k-")
    plt.text(time_in_kbjd[0], shift3 + tsa_lightcurve[0], "TSA-", ha="right")
    plt.text(time_in_kbjd[-1], shift3 + tsa_lightcurve[-1], "-TSA")
    plt.xlim(np.min(time_in_kbjd[I]) - 4., np.max(time_in_kbjd[I]) + 4.) # MAGIC
    plt.xlabel("time (KBJD in days)")
    plt.ylabel("flux (in Kepler SAP ADU)")
    savefig("%s_photometry.png" % prefix)

    return time_in_kbjd, sap_lightcurve, owl_lightcurve

if __name__ == "__main__":
    import sys
    np.random.seed(42)
    quarter = 5
    kicid = 3335426
    if len(sys.argv) > 1:
        kicid = int(sys.argv[1])
    if len(sys.argv) > 2:
        quarter = int(sys.argv[2])
    t, s, o = photometer_and_plot(kicid, quarter)

if False:
    t, s, o = photometer_and_plot(kicid, quarter, fake=True)
    kicid = 8692861
    t, s, o = photometer_and_plot(kicid, quarter)
    kicid = 3223000
    t, s, o = photometer_and_plot(kicid, quarter)
    kicid = 10295224
    t, s, o = photometer_and_plot(kicid, quarter)
