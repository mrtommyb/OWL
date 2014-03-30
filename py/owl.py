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
    ## bugs:
    * Many magic numbers.
    * Needs comment header.
    """
    xc, yc = 3. + 1. / 7., 2. + 4. / 9. # MAGIC NUMBERS
    psf_sigma2 = 1.1 * 1.1 # MAGIC NUMBER (in pixels * pixels)
    flux = 10000. # MAGIC NUMBER (in ADU per image)
    gain = 0.0 # MAGIC NUMBER (in electrons per ADU)
    fake_sky_noise = np.sqrt(1.) * np.random.normal(size = (nt, ny, nx)) # MAGIC NUMBER in ADU per pixel per image
    xc = np.zeros(nt) + xc
    yc = np.zeros(nt) + yc
    xg, yg = np.meshgrid(range(nx), range(ny))
    fake_mean = flux * evaluate_circular_two_d_gaussian(xg[None, :, :] - xc[:, None, None], yg[None, :, :] - yc[:, None, None], psf_sigma2)
    fake_obj_noise = np.sqrt(gain * fake_mean) * np.random.normal(size = (nt, ny, nx))
    fake_mask = np.ones((ny, nx))
    fake_mask[0, 0] = 0
    mean_fake_mean = np.mean(fake_mean, axis=0)
    fake_mask[mean_fake_mean > np.percentile(mean_fake_mean, 87.5)] = 3 # MORE MAGIC
    return fake_mean + fake_sky_noise + fake_obj_noise, fake_mask

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
    return means, covars

def get_objective_function(weights, means, covars):
    """
    ## bugs:
    * Needs more information in this comment header.
    """
    wm = np.dot(weights, means)
    return 1.e6 * np.dot(weights, np.dot(covars, weights)) / (wm * wm)

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
    nees no comment
    """
    return np.linalg.solve(covars, means)

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
        time_in_kbjd = np.arange(len(intensities)) / 24. / 2.
    else:
        prefix = "kic_%08d" % (kicid, )
        title = "KIC %08d" % (kicid, )
        tpf = get_target_pixel_file(kicid, quarter)
        if makeplots:
            fig = tpf.plot(figsize=(fsf * nx, fsf * ny))
            fig.title(title)
            savefig("%s_pixels.png" % prefix)
        with tpf.open() as hdu:
            table = hdu[1].data
            kplr_mask = hdu[2].data
        time_in_kbjd = table["TIME"]
        # raw_cnts = table["RAW_CNTS"]
        intensities = table["FLUX"]
    nt, ny, nx = intensities.shape
    means, covars = get_robust_means_and_covariances(intensities, kplr_mask)

    # get OWL and SAP weights
    sap_weights = np.zeros(kplr_mask.shape)
    sap_weights[kplr_mask == 3] = 1
    sap_weights = sap_weights[kplr_mask > 0]
    owl_weights = get_owl_weights(means, covars)
    owl_weights *= np.sum(sap_weights * means) / np.sum(owl_weights * means)

    # reformat back to image space
    def reformat_as_image(bar):
        foo = np.zeros_like(intensities[0])
        foo[kplr_mask > 0] = bar
        return foo
    mean_img = reformat_as_image(means)
    covar_diag_img = reformat_as_image(np.diag(covars))
    eigvec0_img = reformat_as_image(eigvec0)
    eigvec1_img = reformat_as_image(eigvec1)
    owl_weight_img = reformat_as_image(owl_weights)
    owl_frac_contribs_img = owl_weight_img * mean_img
    sap_weight_img = reformat_as_image(sap_weights)
    sap_frac_contribs_img = sap_weight_img * mean_img

    # get photometry
    pixel_mask = get_pixel_mask(intensities, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask, kplr_mask)
    fubar_intensities = intensities
    fubar_intensities[pixel_mask == 0] = 0.
    sap_lightcurve = np.sum(np.sum(fubar_intensities * sap_weight_img[None, :, :], axis=2), axis=1)
    owl_lightcurve = np.sum(np.sum(fubar_intensities * owl_weight_img[None, :, :], axis=2), axis=1)
    print "SAP", np.min(sap_lightcurve), np.max(sap_lightcurve)
    print "OWL", np.min(owl_lightcurve), np.max(owl_lightcurve)
    if not makeplots:
        return time_in_kbjd, sap_lightcurve, owl_lightcurve

    # get two eigenvectors (for plotting)
    eig = np.linalg.eig(covars)
    eigval = eig[0]
    eigvec = eig[1]
    II = (np.argsort(eigval))[::-1]
    eigvec0 = eigvec[II[0]]
    eigvec1 = eigvec[II[1]]

    # make images plot
    plt.gray()
    plt.figure(figsize=(fsf * nx, fsf * ny)) # MAGIC
    plt.clf()
    plt.title(title)
    vmax = np.percentile(intensities[:, kplr_mask > 0], 99.)
    vmin = -0.1 * vmax
    for ii, sp in [(0, 331), (nt / 2, 332), (nt-1, 333)]:
        plt.subplot(sp)
        plt.imshow(intensities[ii], interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
        plt.title("exposure %d" % ii)
        plt.colorbar()
    plt.subplot(334)
    plt.imshow(mean_img, interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
    plt.title(r"mean $\hat{\mu}$")
    plt.colorbar()
    plt.subplot(335)
    plt.imshow(np.log10(covar_diag_img), interpolation="nearest", origin="lower")
    plt.title(r"log diag($\hat{C}$")
    plt.colorbar()
    plt.subplot(336)
    plt.imshow(eigvec0_img, interpolation="nearest", origin="lower")
    plt.title(r"dominant $\hat{C}$ eigenvector")
    plt.colorbar()
    vmin = -0.12
    vmax = 1.2
    plt.subplot(337)
    plt.imshow(1. * sap_weight_img, interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
    plt.title(r"SAP weights")
    plt.colorbar()
    plt.subplot(338)
    plt.imshow(owl_weight_img, interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
    plt.title(r"OWL weights")
    plt.colorbar()
    plt.subplot(339)
    plt.imshow(owl_weight_img * mean_img, interpolation="nearest", origin="lower")
    plt.title(r"OWL mean contribs")
    plt.colorbar()
    savefig("%s_images.png" % prefix)

    # make photometry plot
    plt.figure(figsize=(fsf * nx, 0.5 * fsf * nx))
    plt.clf()
    plt.title(title)
    I = epoch_mask > 0
    plt.plot(time_in_kbjd[I], sap_lightcurve[I], "k-", alpha=0.5)
    plt.text(time_in_kbjd[-1], sap_lightcurve[-1], "SAP", alpha=0.5)
    plt.plot(time_in_kbjd[I], owl_lightcurve[I], "k-")
    plt.text(time_in_kbjd[-1], owl_lightcurve[-1], "OWL")
    plt.xlim(np.min(time_in_kbjd[I]), np.max(time_in_kbjd[I]) + 4.)
    plt.ylim(0.99 * np.min(sap_lightcurve[I]), 1.01 * np.max(sap_lightcurve[I]))
    plt.xlabel("time (KBJD in days)")
    plt.ylabel("flux (in Kepler SAP ADU)")
    savefig("%s_photometry.png" % prefix)

    return time_in_kbjd, sap_lightcurve, owl_lightcurve

if __name__ == "__main__":
    kicid = 3335426
    quarter = 5
    t, s, o = photometer_and_plot(kicid, quarter, fake=True)
    # t, s, o = photometer_and_plot(kicid, quarter)
