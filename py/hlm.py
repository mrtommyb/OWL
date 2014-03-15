#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the HLM project.
Copyright 2014 Dan Foreman-Mackey and David W. Hogg.

Bugs:
-----
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

def get_target_pixel_file(kicid, quarter):
    kic = client.star(kicid)
    tpfs = kic.get_target_pixel_files(short_cadence=False)
    tpfs = filter(lambda t: t.sci_data_quarter == quarter, tpfs)
    if not len(tpfs):
        raise ValueError("No dataset for that quarter")
    return tpfs[0]

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
