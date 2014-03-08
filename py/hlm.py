#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the HLM project.
# Copyright 2014 Dan Foreman-Mackey and David W. Hogg.

import numpy as np
import kplr

client = kplr.API()

def get_target_pixel_file(kicid, quarter):
    kic = client.star(kicid)
    tpfs = kic.get_target_pixel_files(short_cadence=False)
    tpfs = filter(lambda t: t.sci_data_quarter == quarter, tpfs)
    if not len(tpfs):
        raise ValueError("No dataset for that quarter")
    return tpfs[0]

def get_max_pixel(cnts):
    """
    input:
    * `cnts` - `np.array` shape `(nt, ny, nx)` 

    output:
    * `xc, yc` - integer max pixel location

    comments:
    * Asserts that the max pixel is not on an edge; might be dumb.
    """
    nt, ny, nx = cnts.shape
    max_indx = np.argmax(np.mean(cnts, axis=0))
    xc, yc = max_indx % nx, max_indx / nx
    assert (xc > 0)
    assert (yc > 0)
    assert ((xc + 1) < nx)
    assert ((yc + 1) < ny)
    return xc, yc

def get_one_centroid(one_cnts, xc, yc):
    """
    input:
    * `one_cnts` - `np.array` shape `(ny, nx)`
    * `xc, yc` - max or "central" pixel.

    output:
    * `xc, yc` - floating-point centroid based on quadratic fit

    bugs:
    * Currently does NOTHING.
    """
    return xc, yc

def get_all_centroids(cnts):
    """
    input:
    * `cnts`

    output:
    * `centroids` - some crazy object of centroids

    bugs:
    * Is `map()` dumb?
    * Should I be using a lambda function or something smarter?
    """
    xc, yc = get_max_pixel(cnts)
    def goc(c):
        return get_one_centroid(c, xc, yc)
    return np.array(map(goc, cnts))

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
    raw_cnts = table["RAW_CNTS"]
    bkg_sub_flux = table["FLUX"]
    xc, yc = get_max_pixel(raw_cnts)
    centroids = get_all_centroids(raw_cnts)
    print centroids.shape
