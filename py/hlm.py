#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the HLM project.
# Copyright 2014 Dan Foreman-Mackey and David W. Hogg.

import kplr

client = kplr.API()

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
    fig = tpf.plot()
    fig.savefig(prefix + ".png")
    with tpf.open() as hdu:
        table = hdu[1].data
        mask = hdu[2].data
    time_in_kbjd = table["TIME"]
    raw_cnts = table["RAW_CNTS"]
    bkg_sub_flux = table["FLUX"]
    print(raw_cnts[0].shape)
