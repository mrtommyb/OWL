#!/usr/bin/env python
# -*- coding: utf-8 -*-
## use OWL to make K2 light curves

from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import owl

import glob
from astropy.io import fits as pyfits

if __name__ == '__main__':
    datadirlc = '/Users/tom/Projects/K2_science/Jan2014_postSafeMode/LC/'
    datadirsc = '/Users/tom/Projects/K2_science/Jan2014_postSafeMode/SC/'

    testfile = datadirlc + 'kplr060017809-2014044044430_lpd-targ.fits'

    f = pyfits.open(testfile)

    outs = owl.photometer_and_plot_k2(f)


