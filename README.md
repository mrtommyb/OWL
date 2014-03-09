HLM
===

A new photometry scheme for Kepler and K2.

Authors
-------
* Dan Foreman-Mackey (NYU)
* David W. Hogg (NYU)

License
-------
Copyright 2013 the authors.
Licensed under the MIT License; see the file `LICENSE` for the full text.

Notes
-----
* We start by doing the 3x3 quadratic-fit trick to get a sub-pixel centroid at every epoch.
* We do the quadratic fit within a set of 9 leave-one-out trials and median to be robust to data issues.
* We regress centroid shifts against pixel values to get two positional derivatives for every pixel.
