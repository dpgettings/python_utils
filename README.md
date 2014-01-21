python_utils
============

A repo for some potentially-useful Python code I've developed in the course of my thesis research.


### fits_utils.py

Utilities for dealing with FITS tables. Nearly all of this is probably obviated by updates to Astropy, making FITS tables interface with the Table class. But included here anyway.


### hist_functions.py

For now, just contains my rewrite of Numpy's ``histogramdd`` function, called ``histogram_nd``. It's not quite as advanced (especially in the number of options for specifying binning parameters) but it should be *much* faster and *much* more memory-efficient.
