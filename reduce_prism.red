#
# This is a HiPERCAM "reduce file" which defines the operation of the
# reduce script. It was written by the HiPERCAM pipeline command
# 'genred'.  It consists of a series of sections each of which
# contains a number of parameters. The file is self-documenting on the
# meaning of these parameters. The idea is that these are to large
# extent unchanging and it would be annoying to be prompted every time
# for them, but it also acts as a record of how reduction was carried
# out and is fed into the log file produce by 'reduce'.

# File written on 03 Feb 2023 09:55:32 (UTC)
#
# HiPERCAM pipeline version: 20210602
#
# Start with some general items that tend not to change
# much. 'version' is the version of the reduce file format which
# changes more slowly than the software does. It must match the same
# parameter in 'reduce' for reduction to proceed. This is
# automatically the case at the time of creation, but old versions of
# the reduce file may become incompatible with later versions of
# reduce. Either they will require updating to be used, or the
# software version can be rolled back to give a compatible version of
# reduce using 'git'. The script 'rupdate', which attempts automatic
# update, may be worth trying if you need to update. It attempts to
# make the minimum changes needed to an old reduce file to run with
# later version dates.

[general]
version = 20230222 # must be compatible with the version in reduce

instrument = other # instrument
scale = 0.39 # pixel scale, arcsec/pixel

ldevice = 1/xs # PGPLOT plot device for light curve plots
lwidth = 0.0 # light curve plot width, inches, 0 to let program choose
lheight = 0.0 # light curve plot height, inches

idevice = 2/xs # PGPLOT plot device for image plots [if implot True]
iwidth = 0.0 # image curve plot width, inches, 0 to let program choose
iheight = 0.0 # image curve plot height, inches

toffset = 0 # integer offset to subtract from the MJD

# skip points with bad times in plots. HiPERCAM has a problem in not
# correctly indicating bad times so one does not usually want to
# skip "bad time" points, whereas one should for ULTRACAM and ULTRASPEC.
skipbadt = yes

# series of count levels at which warnings will be triggered for (a)
# non linearity and (b) saturation. Each line starts 'warn =', and is
# then followed by the CCD label, the non-linearity level and the
# saturation level

warn = 1 45000.0 60000.0


# The aperture reposition and extraction stages can be run in separate
# CPUs in parallel for each CCD potentially offering speed
# advantages. 'ncpu' is the number of CPUs to use for this. The
# maximum useful and best number to use is the number of CCDs in the
# instrument, e.g. 5 for HiPERCAM. You probably also want to leave at
# least one CPU to do other stuff, but if you have more than 2 CPUs,
# this parameter may help speed things. If you use this option (ncpu >
# 1), then there is also an advantage in terms of reducing
# parallelisation overheads in reading frames a few at a time before
# processing. This is controlled using 'ngroup'. i.e. with ngroup=10,
# 10 full frames are read before being processed. This parameter is
# ignored if ncpu==1

# I have had mixed results with this, which could depend a lot on the
# installation. On my home desktop is works better with ncpu=1 because
# it already seems to parallelise and it only gets worse if I split things
# up. But no harm trying.

ncpu = 1
ngroup = 1

# The next section '[apertures]' defines how the apertures are
# re-positioned from frame to frame. Apertures are re-positioned
# through a combination of a search near a start location followed by
# a 2D profile fit. Several parameters below are associated with this
# process and setting these right can be the key to a successful
# reduction.  If there are reference apertures, they are located first
# to give a mean shift. This is used to avoid the initial search for
# any non-reference apertures which has the advantage of reducing the
# chance of problems. The search is carried out by first extracting a
# sub-window centred on the last good position of a target. This is
# then smoothed by a gaussian (width 'search_smooth_fwhm'), and the
# closest peak to the last valid position higher than
# 'fit_height_min_ref' above background (median of the square box) is
# taken as the initial position for later profile fits. The smoothing
# serves to make the process more robust against cosmic rays. The
# width of the search box ('search_half_width') depends on how good
# the telescope guiding is. It should be large enough to cope with the
# largest likely shift in position between any two consecutive
# frames. Well-chosen reference targets, which should be isolated and
# bright, can help this process a great deal. The threshold is applied
# to the *smoothed* image. This means that it can be significantly
# lower than simply the raw peak height. e.g. a target might have a
# typical peak height around 100, in seeing of 4 pixels FWHM. If you
# smooth by 10 pixels, the peak height will drop to
# 100*4**2/(4**2+10**2) = 14 counts. It will be much more stable as a
# result, but you should then probably choose a threshold of 7 when
# you might have thought 50 was appropriate. The smoothing itself can
# be carried out by direct convolution or by an FFT-based method. The
# end-result is the same either way but for large values of
# 'search_smooth_fwhm', i.e. >> 1, FFTs may offer an advantage
# speed-wise. But the only way to tell is by explicity running with
# 'search_smooth_fft' switched from 'no' to 'yes'.

# The boxes for the fits ('fit_half_width') need to be large enough to
# include the target and a bit of sky to ensure that the FWHM is
# accurately measured, remembering that seeing can flare of course. If
# your target was defocussed, a gaussian or Moffat function will be a
# poor fit and you may be better keeping the FWHM fixed at a large
# value comparable to the widths of your defoccused images (and use
# the gaussian option in such cases). If the apertures are chosen to
# be fixed, there will be no search or fit carried out in which case
# you must choose 'fixed' as well when it comes the extraction since
# otherwise it needs a FWHM. 'fixed' is a last resort and you will
# very likely need to use large aperture radii in the extraction
# section.

# An obscure parameter is 'fit_ndiv'. If this is made > 0, the fit
# routine attempts to allow for pixellation by evaluating the profile
# at multiple points within each pixel of the fit. First it will
# evaluate the profile for every unbinned pixel within a binned pixel
# if the pixels are binned; second, it will evaluate the profile over
# an ndiv by ndiv square grid within each unbinned pixel. Thus ndiv=1
# means one evaluation at the centre of each unbinned pixel,
# etc. Obviously this will slow things, but it could help if your
# images are under-sampled. I would always start with fit_ndiv=0, and
# only raise it if the measured FWHM seem to be close to or below two
# binned pixels.

# If you use reference targets (you should if possible), the initial
# positions for the non-reference targets should be good. You can then
# guard further against problems using the parameter 'fit_max_shift'
# to reject positions for the non-reference targets that shift too far
# from the initial guess. 'fit_alpha' is another parameter that
# applies only in this case. If reference apertures are being used,
# the expected locations of non-reference apertures can be predicted
# with some confidence. In this case when the non-reference aperture's
# position is measured, its position will be adjusted by 'fit_alpha'
# times the measured change in its position. Its value is bounded by 0
# < fit_alpha <= 1. "1" just means use the full measured change from
# the current frame to update the position. Anything < 1 builds in a
# bit of past history. The hope is that this could make the aperture
# positioning, especially for faint targets, more robust to cosmic
# rays and other issues.  Of course it will correlate the positions
# from frame to frame. fit_alpha = 0.1 for instance will lead to a
# correlation length ~ 10 frames.

# If you use > 1 reference targets, then the parameter 'fit_diff'
# comes into play.  Multiple reference targets should move together
# and give very consistent shifts. If they don't, then a problem may
# have occurred, e.g. one or more have been affected by a meteor trail
# for instance. The maximum acceptable differential shift is defined
# by 'fit_diff'. If exceeded, then the entire extraction will be
# aborted and positions held fixed.

# To get and idea of the right values of some of these parameters, in
# particular the 'search_half_width', the height thresholds,
# 'fit_max_shift' and 'fit_diff', the easiest approach is probably to
# run a reduction with loose values and see how it goes.

[apertures]
aperfile = aperture.ape # file of software apertures for each CCD
location = variable # aperture locations: 'fixed' or 'variable'

search_half_width = 15.0 # for initial search for objects around previous position, unbinned pixels
search_smooth_fwhm = 3.0 # smoothing FWHM, binned pixels
search_smooth_fft = no # use FFTs for smoothing, 'yes' or 'no'.

fit_method = moffat # gaussian or moffat
fit_beta = 3.0 # Moffat exponent
fit_beta_max = 20.0 # max Moffat expt
fit_fwhm = 8.0 # FWHM, unbinned pixels
fit_fwhm_min = 1.5 # Minimum FWHM, unbinned pixels [MUST be < fit_fwhm!]
fit_fwhm_max = 42.9 # Maximum FWHM, unbinned pixels
fit_ndiv = 0 # sub-pixellation factor
fit_fwhm_fixed = no # Might want to set = 'yes' for defocussed images
fit_half_width = 35.0 # for fit, unbinned pixels
fit_thresh = 3.60 # RMS rejection threshold for fits
fit_height_min_ref = 40.0 # minimum height to accept a fit, reference aperture
fit_height_min_nrf = 10.0 # minimum height to accept a fit, non-reference aperture
fit_max_shift = 20.0 # max. non-ref. shift, unbinned pixels.
fit_alpha = 0.10 # Fraction of non-reference aperture shift to apply
fit_diff = 5.0 # Maximum differential shift of multiple reference apertures, unbinned

# The next lines define how the apertures will be re-sized and how the
# flux will be extracted from the aperture. There is one line per CCD
# with format:

# <CCD label> = <resize> <extract method> [scale min max] [scale min max]
#               [scale min max]

# where: <CCD label> is the CCD label; <resize> is either 'variable'
# or 'fixed' and sets how the aperture size will be determined -- if
# variable it will be scaled relative to the FWHM, so profile fitting
# will be attempted; <extract method> is either 'normal' or 'optimal'
# to say how the flux will be extracted -- 'normal' means a straight
# sum of sky subtracted flux over the aperture, 'optimal' use Tim
# Naylor's profile weighting, and requires profile fits to
# work. Finally there follow a series of numbers in three triplets,
# each of which is a scale factor relative to the FWHM for the
# aperture radius if the 'variable' option was chosen, then a minimum
# and a maximum aperture radius in unbinned pixels.  The three triples
# correspond to the target aperture radius, the inner sky radius and
# finally the outer sky radius. The mininum and maximum also apply if
# you choose 'fixed' apertures and can be used to override whatever
# value comes from the aperture file. A common approach is set them
# equal to each other to give a fixed value, especially for the sky
# where one does not necessarily want the radii to vary.  For PSF
# photometry, all these settings have no effect, but this section can
# still be used to determine which CCDs have fluxes extracted.
# DEFAULT = "1 = variable normal 1.80 3.0 25.7 2.5 25.7 25.7 3.0 38.6 38.6"
# CHIMERA = "1 = variable optimal 1.5 2.0 6.0 2.5 7.0 7.0 3.0 10.0 10.0"

[extraction]
1 = variable optimal 1.5 2.0 14.0 2.5 30.0 30.0 3.0 40.0 40.0


# The next lines are specific to the PSF photometry option. 'gfac' is
# used to label the sources according to groups, such that stars
# closer than 'gfac' times the FWHM are labelled in the same
# group. Each group has a PSF model fit independently. The reason
# behind the construction of groups is to reduce the dimensionality of
# the fitting procedure. Usually you want closely seperated stars to
# be fit simultaneously, but too large a value will mean fitting a
# model with many free parameters, which can fail to converge. The
# size of the box over which data is collected for fitting is set by
# 'fit_half_width'. Finally, 'positions' determines whether the star's
# positions should be considered variable in the PSF fitting. If this
# is set to fixed, the positions are held at the locations found in
# the aperture repositioning step, otherwise the positions are refined
# during PSF fitting. This step can fail for PSF photometry of faint
# sources.

[psf_photom]
gfac = 3.0  # multiple of the FWHM to use in grouping objects
fit_half_width = 15.0  # size of window used to collect the data to do the fitting
positions = variable   # 'fixed' or 'variable'

# Next lines determine how the sky background level is
# calculated. Note you can only set error = variance if method =
# 'clipped'. 'median' should usually be avoided as it can cause
# noticable steps in light curves. It's here as a comparator.

[sky]
method = clipped # 'clipped' | 'median'
error  = variance # 'variance' | 'photon': first uses actual variance of sky
thresh = 3.0 # threshold in terms of RMS for 'clipped'

# Calibration frames and constants

# If you specify "!" for the readout, an attempt to estimate it from
# +/- 1 sigma percentiles will be made. This could help if you have no
# bias (and hence variance calculation from the count level will be
# wrong)

# The fringe parameters are the most complex. To de-fringe you need
# first a fringe map, e.g. as returned by makefringe. Then you need a
# file of FringePairs as returned by 'setfringe'. These are pairs of
# points placed at peaks and troughs of fringes. They will be used to
# measure a series of differences in each frame of interest and the
# fringe map from which a series of ratios is formed. The differences
# are measured from a group of pixels extending +/-nhlaf around the
# central pixel of each point of a FringePair. Thus nhalf=2 would give
# a 5x5 array (unbinned pixels). Finally to reduce the effect of bad
# values the individual ratios can be pruned if they lie outside a
# range rmin to rmax prior to their overall median being calculated.
# rmin should be small, but probably slightly less than zero to allow
# for a bit of statistical fluctuation, depending on your setup. One
# would normally expect rmax < 1 if the fringe map came from frames with
# a long exposure compared to the data.

[calibration]
crop = yes # Crop calibrations to match the data
bias =  # Bias frame, blank to ignore
flat =  # Flat field frame, blank to ignore
dark =  # Dark frame, blank to ignore
fmap =  # Fringe map frame, blank to ignore
fpair =  # File of fringe pairs, ignored if fmap blank

nhalf = 3 # Half-size of region used for fringe ratios, binned pix, ignored if fmap blank=""
rmin = -1.0 # Mininum acceptable individual fmap ratio, ignored if fmap blank
rmax = 2.0 # Maximum acceptable individual fmap ratio, ignored if fmap blank
readout = 4.0 # RMS ADU. Float or string name of a file or "!" to estimate on the fly
gain = 1.1 # Gain, electrons/ADU. Float or string name of a file

# The light curve plot which consists of light curves, X & Y
# poistions, the transmission and seeing. All but the light curves can
# be switched off by commenting them out (in full). First a couple of
# general parameters.

[lcplot]
xrange  = 0 # maximum range in X to plot (minutes), <= 0 for everything
extend_x = 10.0 # amount by which to extend xrange, minutes.

# The light curve panel (must be present). Mostly obvious, then a
# series of lines, each starting 'plot' which specify one light curve
# to be plotted giving CCD, target, comparison ('!' if you don't want
# a comparison), an additive offset, a multiplicative scaling factor
# and then a colour for the data and a colour for the error bar There
# will always be a light curve plot, whereas later elements are
# optional, therefore the light curve panel is defined to have unit
# height and all others are scaled relative to this.

[light]
linear  = no # linear vertical scale (else magnitudes): 'yes' or 'no'
y_fixed = no # keep a fixed vertical range or not: 'yes' or 'no'
y1 = 0 # initial lower y value
y2 = 0 # initial upper y value. y1=y2 for auto scaling
extend_y = 0.1 # fraction of plot height to extend when rescaling

# line or lines defining the targets to plot
plot = 1 1 2 0 1 green !   # ccd, targ, comp, off, fac, dcol, ecol

# The X,Y position panel. Comment *every* line
# if you don't want it.

[position]
height = 0.5 # height relative to light curve plot
x_fixed = no # keep X-position vertical range fixed
x_min = -5.0 # lower limit for X-position
x_max = 5.0 # upper limit for X-position
y_fixed = no # keep Y-position vertical range fixed
y_min = -5.0 # lower limit for Y-position
y_max = 5.0 # upper limit for Y-position
extend_y = 0.2 # Vertical extension fraction if limits exceeded

# line or lines defining the targets to plot
plot = 1 2 green ! # ccd, targ, dcol, ecol

# The transmission panel. If you comment it out,
# comment *every* line.
# Simply plots the flux in whatever apertures are chosen,
# scaling them by their maximum (hence one can sometimes
# find that what you thought was 100% transmission was
# actually only 50% revealed as the cloud clears).

[transmission]
height = 0.5 # height relative to the light curve plot
ymax = 110.0 # Maximum transmission to plot (>= 100 to slow replotting)

# line or lines defining the targets to plot
plot = 1 2 green ! # ccd, targ, dcol, ecol

# The seeing panel. Can be commented out if you don't want one but make
# sure to comment it out completely, section name and all
# parameters. You can have multiple plot lines. Don't choose linked
# targets as their FWHMs are not measured.

[seeing]
height = 0.5 # height relative to the light curve plot
ymax = 2.999 # Initial maximum seeing
y_fixed = no # fix the seeing scale (or not)
extend_y = 0.2 # Y extension fraction if out of range and not fixed

# line or lines defining the targets to plot
plot = 1 2 green ! # ccd, targ, dcol, ecol


# This option attempts to correct for a badly-positioned focal plane mask
# in drift mode which combined with a high background can lead to steps in
# illumination in the Y direction. This tries to subtract the median in the
# X-direction of each window. 'dthresh' is a threshold used to reject X
# pixels prior to taking the median. The purpose is to prevent the stars
# from distorting the median. Take care with this option which is experimental.

[focal_mask]
demask = no
dthresh = 3.0

# Monitor section. This section allows you to monitor particular
# targets for problems. If they occur, then messages will be printed
# to the terminal during reduce. The messages are determined by the
# bitmask flag set during the extraction of each
# target. Possibilities:

#  NO_FWHM           : no FWHM measured
#  NO_SKY            : no sky pixels at all
#  SKY_AT_EDGE       : sky aperture off edge of window
#  TARGET_AT_EDGE    : target aperture off edge of window
#  TARGET_SATURATED  : at least one pixel in target above saturation level
#  TARGET_NONLINEAR  : at least one pixel in target above nonlinear level
#  NO_EXTRACTION     : no extraction possible
#  NO_DATA           : no valid pixels in aperture

# For a target you want to monitor, type its label, '=', then the
# bitmask patterns you want to be flagged up if they are set. This is
# designed mainly for observing, as there is less you can do once the
# data have been taken, but it still may prove useful.

[monitor]
1 = NO_EXTRACTION TARGET_SATURATED TARGET_AT_EDGE TARGET_NONLINEAR NO_SKY NO_FWHM NO_DATA SKY_AT_EDGE
2 = NO_EXTRACTION TARGET_SATURATED TARGET_AT_EDGE TARGET_NONLINEAR NO_SKY NO_FWHM NO_DATA SKY_AT_EDGE
3 = NO_EXTRACTION TARGET_SATURATED TARGET_AT_EDGE TARGET_NONLINEAR NO_SKY NO_FWHM NO_DATA SKY_AT_EDGE
4 = NO_EXTRACTION TARGET_SATURATED TARGET_AT_EDGE TARGET_NONLINEAR NO_SKY NO_FWHM NO_DATA SKY_AT_EDGE
5 = NO_EXTRACTION TARGET_SATURATED TARGET_AT_EDGE TARGET_NONLINEAR NO_SKY NO_FWHM NO_DATA SKY_AT_EDGE
6 = NO_EXTRACTION TARGET_SATURATED TARGET_AT_EDGE TARGET_NONLINEAR NO_SKY NO_FWHM NO_DATA SKY_AT_EDGE

