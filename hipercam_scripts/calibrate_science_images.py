# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 2024

@author: Joseph Guidry, Zach Vanderbosch
"""
	
#!/usr/bin/env python


"""This script ingests all calibration images (biases, darks, flats)
and uncalibrated science images. It then makes master
calibration images and reduces the science images with those
frames. This script does not perform aperture photometry.

This script is designed to be run within a folder containing
your targets raw images, e.g. "~/WD1145+017/". Outside this
folder it is assumed you have folders for each of your calibration
image types, e.g. "../bias/", "../dark:, "../dome_flat/".
"""

import argparse
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
import ccdproc
from datetime import datetime as dt
from datetime import timedelta as td
from glob import glob
import numpy as np
from os import getcwd, mkdir, system
from os.path import isfile, isdir
from pandas import read_csv,DataFrame
from scipy.stats import mode
from shutil import copyfile
import sys


#############################################################
##
##  Progress Bar Code. I got this code from Stack Overflow,
##  "Python to print out status bar and percentage"
##
#############################################################

## Provide the interation counter (count=int)
## and the action being performed (action=string)
def progress_bar(count,total,action):
    sys.stdout.write('\r')
    sys.stdout.write(action)
    sys.stdout.write("[%-20s] %d%%  %d/%d" % ('='*int((count*20/total)),\
                                              count*100/total,\
                                              count,total))
    sys.stdout.flush()
    return


def make_ilist(path,instrument):
    # Try loading FITS images using the SPE file.  
    # If that doesn't work, ask for a search string.
    if instrument=='proem' or instrument=='PROEM' or instrument=='ProEM':
    	obj_name = glob(path + '*.spe')
    elif instrument=='prism' or instrument=='PRISM' or instrument=='LMI' or instrument=='lmi':
    	obj_name = glob(path + '2*.fits')
    try:
        if (len(obj_name) == 0) or (len(obj_name) > 1):
            if sys.version_info[0] < 3:
                search_string = raw_input('\nPlease provide a search string for FITS images: ')
            else:
                search_string = input('\nPlease provide a search string for FITS images: ')
            fits_names = sorted(glob(search_string))
        else:
            search_string = obj_name[0][0:-4] + '*' + '.fits'
            fits_names = sorted(glob(search_string))
    except:
        if sys.version_info[0] < 3:
            search_string = raw_input('\nProvide a search string for FITS images: ')
        else:
            search_string = input('\nProvide a search string for FITS images: ')
        fits_names = sorted(glob(search_string))

    # print(fits_names)
    if len(fits_names) == 0:
        print('\nNo file names were returned using the provided search string.\n')
        sys.exit()

    # if this script is being run more than once, some files may
    # already exist with the "c" suffix added on.  Filter these out.
    bool_list = [x[-6] == 'c' for x in fits_names]
    c_ind = [x for x, y in enumerate(bool_list) if not y]
    fits_names = [fits_names[i].split('/')[-1].strip() for i in c_ind]

    print('\nFirst File Name Returned: %s' %fits_names[0])
    print('Last  File Name Returned: %s'   %fits_names[-1])
    print('Number Of Files Returned: %s\n' %len(fits_names))

    # create the corrected list of filenames
    cfits_names = [x[0:-5] + 'c' + x[-5:] for x in fits_names]

    # create the list of filenames for the hcm files
    hcm_names = ['hcm_files/' + x.replace('.fits','.fits2hcm.hcm') for x in cfits_names]

    # save the file names into a list
    iname = 'ilist'
    oname = 'olist'
    hcmname = 'hcm.lis'
    np.savetxt(iname,fits_names,fmt='%s',delimiter = ' ')
    np.savetxt(oname,cfits_names,fmt='%s',delimiter = ' ')
    np.savetxt(hcmname,hcm_names,fmt='%s',delimiter = ' ')

    return fits_names, cfits_names, hcm_names


def sf_impar(path, ilist):
    #########################################################
    ##
    ##  Load in the file names which need to be parsed
    ##
    #########################################################

    ## Ask the user to supply the name of a file
    ## containing a list of all images to be worked on
    print('')
    file_list = 'ilist'
    # try:
    #     filenames  = np.loadtxt(path + file_list, dtype=str)
    # except IOError:
    #     try:
    #         print('Could not find a working list using default name "%s"' %file_list)
    #         file_list = input("Please enter the working list name to search for: ")
    #         filenames  = np.loadtxt(path + file_list, dtype=str)
    #     except IOError:
    #         print('Could not find a working list with name "%s"' %file_list)
    #         print('\nProgram TERMINATED\n')
    #         sys.exit(1)
    filenames = ilist

    ## Save the original file names for comparison later
    og_filenames = filenames
    num_files    = len(filenames)


    ###############################################################
    ##
    ##  First, let's change the exposure time value in
    ##  the actual headers of the FITS files (EXPTIME)
    ##  and add in the filter information with a new
    ##  keyword, FILTER.
    ##
    ###############################################################

    ## Inform the user what values will be changed and ask whether to proceed
    print('')
    print('#####################################################')
    print('##          This section will add and edit         ##')
    print('##         the following FITS header values:       ##')
    print('## ----------------------------------------------- ##')
    print('## EXPTIME  - The image exposure time              ##')
    print('## FILTER   - The filter in use                    ##')
    print('## OBJECT   - The object name e.g. SDSSJ1529+2928  ##')
    print('## INSTRUME - The instrument used. e.g. ProEM      ##')
    print('## OBSERVER - The initials of the observer         ##')
    print('#####################################################')
    print('')
    check_edit_headers = input('Would you like to proceed? (Y/[N]): ')
    print('')

    if (check_edit_headers == 'Y') or (check_edit_headers == 'y'):

        ## Guess the exposure time from the first frame.
        ## EXPTIME is the only header value that pre-exists
        ## from the Lightfield export-to-FITS process.
        hdu_temp  = fits.open(path + filenames[0])
        texp_read = float(hdu_temp[0].header['EXPTIME'])
        ## Check whether header exposure time is already in milliseconds or not
        ## 500ms is used as a limiting case since the shortest exposures
        ## are only ever 995ms.
        if texp_read > 500.0:
            texp_guess = round(texp_read/1000.0)
        elif texp_read < 500.0:
            texp_guess = texp_read
        hdu_temp.close()

        ## Define a function which runs the user through a
        ## prompting routine in order to change/keep a certain
        ## header keyword value.  An optional parameter is
        ## provided in case a 'best guess' at the header value
        ## is possible, such as for EXPTIME.
        def get_header_val(header_name,pass_value=None):
            
            ## Open the first FITS header
            hdu_temp = fits.open(path + filenames[0])
            
            ## Try reading the value of a header keyowrd
            try:
                ## If a pass_value was defined, use it to define header_value
                ## Else, try reading the header value from the FITS file
                if pass_value != None:
                    header_value = pass_value
                else:
                    ## This line throws a KeyError if the header_name
                    ## keyword does not exist in the FITS header
                    header_value = hdu_temp[0].header[header_name]
                
                ## If a current FILTER value exists, print it and
                ## ask the user if they want to change/keep it.
                print('%s = %s' %(header_name,header_value))
                change_value = input('Change %s value? (Y/[N]): ' %header_name)
            
                if (change_value == 'Y') or (change_value == 'y'):
                    new_value    = input('Please provide new value for %s: ' %header_name)
                    header_value = new_value
                    print('')
                else:
                    print('%s value was not changed.' %header_name)
                    print('')
            
            ## KeyError will be thrown if you try to read a header
            ## value that does not exist.  In this case, simply ask
            ## for a value that will be added into the header later.
            except KeyError:
                header_value = input('Please provide a value for %s: ' %header_name)

            ## Close the hdu
            hdu_temp.close()

            return header_value


        ## Use get_header_val to get values for each
        ## header keyword that needs to be added/edited
        texp        = get_header_val('EXPTIME', pass_value=texp_guess)
        filt_name   = get_header_val('FILTER')
        object_name = get_header_val('OBJECT')
        instr_name  = get_header_val('INSTRUME')
        observ_name = get_header_val('OBSERVER')
        print('')

        ## Let the user know what will be changed before proceeding
        print('The following values will be added to the FITS headers:')
        print('EXPTIME   = %3.1f' %float(texp))
        print('FILTER    = %s' %filt_name)
        print('OBJECT    = %s' %object_name)
        print('INSTRUME  = %s' %instr_name)
        print('OBSERVER  = %s' %observ_name)
        continue_edit_headers = input('Continue? (Y/[N]): ')
        print('')

        ## Defining a function to open, edit, and save a
        ## new FITS file containing the new header info
        def edit_FITS(fname, texp0, filt):
            with fits.open(fname, mode='update') as hdu:
                hdu[0].header.set('EXPTIME' ,texp0)                          
                hdu[0].header.set('FILTER'  ,filt_name  ,comment='Filter Type',before='LONGSTRN')      
                hdu[0].header.set('OBJECT'  ,object_name,comment='Object Name',before='LONGSTRN')    
                hdu[0].header.set('INSTRUME',instr_name ,comment='Instrument Name',before='LONGSTRN')   
                hdu[0].header.set('OBSERVER',observ_name,comment='Observer(s) Initials',before='LONGSTRN') 
                hdu.close() # Automatically saves changes to file in 'update' mode
            return

        if (continue_edit_headers == 'Y') or (continue_edit_headers == 'y'):
        
            ## Loop through all FITS files in the working list
            ## and change their header values
            for i in range(num_files):
            
                ## Print Progress Bar
                count1  = i+1
                action1 = 'Editing header values..................'
                progress_bar(count1, num_files, action1)
            
                ## Now to actually open, edit, and save the FITS files
                edit_FITS(path + filenames[i], float(texp), filt_name)
            
            print('')
            print('')
            print('FITS header values were successfully edited.')

        else:
            print('FITS headers were not changed.')

    else:
        print('FITS headers were not changed.')


    ################################################################
    ##
    ##  Now, let's perform the task which Keatons "mcdoheader2.py"
    ##  performs.  That is, we need to put the absolute timestamps
    ##  of start-exposure into the FITS header under the keywords of
    ##  DATE-OBS and TIME-OBS.  A lot of the code here was taken
    ##  directly from Keaton's script with some slight modifications.
    ##
    ################################################################

    ## First load the timestamps CSV data file.
    csv_name   = glob('*_timestamps.csv')
    path_csv   = path + csv_name[0]
    time_data  = read_csv(path_csv)

    ## Inform the user what values will be changed and ask whether to proceed
    print('')
    print('#####################################################')
    print('##  This section will add and edit the following   ##')
    print('##  FITS header values using a timestamps file:    ##')
    print('## ----------------------------------------------- ##')
    print('## DATE-OBS - UT Date at start of exposure         ##')
    print('## TIME-OBS - UT Time at start of exposure         ##')
    print('#####################################################')
    print('')
    print('Timestamps file:  %s' %csv_name[0])
    print('')
    check_add_times = input('Would you like to proceed? (Y/[N]): ')
    print('')

    if (check_add_times == 'Y') or (check_add_times == 'y'):
        
        ## Defining a function which gets the exposure
        ## time from the FITS header
        def get_exptime(path_to_fits):
            hdr = fits.getheader(path_to_fits)
            exptime   = float(hdr['EXPTIME'])
            return exptime
        
        ## Defining a function to add timestamps to FITS files
        def addtimestamp(fitsname,timestamp):
            with fits.open(fitsname.strip(), mode='update') as hdu:                                               
                hdu[0].header.set('DATE-OBS',str(timestamp.date()),comment='UT Date at Start of Exposure') 
                hdu[0].header.set('TIME-OBS',str(timestamp.time()),comment='UT Time of Start of Exposure',after='DATE-OBS') 
                hdu.close() # Automatically saves changes to FITS file in "update" mode
            return
        
        ## First, load the exposure times from the FITS
        ## frames and save them into a list.
        exp_times = []
        for i in range(num_files):
            ## Print progress bar
            count2  = i+1
            action2 = 'Loading Exp. Times from FITS headers...'
            progress_bar(count2, num_files, action2)
            
            texp1 = get_exptime(path + filenames[i])
            exp_times.append(texp1)
        
        
        ## Perform a check to make sure the exposure times
        ## are in units of seconds, not milliseconds.  In
        ## practice, no exposure time has ever been as long
        ## as 500 seconds, but almost all exposures are longer
        ## than 500 milliseconds, so I'm using 500 as the
        ## comparison value.
        exptime_zero = exp_times[0]
        print('')
        print('First-Frame Exposure time: {} seconds.'.format(exptime_zero))
        if exptime_zero > 500:
            print('WARNING: Your exposure time value is VERY LARGE!')
            print('Make sure you have corrected the header exposure')
            print('times and converted them from milliseconds to seconds')
        
        ## Convert the timestamp.csv file loaded in the
        ## previous section from a Pandas data frame
        ## object to a simple Numpy matrix array.
        raw_times = time_data.values
        
        ## Create empty lists to store data
        index   = []  # Frame tracking # for the images
        tstart  = []  # Time stamp for start of exposure
        tend    = []  # Time stamp for end of exposure
        dtstart = []  # t-delta b/t starts of current & previous frames
        dtend   = []  # t-delta b/t ends   of current & previous frames

        for line in raw_times:
        
            ## First split each line up into individually named values
            tindex, ttstart, ttend, tdtstart, tdtend = line
            ## Append the frame tracking # to the "index" list
            index.append(int(tindex))
            
            ## Next append the starting time stamp to "tstart".
            ## IF/ELSE statements check whether or not
            ## micro-seconds were included in the CSV.  It's
            ## typical for microseconds to be included unless
            ## the metadata was not saved properly.
            if len(ttstart) != 26: # If no microseconds
                tstart.append(dt.strptime(ttstart,'%Y-%m-%d %H:%M:%S'))
            else:                  # If there are micro-seconds
                tstart.append(dt.strptime(ttstart,'%Y-%m-%d %H:%M:%S.%f'))
            
            ## Next append the ending time stamp to "tend"
            if len(ttend) != 26:   # If no microseconds
                tend.append(dt.strptime(ttend,'%Y-%m-%d %H:%M:%S'))
            else:                  # If there are microseconds
                tend.append(dt.strptime(ttend,'%Y-%m-%d %H:%M:%S.%f'))

            ## Lastly, append the start-to-start and end-to-end
            ## delta-t values between adjacent exposures.
            if np.isnan(tdtstart) == False: # If not the first frame
                dtstart.append(float(tdtstart)/1e9)
                dtend.append(float(tdtend)/1e9)
            else:                            # If the first frame
                dtstart.append(exptime_zero)
                dtend.append(exptime_zero)


        ## All times are defined relative to first frame
        tzero = tstart[0]
        ms    = tzero.microsecond # Number of microseconds in time stamp
        
        ## Any reason to worry that GPS triggering was not used?
        ## This IF statement checks whether the first time stamp
        ## came more than 0.05 seconds before or after an
        ## integer second.
        if (np.abs(ms - 1e6) > 5e4) & (np.abs(ms) > 5e4):
            print("WARNING: First exposure > 0.05 seconds away from integer second.")
            print("Check that you were using GPS triggers.")

        ## Round tzero to the nearest second
        ms = tzero.microsecond
        if ms > 5e5: #round up
            tzero += td(microseconds = 1e6 - ms)
        else:        #round down
            tzero += td(microseconds = -1 * ms)
        
        ## Create a list to hold the final timestamp values
        ## which will then be added into the FITS headers
        times = [tzero]
        #Add first timestamp to first fits file
        addtimestamp(path + filenames[0],times[0])
        
        #Determine accurate timestamps and place in fits headers.
        for i in range(1,num_files):
            
            ## Print progress bar
            count3  = i+1
            action3 = 'Adding timestamps to FITS headers......'
            progress_bar(count3, num_files, action3)
            
            ## Get the exposure time for the current frame
            exptime = exp_times[i]
            
            ## Check that the expected time has elapsed.
            ## Unfortunately, there's an indexing issue here.
            ## The exp-time for one frame has to be compared
            ## to the "dtstart" of the next frame.  This works
            ## fine until the last frame when there is no
            ## longer a dtstart to compare to.  In that case.\,
            ## the "elif" statement below, I just compare to
            ## the "dtend" of the same frame and hope for the best.
            if   (i < num_files-1  and round(exptime) == round(dtstart[i+1])):
                times.append(times[i-1] + td(seconds = round(dtstart[i])))
            elif (i == num_files-1 and round(exptime) == round(dtend[i])):
                times.append(times[i-1] + td(seconds = round(dtstart[i])))
            else:
                print('')
                print('')
                print("WARNING: timestamp anomaly on frame {}".format(index[i]))
                ## Sometimes a bad timestamp comes down the line and
                ## is corrected on the next exposure. Check to see if
                ## things get back on track and the exposure time was
                ## really the expected duration.
                ## WARNING! The checks below may or may not work for
                ## multi-filter data, yet to be confirmed.
                if i < len(index)-3:
                    
                    dt_check1  = dtstart[i]+dtstart[i+1]+dtstart[i+2]
                    dt_check2  = dtstart[i-1]+dtstart[i]+dtstart[i+1]
                    exp_check1 = exp_times[i]+exp_times[i+1]+exp_times[i+2]
                    exp_check2 = exp_times[i-1]+exp_times[i]+exp_times[i+1]
                    
                    if round(dt_check1) == round(exp_check1):
                        print("It appears to get back on track.")
                        times.append(times[i-1] + td(seconds = round(exptime)))
                    elif round(dt_check2) == round(exp_check2):
                        print("Making up for the last frame.")
                        times.append(times[i-1] + td(seconds = round(exptime)))
                    else:
                        print("Looks like triggers were missed.")
                        times.append(times[i-1] + td(seconds = round(dtstart[i])))
                else:
                    print("Last couple of timestamps from this run are suspect.")
                    times.append(times[i-1] + td(seconds = round(exptime)))

            ## Add timestamp to fits file:
            addtimestamp(path + filenames[i],times[-1])

        print('')
        print('')
        print('Successfully added UT timestamps to FITS headers.')
        print('')

    else:
        print('Timestamps were not added to the FITS headers.')
        print('')

 


# I had to make a unique function for PTO+PRISM data, since we don't have as robust time-keeping
def sf_impar_perkins(path, ilist):
    #########################################################
    ##
    ##  Load in the file names which need to be parsed
    ##
    #########################################################

    path = getcwd() + '/'  # Get the current working directory

    ## Ask the user to supply the name of a file
    ## containing a list of all images to be worked on
    print('')
    file_list = 'ilist'
    try:
        filenames  = np.loadtxt(path + file_list, dtype=str)
    except IOError:
        try:
            print('Could not find a working list using default name "%s"' %file_list)
            file_list = input("Please enter the working list name to search for: ")
            filenames  = np.loadtxt(path + file_list, dtype=str)
        except IOError:
            print('Could not find a working list with name "%s"' %file_list)
            print('\nProgram TERMINATED\n')
            sys.exit(1)

    ## Save the original file names for comparison later
    og_filenames = filenames
    num_files    = len(filenames)


    ###############################################################
    ##
    ##  First, let's change the exposure time value in
    ##  the actual headers of the FITS files (EXPTIME)
    ##  and add in the filter information with a new
    ##  keyword, FILTER.
    ##
    ###############################################################

    ## Inform the user what values will be changed and ask whether to proceed
    print('')
    print('#####################################################')
    print('##          This section will add and edit         ##')
    print('##         the following FITS header values:       ##')
    print('## ----------------------------------------------- ##')
    print('## EXPTIME  - The image exposure time              ##')
    print('## FILTER   - The filter in use                    ##')
    print('## OBJECT   - The object name e.g. SDSSJ1529+2928  ##')
    print('## INSTRUME - The instrument used. e.g. ProEM      ##')
    print('## OBSERVER - The initials of the observer         ##')
    print('#####################################################')
    print('')
    check_edit_headers = input('Would you like to proceed? (Y/[N]): ')
    print('')

    if (check_edit_headers == 'Y') or (check_edit_headers == 'y'):

        ## Guess the exposure time from the first frame.
        ## EXPTIME is the only header value that pre-exists
        ## from the Lightfield export-to-FITS process.
        hdu_temp  = fits.open(path + filenames[0])
        texp_read = float(hdu_temp[0].header['EXPTIME'])
        ## Check whether header exposure time is already in milliseconds or not
        ## 500ms is used as a limiting case since the shortest exposures
        ## are only ever 995ms.
        if texp_read > 500.0:
            texp_guess = round(texp_read/1000.0)
        elif texp_read < 500.0:
            texp_guess = texp_read
        hdu_temp.close()

        ## Define a function which runs the user through a
        ## prompting routine in order to change/keep a certain
        ## header keyword value.  An optional parameter is
        ## provided in case a 'best guess' at the header value
        ## is possible, such as for EXPTIME.
        def get_header_val(header_name,pass_value=None):
            
            ## Open the first FITS header
            hdu_temp = fits.open(path + filenames[0])
            
            ## Try reading the value of a header keyowrd
            try:
                ## If a pass_value was defined, use it to define header_value
                ## Else, try reading the header value from the FITS file
                if pass_value != None:
                    header_value = pass_value
                else:
                    ## This line throws a KeyError if the header_name
                    ## keyword does not exist in the FITS header
                    header_value = hdu_temp[0].header[header_name]
                
                ## If a current FILTER value exists, print it and
                ## ask the user if they want to change/keep it.
                print('%s = %s' %(header_name,header_value))
                change_value = input('Change %s value? (Y/[N]): ' %header_name)
            
                if (change_value == 'Y') or (change_value == 'y'):
                    new_value    = input('Please provide new value for %s: ' %header_name)
                    header_value = new_value
                    print('')
                else:
                    print('%s value was not changed.' %header_name)
                    print('')
            
            ## KeyError will be thrown if you try to read a header
            ## value that does not exist.  In this case, simply ask
            ## for a value that will be added into the header later.
            except KeyError:
                header_value = input('Please provide a value for %s: ' %header_name)

            ## Close the hdu
            hdu_temp.close()

            return header_value


        ## Use get_header_val to get values for each
        ## header keyword that needs to be added/edited
        texp        = get_header_val('EXPTIME', pass_value=texp_guess)
        filt_name   = get_header_val('FILTER')
        object_name = get_header_val('OBJECT')
        instr_name  = get_header_val('INSTRUME')
        observ_name = get_header_val('OBSERVER')
        print('')

        ## Let the user know what will be changed before proceeding
        print('The following values will be added to the FITS headers:')
        print('EXPTIME   = %3.1f' %float(texp))
        print('FILTER    = %s' %filt_name)
        print('OBJECT    = %s' %object_name)
        print('INSTRUME  = %s' %instr_name)
        print('OBSERVER  = %s' %observ_name)
        continue_edit_headers = input('Continue? (Y/[N]): ')
        print('')

        ## Defining a function to open, edit, and save a
        ## new FITS file containing the new header info
        def edit_FITS(fname, texp0, filt):
            with fits.open(fname, mode='update') as hdu:
                hdu[0].header.set('EXPTIME' ,texp0)                          
                hdu[0].header.set('FILTER'  ,filt_name  ,comment='Filter Type')#,before='LONGSTRN')      
                hdu[0].header.set('OBJECT'  ,object_name,comment='Object Name')#,before='LONGSTRN')    
                hdu[0].header.set('INSTRUME',instr_name ,comment='Instrument Name')#,before='LONGSTRN')   
                hdu[0].header.set('OBSERVER',observ_name,comment='Observer(s) Initials')#,before='LONGSTRN') 
                hdu.close() # Automatically saves changes to file in 'update' mode
            return

        if (continue_edit_headers == 'Y') or (continue_edit_headers == 'y'):
        
            ## Loop through all FITS files in the working list
            ## and change their header values
            for i in range(num_files):
            
                ## Print Progress Bar
                count1  = i+1
                action1 = 'Editing header values..................'
                progress_bar(count1, num_files, action1)
            
                ## Now to actually open, edit, and save the FITS files
                edit_FITS(path + filenames[i], float(texp), filt_name)
            
            print('')
            print('')
            print('FITS header values were successfully edited.')

        else:
            print('FITS headers were not changed.')

    else:
        print('FITS headers were not changed.')



    ################################################################
    ##
    ##  Now, let's perform the task which Keatons "mcdoheader2.py"
    ##  performs.  That is, we need to put the absolute timestamps
    ##  of start-exposure into the FITS header under the keywords of
    ##  DATE-OBS and TIME-OBS.  A lot of the code here was taken
    ##  directly from Keaton's script with some slight modifications.
    ##
    ################################################################

    ## First load the timestamps CSV data file.
    # print(num_files)
    # cols = [["frame_tracking_number","time_stamp_exposure_started","time_stamp_exposure_ended","diff_time_stamp_exposure_started","diff_time_stamp_exposure_ended"]]
    # write_df = DataFrame(list(zip([np.nan]*num_files, [np.nan]*num_files, [np.nan]*num_files, [np.nan]*num_files, [np.nan]*num_files)), columns=cols)
    # write_df.to_csv(object_name+"_timestamps.csv", index=False)

    # csv_name   = glob('*_timestamps.csv')
    # path_csv   = path + csv_name[0]
    # time_data  = read_csv(path_csv)

    ## Inform the user what values will be changed and ask whether to proceed
    print('')
    print('#####################################################')
    print('##  This section will add and edit the following   ##')
    print('##  FITS header values so they conform with the    ##')
    print('##  hsp_nd and phot2lc reduction routines.         ##')
    print('## ----------------------------------------------- ##')
    print('##  DATE-OBS - UT Date at start of exposure        ##')
    print('##  TIME-OBS - UT Time at start of exposure        ##')
    print('#####################################################')
    print('')
    # print('Timestamps file:  %s' %csv_name[0])
    # print('')
    check_add_times = input('Would you like to proceed? (Y/[N]): ')
    print('')

    if (check_add_times == 'Y') or (check_add_times == 'y'):
        
        ## Defining a function which gets the exposure
        ## time from the FITS header
        def get_exptime(path_to_fits):
            hdr = fits.getheader(path_to_fits)
            exptime   = float(hdr['EXPTIME'])
            return exptime

        def get_utc_start(path_to_fits):
            hdr = fits.getheader(path_to_fits)
            return hdr['UTCSTART']
        
        # ## Defining a function to add timestamps to FITS files
        # def addtimestamp(fitsname,timestamp):
        #     with fits.open(fitsname.strip(), mode='update') as hdu:                                               
        #         hdu[0].header.set('DATE-OBS',str(timestamp.date()),comment='UT Date at Start of Exposure') 
        #         hdu[0].header.set('TIME-OBS',str(timestamp.time()),comment='UT Time of Start of Exposure',after='DATE-OBS') 
        #         hdu.close() # Automatically saves changes to FITS file in "update" mode
        #     return

        ## Defining a function to add timestamps to FITS files
        def addtimestamp(fitsname,timestamp):
            with fits.open(fitsname.strip(), mode='update') as hdu:                                               
                # hdu[0].header.set('DATE-OBS',str(timestamp.date()),comment='UT Date at Start of Exposure') 
                hdu[0].header.set('TIME-OBS',str(timestamp),comment='UT Time of Start of Exposure',after='DATE-OBS') 
                hdu.close() # Automatically saves changes to FITS file in "update" mode
            return
        
        ## First, load the exposure times from the FITS
        ## frames and save them into a list.
        exp_times = []
        for i in range(num_files):
            ## Print progress bar
            count2  = i+1
            action2 = 'Loading Exp. Times from FITS headers...'
            progress_bar(count2, num_files, action2)
            
            texp1 = get_exptime(path + filenames[i])
            exp_times.append(texp1)
        
        
        ## Perform a check to make sure the exposure times
        ## are in units of seconds, not milliseconds.  In
        ## practice, no exposure time has ever been as long
        ## as 500 seconds, but almost all exposures are longer
        ## than 500 milliseconds, so I'm using 500 as the
        ## comparison value.
        exptime_zero = exp_times[0]
        print('')
        print('First-Frame Exposure time: {} seconds.'.format(exptime_zero))
        if exptime_zero > 500:
            print('WARNING: Your exposure time value is VERY LARGE!')
            print('Make sure you have corrected the header exposure')
            print('times and converted them from milliseconds to seconds')

        
        #Determine accurate timestamps and place in fits headers.
        for i in range(1,num_files):
            
            ## Print progress bar
            count3  = i+1
            action3 = 'Adding timestamps to FITS headers......'
            progress_bar(count3, num_files, action3)

            addtimestamp(path + filenames[i],get_utc_start(path + filenames[i]))
            
        print('')
        print('Successfully added UT timestamps to FITS headers.')
        print('')

    else:
        print('Timestamps were not added to the FITS headers.')
        print('')
 


# Function to acquire exposure time
def get_texp(fname, instrument):
	hdu    = fits.open(fname)                    
	texp_read = float(hdu[0].header['EXPTIME'])
	hdu.close()
	if instrument == 'proem' or instrument == 'ProEM':
		print(texp_read,texp_read/1000.0,round(texp_read/1000.0),int(round(texp_read/1000.0)))
		return int(round(texp_read/1000.0))
	elif instrument == 'prism' or instrument == 'PRISM' or instrument=='lmi' or instrument=='LMI':
		return int(texp_read)



# Grab the image dimensions
def get_images_dimensions(image_name):
	with fits.open(image_name) as hdul:
			hdr = hdul[0].header
			return hdr['NAXIS1'],hdr['NAXIS2']


# Function to get the photometric band-pass filter
def get_filter(fname, instrument):
		hdu = fits.open(fname) 
		if instrument=='prism' or instrument=='PRISM':                   
			filter_name = str(hdu[0].header['FILTNME3'])
		elif instrument=='lmi' or instrument=='LMI':
			filter_name = str(hdu[0].header['FILTER1'])
            # filter_name = str(hdu[0].header['FILTER'])
		else:
			filter_name = str(hdu[0].header['FILTER'])
		hdu.close()
		return str(filter_name)



# Function to collate biases into a single master frame
def multibias(path,instrument):
    try:
        ims = glob(path + '*.fits').remove('test.fits')
    except ValueError:
        ims = glob(path + '*.fits')
    # Iterate through dlists to median combine the respective darks and write out master darks:
    final_empty = []
    for i,b in enumerate(ims):
        with fits.open(b) as hdul:
            # Grab header of first image for writing out
            if i == 0:
                hdr = hdul[0].header
                hdr['COMMENT'] = "Master Bias"
                hdr['COMMENT'] = "Median combined"
            # Append data into empty array
            if instrument == 'proem' or instrument == 'ProEM':
                final_empty.append(hdul[0].data[0])
            else:
                final_empty.append(hdul[0].data)
			# Delete data to avoid mmap getting angry
            del hdul[0].data
    # Perform median combine and then write out
    master_bias = np.nanmedian(final_empty,axis=0)
    print('\nMaster bias written to:',path+'Bias.fits\n')
    fits.writeto(path+'Bias.fits',data=master_bias,header=hdr,overwrite=True)
	# Eliminate cosmic rays
	# bias_og = CCDData(final_bias,unit=u.adu)
	# master_bias_cr = ccdproc.cosmicray_lacosmic(bias_og,gain_apply=False,sigclip=5)
	# master_bias_cr.unit = u.adu
	# fits.writeto(path+'Bias.fits',data=master_bias_cr.data,header=hdr,overwrite=True)
	# return master_bias_cr.data
    return master_bias
			



# Function to collate darks into a single master frame
def multidark(path,master_bias,instrument,texp_science):
	if instrument == 'proem' or instrument == 'ProEM' or instrument == 'PROEM':
		# try:
		dark_names = glob(path + 'dark_*.spe')
		if len(dark_names)==0:
			dark_names = glob(path + 'dark*.spe')


		for i in range(len(dark_names)):
	    # get the name of each dark exposure
			dname = dark_names[i][0:-4]
			if len(dname.split('/')[-1].strip()) <= 6:
				dsuff = dname.split('/')[-1].strip()[4:]+'s'
			else:
				dsuff = dname.split('/')[-1].strip()[5:]

			# search for the FITS files using dname
			try:
				flist = sorted(glob(dname + '-*.fits'))
			except Exception as ex:
				print(ex)
				print('Could not generate a list of FITS files for name: %s' %dname)
				print('No List was generated for this exposure time.')
				continue
			# generate the string format
			file_len = str(len(flist[0]))
			f_format = '%' + file_len + 's'

			# save the file names into a list
			lname = 'dlist_' + dsuff
			np.savetxt(path+lname,flist,fmt=f_format,delimiter = ' ')

		# Grab all the dlists:
		dlists = glob(path+'dlist_*s')

		# Iterate through dlists to median combine the respective darks and write out master darks:
		for d in dlists:
			# Grab exposure time for subtracting out master dark
			# t_exp = d[-3:]
			t_exp = d.split('_')[-1].split('s')[0].strip()
			# Load in individual images
			ims = np.loadtxt(d,dtype='str')
			# Make a master array to perform combining on:
			master_empty = []
			# Loop through indiviual images to append data into master array:
			for i in range(len(ims)):
				with fits.open(path+ims[i]) as hdul:
					# Grab header of first image for writing out
					if i == 0:
						hdr = hdul[0].header
						hdr['COMMENT'] = "Master " + t_exp + " Dark"
						hdr['COMMENT'] = "Median combined and bias subtracted"
					# Append data into empty array
					if instrument == 'prism' or instrument == 'PRISM':
						master_empty.append(hdul[0].data)
					elif instrument == 'proem' or instrument == 'ProEM':
						master_empty.append(hdul[0].data[0] - master_bias)
					# Delete data to avoid mmap getting angry
					del hdul[0].data
			# Perform median combine and then write out
			master_dark = np.nanmedian(master_empty,axis=0)
			# Eliminate cosmic rays
			# dark_og = CCDData(master_dark,unit=u.adu)
			# master_dark_cr = ccdproc.cosmicray_lacosmic(dark_og,gain_apply=False,sigclip=5)
			# master_dark_cr.unit = u.adu
			# fits.writeto(path+'Dark_'+t_exp+'s.fits',data=master_dark_cr.data,header=hdr,overwrite=True)
			fits.writeto(path+'Dark_'+t_exp+'s.fits',data=master_dark,header=hdr,overwrite=True)

		# Return image 
		with fits.open(glob(path+'Dark_'+texp_science+'s.fits')[0]) as hdul:
			return hdul[0].data
	elif instrument == 'prism' or instrument == 'PRISM' or instrument == 'LMI' or instrument == 'lmi':
		return np.zeros(np.shape(master_bias)[::-1])



# Function to collate flats into a single master frame
def multiflat(path, master_bias, instrument, skip_darks):
    if instrument == 'proem' or instrument == 'ProEM':
        flat_names = sorted(glob(path+'*.spe'))
        for i in range(len(flat_names)):
            # get the name of each flat field exposure
            fname = flat_names[i][0:-4] #removes .spe extension
            fsuff = fname.split('/')[-1].strip()
            # search for the FITS files using dname
            try:
                flist = sorted(glob(fname + '*.fits'))
            except Exception as ex:
                print(ex)
                print('Could not generate a list of FITS files for name: %s' %fname)
                print('No List was generated for this flat field exposure.')
                continue
            # if this script is being run more than once, some files may 
            # already exist with the "ds" suffix added on.  Filter these out.
            bool_list = [x[-7:-5] == 'ds' for x in flist]
            ds_ind = [x for x, y in enumerate(bool_list) if not y]
            flist = [flist[j] for j in ds_ind]
            # generate the string format
            file_len = str(len(flist[0]))
            # save the file names into a list
            lname = 'flist_' + fsuff
            np.savetxt(path+lname,flist,fmt='%s',delimiter = ' ')
		    # now create the output list for dark subtracted flats
		    # flist_ds = [x[0:-5] + 'ds' + x[-5:] + '[0]' for x in flist]
		    # flist_ds = [x[0:-5] + 'ds' + x[-5:] for x in flist]
		    # lname_ds = lname + '_ds'
		    # np.savetxt(path+lname_ds,flist_ds,fmt='%s',delimiter=' ')
	    
    elif instrument == 'prism' or instrument == 'PRISM' or instrument == 'lmi' or instrument == 'LMI':
        try:
            flat_names = sorted(glob(path+'*.fits')).remove('test.fits')
        except ValueError:
            flat_names = sorted(glob(path+'*.fits'))
        # Find out how many different filters there are
        filts = np.array([[get_filter(f,instrument),f] for f in flat_names])
        unique_filts = sorted(list(set(filts[:,0])))
        for f in unique_filts:
            flist = filts[:,1][np.where(filts[:,0] == f)]
            bool_list = [x[-7:-5] == 'ds' for x in flist]
            ds_ind = [x for x, y in enumerate(bool_list) if not y]
            flist = [flist[j] for j in ds_ind]
            file_len = str(len(flist[0]))
            f_format = '%' + file_len + 's'
            # save the file names into a list
            t_exp = str(int(get_texp(flist[0],instrument)))+'s'
            lname = 'flist_' + f + '_' + t_exp
            np.savetxt(path+lname,flist,fmt='%s',delimiter = ' ')
		    # now create the output list for dark subtracted flats
		    # flist_ds = [x[0:-5] + 'ds' + x[-5:] + '[0]' for x in flist]
		    # flist_ds = [x[0:-5] + 'ds' + x[-5:] for x in flist]
		    # lname_ds = lname + '_ds'
		    # np.savetxt(path+lname_ds,flist_ds,fmt='%s',delimiter=' ')
    
    # Grab all the flists:
    flists = glob(path+'flist*')
    # Make a master array to perform combining on:
    master_empty = []
    # Iterate through dlists to median combine the respective darks and write out master darks:
    # for i,f in enumerate(flat_names):
    for l in flists:
        flat_names = np.loadtxt(l,dtype=str,delimiter=' ')
        for i,f in enumerate(flat_names):
            if i==0:
                # Grab exposure time for writing our master flat
                t_exp_flat = str(get_texp(f,instrument))
                # Grab master dark with correct texp for the flats
                if skip_darks:
                    master_dark_flat = np.zeros((xdim,ydim))
                else:
                    with fits.open(glob('../dark/Dark*'+t_exp_flat+'s.fits')[0]) as hdul:
                        master_dark_flat = hdul[0].data
                # Grab filter for writing out master flat
                # filt = get_filter(f,instrument)
            # Loop through indiviual images to append data into master array:
            with fits.open(f) as hdul:
                # Grab header of first image for writing out
                if i == 0:
                    hdr = hdul[0].header
                    hdr['COMMENT'] = "Master " + filter_name + " Flat"
                    hdr['COMMENT'] = "Median combined"
                    hdr['COMMENT'] = "Bias and dark subtracted"
                # Append data into empty array
                if instrument == 'prism' or instrument == 'PRISM' or instrument=='LMI' or instrument=='lmi':
                    master_empty.append(hdul[0].data - master_bias)
                elif instrument == 'proem' or instrument == 'ProEM' or instrument=='PROEM':
                    master_empty.append(hdul[0].data[0] - master_dark_flat - master_bias)
                # Delete data to avoid mmap getting angry
                del hdul
        # Perform median combine, mode normalize and then write out
        master_flat = np.nanmedian(master_empty,axis=0)
        # mask to greater than bias to avoid overscans
        if instrument == 'prism' or instrument == 'PRISM':
            master_flat /= mode(master_flat[master_flat>master_bias.max()],axis=None,keepdims=False).mode
        elif instrument=='LMI' or instrument=='lmi': # also mask out oversaturated overscan region
            master_flat /= mode(master_flat[(master_flat>master_bias.max()) & (master_flat<4e4)],axis=None,keepdims=False).mode
        else:
            master_flat /= mode(master_flat,axis=None,keepdims=False).mode

        # Eliminate cosmic rays
        # mf_og = CCDData(master_flat,unit=u.adu)
        # master_flat_cr = ccdproc.cosmicray_lacosmic(mf_og,gain_apply=False,sigclip=5)
        # master_flat_cr.unit = u.adu
        if 'sky' in path or 'Sky' in path:
            fits.writeto(path+'Sky_Flat_'+filter_name+'_'+t_exp_flat+'s.fits',data=master_flat,header=hdr,overwrite=True)
        elif 'dome' in path or 'Dome' in path:
            fits.writeto(path+'Dome_Flat_'+filter_name+'_'+t_exp_flat+'s.fits',data=master_flat,header=hdr,overwrite=True)
            # fits.writeto(path+'Dome_Flat_'+filter_name+'_'+t_exp_flat+'s.fits',data=master_flat_cr.data,header=hdr,overwrite=True)
		# return master_flat


# Finally reduce your raw science images with your master calibration iamges
def reduce_ims(path,ilist,olist,master_bias,master_dark,master_flat,instrument):
    # Initialize progress bar:
    action = 'Reducing Images...' # Progress bar message
    progress_bar(0,len(ilist),action)
    # Loop through each image to read in, dark subtract, flat field, and then write out reduced image:
    for i in range(len(ilist)):
        with fits.open(path + ilist[i]) as hdul:
            progress_bar(i+1,len(ilist),action)
            if instrument=='proem' or instrument=='ProEM' or instrument=='PROEM':
                ccd_og = CCDData(hdul[0].data[0],unit=u.adu)
                # Correct for cosmic rays
                ccd = ccdproc.cosmicray_lacosmic(ccd_og,gain_apply=False,sigclip=5)
                ccd.unit = u.adu
                ccd.header['exposure'] = float(texp_science)
                reduced = ccdproc.ccd_process(ccd, #oscan='[201:232,1:100]',
                                    # trim='[4:{}, 1:{}]'.format(int(xdim-40),int(ydim)),
                                    master_bias=CCDData(master_bias,unit=u.adu),
                                    gain_corrected=True,
                                    dark_frame=CCDData(master_dark,unit=u.adu,meta={'exposure':float(texp_science)}),
                                    exposure_key='exposure',
                                    exposure_unit=u.second,
                                    dark_scale=False,
                                    master_flat=CCDData(master_flat,unit=u.adu))
            elif instrument == 'prism' or instrument == 'PRISM' or instrument=='LMI' or instrument=='lmi':
                ccd = CCDData(hdul[0].data,unit=u.adu)
                ccd.header['exposure'] = float(texp_science)
                reduced = ccdproc.ccd_process(ccd, #oscan='[201:232,1:100]',
                                    # trim='[4:{}, 1:{}]'.format(int(xdim-40),int(ydim)),
                                    master_bias=CCDData(master_bias,unit=u.adu),
                                    gain_corrected=True,
                                    dark_frame=CCDData(master_dark,unit=u.adu,meta={'exposure':float(texp_science)}),#FIX
                                    exposure_key='exposure',
                                    exposure_unit=u.second,
                                    dark_scale=False,
                                    master_flat=CCDData(master_flat,unit=u.adu))
            hdr = hdul[0].header
            hdr['COMMENT'] = 'Image bias and dark subtracted and flat-fielded.'
            # Remove any NaNs that might exist
            if instrument == 'prism' or instrument == 'PRISM':
                im_no_nans = reduced.data[:,5:int(xdim-40)] # Trim overscans to avoid issues with hipercam reduce
            else:
                im_no_nans = reduced.data
            im_no_nans[np.isnan(im_no_nans)] = np.nanmedian(im_no_nans)
            fits.writeto(path + olist[i], data=im_no_nans, header=hdr, overwrite=True)
            del hdul, reduced, im_no_nans
    print('\nFinished reducting images! \n')




parser = argparse.ArgumentParser(description='Provide path name to data directory.')
parser.add_argument('-p', '--path',type=str,default='./',
                    help="Path to directory with images to reduce and calibrate.")
parser.add_argument('-i', '--instrument',type=str,default='PRISM',
                    help="Name of instrument used to collect data. Needed to parse image headers.")
args = parser.parse_args()
instrument = args.instrument
skipdarks=False
if instrument=='prism' or instrument=='PRISM' or instrument=='lmi' or instrument=='LMI':
    skipdarks = True


# Get the current working directory
path = getcwd() + '/'  


# Make ilist and olist
try:
    ilist, olist, hcm_files = np.loadtxt('ilist',dtype=str), np.loadtxt('olist',dtype=str), np.loadtxt('hcm.lis',dtype=str)
except FileNotFoundError:
    ilist, olist, hcm_names = make_ilist(path,instrument)
# Get image dimensions
xdim, ydim = get_images_dimensions(ilist[0])


# Edit image headers
if instrument=='prism' or instrument=='PRISM' or instrument=='lmi' or instrument=='LMI':
	sf_impar_perkins(path,ilist)
else:
	sf_impar(path,ilist)

# Get filter name
filter_name = get_filter(ilist[0],instrument)

# Grab the exposure time
with fits.open(ilist[0]) as hdul:
	texp_science = str(int(hdul[0].header['EXPTIME']))


##### Reudce biases #####
# First look to see if a master bias already exists:
if isfile('../bias/Bias.fits'):
	print('\nYou already have a master bias image. Proceeding ahead...\n')
	with fits.open('../bias/Bias.fits') as hdul:
		if instrument=='proem' or instrument=='ProEM' or instrument=='PROEM':
			master_bias = hdul[0].data
		elif instrument=='prism' or instrument=='PRISM' or instrument=='lmi' or instrument=='LMI':
			master_bias = hdul[0].data
else:
	try:
		print('Making master bias...')
		master_bias= multibias('../bias/',instrument)
	except (FileNotFoundError,UnboundLocalError):
		bias_path = input('Enter the path to your biases directory from your current working directory (e.g., "../bias/") or enter "N" or "n" to skip biases: ')
		if bias_path == "N" or bias_path == "n":
			master_bias = np.zeros((xdim,ydim))
		else:
			master_bias = multibias(bias_path,instrument)

##### Reudce Darks #####
# First look to see if a master flat already exists:
if instrument=='proem' or instrument=='ProEM' or instrument=='PROEM':
	try:
		if isfile(glob('../dark/Dark_*'+texp_science+'s.fits')[0]):
			print('\nYou already have a master dark image. Proceeding ahead...\n')
			print('Opening:',glob('../dark/Dark_*'+texp_science+'s.fits')[0],'\n')
			with fits.open(glob('../dark/Dark_*'+texp_science+'s.fits')[0]) as hdul:
				master_dark = hdul[0].data
	except IndexError:
		try:
			master_dark = multidark('../dark/',master_bias,instrument,texp_science)
		except FileNotFoundError:
			dark_path = input('Enter the path to your darks directory from your current working directory (e.g., "../dark/") or enter "N" or "n" to skip darks: ')
			if dark_path != "N" or dark_path != "n":
				master_dark = multidark(dark_path,master_bias,instrument,texp_science)
			else:
				master_dark = np.zeros((xdim,ydim))
elif instrument=='prism' or instrument=='PRISM' or instrument=='lmi' or instrument=='LMI':
	master_dark = np.zeros_like(master_bias)



##### Reudce Flats #####
# First look to see if a master flat already exists:
try:
	# Check that you have the case of the filter name to correctly match the images
	try:
		isfile(glob('../dome_flat/Dome_Flat_*'+filter_name+'*.fits')[0])
	except IndexError:
		filter_name = filter_name.lower()
	# Load in the flats
	if isfile(glob('../dome_flat/Dome_Flat_*'+filter_name+'*.fits')[0]):
		print('\nYou already have a master dome flat image. Proceeding ahead...\n')
		print('Opening:',glob('../dome_flat/Dome_Flat_*'+filter_name+'*.fits')[0])
		with fits.open(glob('../dome_flat/Dome_Flat_*'+filter_name+'*.fits')[0]) as hdul:
			if instrument=='proem' or instrument=='ProEM' or instrument=='PROEM':
				master_flat = hdul[0].data
			elif instrument=='prism' or instrument=='PRISM' or instrument=='lmi' or instrument=='LMI':
				master_flat = hdul[0].data
	elif isfile('../sky_flat/Sky_Flat*.fits'):
		print('\nYou already have a master sky flat image. Proceeding ahead...\n')
		with fits.open('../sky_flat/Sky_Flat*.fits') as hdul:
			if instrument=='proem' or instrument=='ProEM' or instrument=='PROEM':
				master_flat = hdul[0].data[0]
			elif instrument=='prism' or instrument=='PRISM' or instrument=='lmi' or instrument=='LMI':
				master_flat = hdul[0].data
except IndexError:
	try:
		multiflat('../dome_flat/',master_bias,instrument,skip_darks=skipdarks)
		with fits.open(glob('../dome_flat/Dome_Flat*'+filter_name+'*.fits')[0]) as hdul:
			master_flat = hdul[0].data
	except (FileNotFoundError,IndexError):
		try:
			multiflat('../sky_flat/',master_bias,instrument,skip_darks=skipdarks)
			with fits.open(glob('../sky_flat/Sky_Flat*'+filter_name+'*.fits')[0])  as hdul:
				master_flat = hdul[0].data
		except (FileNotFoundError,IndexError):
			flat_path = input('Enter the path to your flats directory from your current working directory and search string (e.g., "../flats/*.fits"). Enter "N" to pass. : ')
			if flat_path!='n' or flat_path!='N':
				multiflat(flat_path,master_bias,instrument,skip_darks=skipdarks)
				with fits.open(glob(flat_path+'*Flat*'+filter_name+'*.fits')[0]) as hdul: #get_filter(ilist[0],instrument)
					master_flat = hdul[0].data
			else:
				master_flat=np.zeros((xdim,ydim))+1.


# Reduce images
# Check image dimensions before reducing:
if len(np.shape(master_bias))==3:
	master_bias=master_bias[0]
if len(np.shape(master_dark))==3:
	master_dark=master_dark[0]
if len(np.shape(master_flat))==3:
	master_flat=master_flat[0]

# Reduce your images
reduce_ims(path,ilist,olist,master_bias,master_dark,master_flat,instrument)

# Do preparations for other hipercam routines
# Make a hcm file directory for fits2hcm
if isdir(path+'hcm_files/')==False:
	mkdir(path+'hcm_files/')

# Create a blank aperture.ape file
with open("aperture.ape", "w") as file:
    file.write("[\n")  # Write the first line with a left bracket
    file.write("]")  # Write the second line with a right bracket

# Copy the correct reduce.red file
if instrument=='proem' or instrument=='ProEM' or instrument=='PROEM':
    copyfile('/Users/astrojoe/Research/hipercam/reduce_proem.red','reduce.red')
if instrument=='prism' or instrument=='PRISM':
    copyfile('/Users/astrojoe/Research/hipercam/reduce_prism.red','reduce.red')
if instrument=='lmi' or instrument=='LMI':
    copyfile('/Users/astrojoe/Research/hipercam/reduce_lmi.red','reduce.red')

# Suppress ImportError
try:
    print('')
except ImportError:
    print('')

