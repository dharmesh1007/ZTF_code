
# Code from Roy Williams to convert magpsf into DCmag.
def dc_mag(fid, magpsf,sigmapsf, magnr,sigmagnr, magzpsci, isdiffpos):
    import math
    """ Compute apparent magnitude from difference magnitude supplied by ZTF
    Parameters
    ----------
    fid
        filter, 1 for green and 2 for red
    magpsf,sigmapsf
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr
        magnitude of nearest source in reference image PSF-catalog within 30 arcsec
        and 1-sigma error
    magzpsci
        Magnitude zero point for photometry estimates
    isdiffpos
        t or 1 => candidate is from positive (sci minus ref) subtraction; 
        f or 0 => candidate is from negative (ref minus sci) subtraction
    """

    # zero points. Looks like they are fixed.
    ref_zps = {1:26.325, 2:26.275, 3:25.660}
    magzpref = ref_zps[fid]

    # reference flux and its error
    magdiff = magzpref - magnr
    if magdiff > 12.0:
        magdiff = 12.0
    ref_flux = 10**( 0.4* ( magdiff) )
    ref_sigflux = (sigmagnr/1.0857)*ref_flux

    # difference flux and its error
    if magzpsci == 0.0: magzpsci = magzpref
    magdiff = magzpsci - magpsf
    if magdiff > 12.0:
        magdiff = 12.0
    difference_flux = 10**( 0.4* ( magdiff) )
    difference_sigflux = (sigmapsf/1.0857)*difference_flux

    # add or subract difference flux based on isdiffpos
    if isdiffpos == 't': dc_flux = ref_flux + difference_flux
    else:                dc_flux = ref_flux - difference_flux

    # assumes errors are independent. Maybe too conservative.
    dc_sigflux =  math.sqrt( difference_sigflux**2 + ref_sigflux**2 )

    # apparent mag and its error from fluxes
    if dc_flux > 0.0:
        dc_mag = magzpsci - 2.5 * math.log10(dc_flux)
        dc_sigmag = dc_sigflux/dc_flux*1.0857
    else:
        dc_mag = magzpsci
        dc_sigmag = sigmapsf

    return {'dc_mag':dc_mag, 'dc_sigmag':dc_sigmag}


# Function to store light curves in a dictionary
def store_LCs(startPath, LC_folderPath, objectList, date):
    import os
    import json
    import pandas as pd

    # Storage dictionary
    LC_dict = {}

    # Change to start path
    os.chdir(startPath)
    print(f'Directory before LC downloads: {os.getcwd()}')
    
    # Change to LC folder directory
    os.chdir(LC_folderPath)
    print(f'Folder Path: {os.getcwd()}')

    # Iterate through lists in object_lists.
    for count, oid in enumerate(objectList, start=0):
        print(count, oid)
        # Store light curves in dictionary for easy access.
        with open(f'{oid}_{date}.json') as json_file:
            LC_dict[oid] = pd.DataFrame(json.load(json_file))

    # Return to original folder path.
    os.chdir(startPath)
    print(f'Directory affter LC downloads: {os.getcwd()}')
    return LC_dict


# Function to remove null and erroneous apparent magnitudes, and produce a new error column which uses the extended source
# errors primarily. If they are not available then the non-extended source errors are used.
def AlerceClean(df):
    import numpy as np
    if 'magpsf_corr' in df.columns:
        # Remove magpsf_corr null values
        df = df[df['magpsf_corr'].notna()]

        # Remove magpsf_corr values of 100
        df = df[df['magpsf_corr'] != 100]

        # Create new column for error values. Assign the values of sigmapsf_corr_ext. If value is equal to 100
        # assign this the value of sigmapsf_corr if that is not 100. Otherwise leave as null.
        df.loc[(df['sigmapsf_corr_ext'] != 100), ['sigmapsf_corr_revised']] = df['sigmapsf_corr_ext']
        df.loc[(df['sigmapsf_corr_revised'].isnull()) & (df['sigmapsf_corr'] != 100), ['sigmapsf_corr_revised']] = df['sigmapsf_corr']

        # Reset index.
        df =df.reset_index(drop=True)

    else:
        print('magpsf_corr not in df.columns')
        df['magpsf_corr']=np.nan
        df['sigmapsf_corr']=np.nan
        df['sigmapsf_corr_ext']=np.nan
        df['sigmapsf_corr_revised']=np.nan

    return df

def AlerceErr(df,errCol, prob=None, zscore_max=None):
    from scipy import stats
    import numpy as np

    df = df.copy()

    # calculate mean and std of sigma_corr_ext column.
    df_mean = df[errCol].mean()
    df_std = df[errCol].std()
    
    # Use the probability distribution function to calculate the probability of of sigma_corr_ext value being above its value. Then assign
    # this value to the prob columm.
    try:
        pdf = stats.norm(loc=df_mean, scale=df_std)
        df['prob'] = df.apply(lambda x: 1-pdf.cdf(x[errCol]), axis=1)
    except:
        df['prob']=np.nan

    # Calculate z-score and assign to new column.
    df['z_score'] = stats.zscore(df[errCol], axis=0)

    if prob is not None:
        # Drop all data where below a probability threshold of prob.
        prob_val = prob
        df = df[df.prob > prob_val]

    if zscore_max is not None:
        # Filter by z-score based on zscore.
        df = df[df.z_score<zscore_max]
    
    # Reset index.
    #df = df.reset_index(drop=True)
    # print('index not reset')
    return df


# Function to plot Alerce LCs
def Alerce_plot(name, g_band_lc, r_band_lc, pltErr=True, col=None):
    import matplotlib.pyplot as plt

    # Split dataframe into one for each filter.
    df1 = g_band_lc
    df2 = r_band_lc


    if col == None:
        plt.figure(figsize=(15,5))
        if pltErr == True:
            try:
                plt.errorbar(df1['mjd'], df1['magpsf_corr'], df1['sigmapsf_corr_revised'], fmt='.', label='g', color='green')
            except:
                pass
            try:
                plt.errorbar(df2['mjd'], df2['magpsf_corr'], df2['sigmapsf_corr_revised'], fmt='.', label='r', color='red')
            except:
                pass

        elif pltErr==False:
            try:
                plt.plot(df1['mjd'], df1['magpsf_corr'], '.', markersize=20, label='g', color='green')
            except:
                pass
            try:
                plt.plot(df2['mjd'], df2['magpsf_corr'], '.', markersize=20, label='r', color='red')
            except:
                pass
        
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title(name)
        plt.show()
    
    elif col == 'g':
        try:
            plt.figure(figsize=(15,5))
            if pltErr == True:
                plt.errorbar(df1['mjd'], df1['magpsf_corr'], df1['sigmapsf_corr_revised'], fmt='.', label='g', color='green')

            else:
                plt.plot(df1['mjd'], df1['magpsf_corr'], '.', label='g', color='green')
        
            plt.gca().invert_yaxis()
            plt.legend()
            plt.title(name)
            plt.show()

        except:
            pass
    
    elif col == 'r':
        try:
            plt.figure(figsize=(15,5))
            if pltErr == True:
                plt.errorbar(df2['mjd'], df2['magpsf_corr'], df2['sigmapsf_corr_revised'], fmt='.', label='r', color='red')

            else:
                plt.plot(df2['mjd'], df2['magpsf_corr'],'.', label='r', color='red')
            
            plt.gca().invert_yaxis()
            plt.legend()
            plt.title(name)
            plt.show()
        except:
            pass


# Function to plot Lasair LCs
def Lasair_plot(name, df_lc, timeCol, magCol, errCol, pltErr=True, col=None):
    import matplotlib.pyplot as plt

    df1 = df_lc[df_lc['fid']==1]
    df2 = df_lc[df_lc['fid']==2]

    if col == None:
        plt.figure(figsize=(15,5))
        if pltErr == True:
            plt.errorbar(df1[timeCol], df1[magCol], df1[errCol], fmt='.', label='g', color='green')
            plt.errorbar(df2[timeCol], df2[magCol], df2[errCol], fmt='.', label='r', color='red')

        elif pltErr==False:
            plt.plot(df1[timeCol], df1[magCol], '.', label='g', color='green')
            plt.plot(df2[timeCol], df2[magCol], '.', label='r', color='red')
        
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title(name)
        plt.show()
    
    elif col == 'g':
        plt.figure(figsize=(15,5))
        if pltErr == True:
            plt.errorbar(df1[timeCol], df1[magCol], df1[errCol], fmt='.', label='g', color='green')

        else:
            plt.plot(df1[timeCol], df1[magCol], '.', label='g', color='green')
        
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title(name)
        plt.show()
    
    elif col == 'r':
        plt.figure(figsize=(15,5))
        if pltErr == True:
            plt.errorbar(df2[timeCol], df2[magCol], df2[errCol], fmt='.', label='r', color='red')

        else:
            plt.plot(df2[timeCol], df2[magCol],'.', label='r', color='red')
        
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title(name)
        plt.show()



def test():
    print('test')