# Function to download Alerce light curves.
import os
import json
from alerce.core import Alerce
import numpy as np
import lasair
import math
import pandas as pd
import plotly.express as px
from scipy import stats



def download_alerce_lc(oid, folderpath):
    alerce = Alerce()
    if not os.path.exists(f'{folderpath}/{oid}.json'):
        print(f'downloading light curve {oid}')
        try:
            with open(f'{folderpath}/{oid}.json', 'w') as outfile:
                json.dump(alerce.query_detections(oid, format="json"), outfile)
        except Exception as e:
            os.remove(f'{folderpath}/{oid}.json')
            print(e)


# Function to download lasair light curves.
def download_lasair_lc(object_list, folderpath, cache):

    # Split list of df_ZTF_CVs CVs into batches of 50 - a list of lists - each sublist is of length 50 elements.
    batch_size=50
    object_lists = [object_list[i:i + batch_size] for i in range(0, len(object_list), batch_size)]   # The list of lists

    # Tokens required for Lasair database access via the API: https://lasair-iris.roe.ac.uk/api.
    token = '4607a33defa78fa20bef98791680574b6cc13b23'

    # Create cache folder for light curves
    if not os.path.exists(cache):
        os.makedirs(cache)

    # Initiate lasair client instance with light curves cache.
    L = lasair.lasair_client(token, cache=cache)

    # Iterate through lists in object_lists.
    for ls in object_lists:
        # Download light curves for the 50 objects in the list.
        c = L.lightcurves(ls)
        # Iterate through objects in list.
        for j, oid in enumerate(ls, start=0):
            print(j, oid)
            # Store light curves in Lasair_LCs folder in json format.
            if not os.path.exists(f'{folderpath}/{oid}.json'):
                print(f'downloading light curve {oid}')
                with open(f'{folderpath}/{oid}.json', 'w') as outfile:
                    json.dump(c[j], outfile)



# Function to remove null and erroneous apparent magnitudes, and produce a new error column which uses the extended source
# errors primarily. If they are not available then the non-extended source errors are used.
def alerce_clean(df, prob=None, zscore_max=None, magerrlim=None):
    if 'magpsf_corr' in df.columns:
        # Remove magpsf_corr null values
        # df = df[df['magpsf_corr'].notna()]

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

    df = df.copy()

    # calculate mean and std of sigma_corr_ext column.
    df_mean = df['sigmapsf_corr_revised'].mean()
    df_std = df['sigmapsf_corr_revised'].std()
    
    # Use the probability distribution function to calculate the probability of of sigma_corr_ext value being above its value. Then assign
    # this value to the prob columm.
    try:
        pdf = stats.norm(loc=df_mean, scale=df_std)
        df['prob'] = df.apply(lambda x: 1-pdf.cdf(x['sigmapsf_corr_revised']), axis=1)
    except:
        df['prob']=np.nan

    # Calculate z-score and assign to new column.
    df['z_score'] = stats.zscore(df['sigmapsf_corr_revised'], axis=0)

    if prob is not None:
        # Drop all data where below a probability threshold of prob.
        df = df[df.prob > prob]

    if zscore_max is not None:
        # Filter by z-score based on zscore.
        df = df[df.z_score<zscore_max]
    
    if magerrlim is not None:
        # Filter by z-score based on zscore.
        df = df[df.sigmapsf_corr_revised < magerrlim]
    
    # Reset index.
    df = df.reset_index(drop=True)

    return df




# Code from Roy Williams to convert magpsf into DCmag.
def dc_mag(fid, magpsf,sigmapsf, magnr,sigmagnr, magzpsci, isdiffpos):
    
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
    
    # # zero points. Looks like they are fixed.
    # ref_zps = {1:26.325, 2:26.275, 3:25.660}
    # magzpref = ref_zps[fid]

    # # reference flux and its error
    # magdiff = magzpref - magnr
    # if magdiff > 12.0:
    #     magdiff = 12.0
    # ref_flux = 10**( 0.4* ( magdiff) )
    # ref_sigflux = (sigmagnr/1.0857)*ref_flux

    # # difference flux and its error
    # if magzpsci == 0.0: magzpsci = magzpref
    # magdiff = magzpsci - magpsf
    # if magdiff > 12.0:
    #     magdiff = 12.0
    # difference_flux = 10**( 0.4* ( magdiff) )
    # difference_sigflux = (sigmapsf/1.0857)*difference_flux

    # # add or subract difference flux based on isdiffpos
    # if isdiffpos == 't': dc_flux = ref_flux + difference_flux
    # else:                dc_flux = ref_flux - difference_flux

    # # assumes errors are independent. Maybe too conservative.
    # dc_sigflux = math.sqrt( difference_sigflux**2 + ref_sigflux**2 )

    # # apparent mag and its error from fluxes
    # if dc_flux > 0.0:
    #     dc_mag = magzpsci - 2.5 * math.log10(dc_flux)
    #     dc_sigmag = dc_sigflux/dc_flux*1.0857
    # else:
    #     dc_mag = magzpsci
    #     dc_sigmag = sigmapsf

    dc_mag = magpsf
    dc_sigmag = sigmapsf
    
    if isdiffpos=='t':
        dc_mag = -2.5*math.log10(10**(-0.4*magnr)+10**(-0.4*magpsf))
        dc_sigmag_num = (10**(-0.8*magpsf)*sigmapsf**2)**0.5
        dc_sigmag_den = 10**(-0.4*magnr)+10**(-0.4*magpsf)
        dc_sigmag = dc_sigmag_num/dc_sigmag_den

    elif isdiffpos=='f':
        if magpsf<=magnr:
            # print('illegal', f'magpsf:{magpsf}', f'magnr:{magnr}')
            dc_mag = np.nan
            dc_sigmag = np.nan
        else:
            dc_mag = -2.5*math.log10(10**(-0.4*magnr)-10**(-0.4*magpsf))
            dc_sigmag_num = (10**(-0.8*magpsf)*sigmapsf**2)**0.5
            dc_sigmag_den = 10**(-0.4*magnr)-10**(-0.4*magpsf)
            dc_sigmag = dc_sigmag_num/dc_sigmag_den

    return {'dc_mag':dc_mag, 'dc_sigmag':dc_sigmag}


# Convert lasair light curve difference magnitudes to apparent magnitudes.
def convert_to_appmag(dataframe):

    df = dataframe.copy()

    df['dc_mag'] = df.apply(lambda x: dc_mag(x['fid'],
        x['magpsf'], x['sigmapsf'], x['magnr'], x['sigmagnr'], x['magzpsci'],
        x['isdiffpos'])['dc_mag'], axis=1)

    df['dc_sigmag'] = df.apply(lambda x: dc_mag(x['fid'], 
        x['magpsf'], x['sigmapsf'], x['magnr'], x['sigmagnr'], x['magzpsci'],
        x['isdiffpos'])['dc_sigmag'], axis=1)
    
    return df


# Further cuts for refinement.
def lasair_clean(dataframe, limit=None, dropnull=True, dropdup=True, prob=None, zscore_max=None, magerrlim=None):

    df = dataframe.copy()

    # Obtain apparent magnitudes.
    df = convert_to_appmag(df)
    
    # Remove rows with null magnitude values.
    if dropnull:
        df = df[~df['dc_mag'].isnull()]

    # Split data by filter.
    dfg = df[df['fid']==1]
    dfr = df[df['fid']==2]

    if dropdup:
        # Drop duplicates for Julien date.
        dfg = dfg.drop_duplicates(subset='jd', keep='first', inplace=False, ignore_index=True)
        dfr = dfr.drop_duplicates(subset='jd', keep='first', inplace=False, ignore_index=True)

        df = pd.concat([dfg, dfr], axis=0).sort_values(by='jd', ascending=True).reset_index(drop=True)

    # calculate mean and std of sigma_corr_ext column.
    df_mean = df['dc_sigmag'].mean()
    df_std = df['dc_sigmag'].std()
    
    # Use the probability distribution function to calculate the probability of of sigma_corr_ext value being above its value. Then assign
    # this value to the prob columm.
    try:
        pdf = stats.norm(loc=df_mean, scale=df_std)
        df['prob'] = df.apply(lambda x: 1-pdf.cdf(x['dc_sigmag']), axis=1)
    except:
        df['prob']=np.nan

    # Calculate z-score and assign to new column.
    df['z_score'] = stats.zscore(df['dc_sigmag'], axis=0)

    if prob is not None:
        # Drop all data where below a probability threshold of prob.
        df = df[df.prob > prob]

    if zscore_max is not None:
        # Filter by z-score based on zscore.
        df = df[df.z_score<zscore_max]
    
    # Limit by magnitude error.
    if magerrlim is not None:
        # Filter by dc_sigmag.
        df = df[df.dc_sigmag < magerrlim]
    
    # See https://iopscience.iop.org/article/10.1088/1538-3873/aaecbe for limiting magnitudes curve.
    if limit is not None:
        # Filter by magnitude.
        df = df[df['dc_mag']<limit]
    
    # Reset index.
    df = df.reset_index(drop=True)
    return df



# Display light curve
def display_lightcurve(lc_df, x, y, errorCol=None, autorange='reversed'):     
    # autorange = 'reversed' or True

    df = lc_df.copy()

    df['fid'] = df['fid'].astype(str)

    fig = px.scatter(df, x=x, y=y, error_y=errorCol, width=1000, height=250, color='fid',
        color_discrete_map={'1':'green','2':'red'}, opacity=0.75,
        labels={"jd":"Julian Date",
                "dc_mag": "apparant magnitude",
                "fid": "Filter"})
    
    fig.update_layout(
        yaxis = dict(autorange=autorange),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_tickformat='float',
        font=dict(
        size=10,  # Set the font size here
    ))
    
    fig.show()




def load_lasair_lc(oid, path):
    with open(f'{path}/{oid}.json') as json_file:
        df = pd.json_normalize(json.load(json_file), record_path=['candidates'])
    return df




