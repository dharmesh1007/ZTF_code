import lasair
import pandas as pd
import datetime
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky
import numpy as np
from lcfunctions import download_lasair_lc, load_lasair_lc, lasair_clean, display_lightcurve

# Function to get alerts from Lasair based on the query below
def get_lasair_alerts_q1(day_first=1, day_last=0):

    # day first is the number of days ago to start the query
    # day last is the number of days ago to end the query

    token = '4607a33defa78fa20bef98791680574b6cc13b23'
    L = lasair.lasair_client(token, cache=None)

    # SELECT TABLES.
    tables = 'objects,sherlock_classifications,crossmatch_tns'

    # SELECT COLUMNS FROM TABLE.
    selected    = """
        objects.objectId,
        objects.ramean,
        objects.decmean,
        objects.gmag, 
        objects.rmag,
        objects.maggmean,
        objects.magrmean,
        objects.g_minus_r,
        (objects.maggmean - objects.magrmean) as clr_mean,
        objects.dmdt_g,
        objects.dmdt_r,
        objects.distpsnr1, 
        objects.sgmag1, 
        (objects.gmag - objects.sgmag1) as brightening_g,
        (objects.rmag - objects.srmag1) as brightening_r, 
        objects.sgscore1,
        JDNOW()-objects.jdmax as last_alert, 
        sherlock_classifications.classification,
        sherlock_classifications.classificationReliability,
        sherlock_classifications.catalogue_table_name,
        sherlock_classifications.separationArcsec,
        sherlock_classifications.physical_separation_kpc,
        sherlock_classifications.direct_distance,
        sherlock_classifications.distance,
        sherlock_classifications.z,
        sherlock_classifications.photoZ,
        sherlock_classifications.major_axis_arcsec,
        crossmatch_tns.tns_prefix,
        sherlock_classifications.association_type,
        sherlock_classifications.description
        """

    # QUERIES.
    # query that omits supernovae, variable stars, asteroids, AGN, and bright stars
    query  = f"""
        (objects.jdmax BETWEEN JDNOW()-{day_first} AND JDNOW()-{day_last}) AND NOT
        (sherlock_classifications.classification ='AGN' AND sherlock_classifications.classificationReliability = 1) AND NOT
        (sherlock_classifications.classification ='NT' AND sherlock_classifications.classificationReliability = 1) AND NOT
        (sherlock_classifications.classification='SN' AND sherlock_classifications.separationArcsec <= 1.5) AND NOT
        crossmatch_tns.tns_prefix IN ('SN')
        """ 
    # GET RESULTS OF QUERY.
    c1 = L.query(selected, tables, query, limit=1000000)

    # Convert to pandas dataframe
    query_result = pd.DataFrame(c1).rename(columns={'ramean':'ra','decmean':'dec'})

    # Remove sources that remain that are within 1.5 arcsec of a galaxy
    query_result = query_result[~((query_result['separationArcsec']<=1.5) & (query_result['description'].str.contains('galaxy')) & (query_result['description'].str.contains('synonymous')))].reset_index(drop=True)

    # Drop duplicates
    alerts_df = query_result.drop_duplicates(subset=['objectId'], keep='first').reset_index(drop=True)
    # print(f'Number of alerts: {len(alerts_df)}')

    # Get the data and time in the above format minus day_last days in the format ddmmmyy_hh:00:00
    zeros = '00'
    first_day = (datetime.datetime.now()-datetime.timedelta(days=day_first)).strftime(f"%d%m%y_%H:{zeros}:{zeros}")
    last_day = (datetime.datetime.now()-datetime.timedelta(days=day_last)).strftime(f"%d%m%y_%H:{zeros}:{zeros}")
    
    # Save the results of the query to a csv file
    alerts_df.to_csv(f'../results/alerts/alerts_query1_{first_day}_to_{last_day}.csv', index=False)
    print(f'Number of alerts from initial query: {alerts_df.shape[0]}')

    # Select number of rows to display
    pd.options.display.max_rows=10
    # Select number of columns to display
    pd.options.display.max_columns=None

    # Display dataframe
    print(f"query:\n{alerts_df['classification'].value_counts()}")

    return alerts_df

# Function to process the aavso data of pulsating variables
def aavsopulsatingprocess(filepath, savepath):
    # Read in AAVSO data
    aavso_df = pd.read_csv(filepath, keep_default_na=False)
    aavso_df['ra'] = aavso_df.apply(lambda x: SkyCoord(f'{x["Coords"]}', unit=(u.hourangle, u.deg), equinox='J2000').ra.deg, axis=1)
    aavso_df['dec'] = aavso_df.apply(lambda x: SkyCoord(f'{x["Coords"]}', unit=(u.hourangle, u.deg), equinox='J2000').dec.deg, axis=1)

    # Remove objects with uncertain classification and or have several possible types (:, |). 
    aavso_df = aavso_df[(aavso_df['Type'].str.contains(':', regex=True)==False)]
    aavso_df = aavso_df[(aavso_df['Type'].str.contains('\|', regex=True)==False)]
    aavso_df = aavso_df.reset_index(drop=True)

    # Remove objects with uncertain classification and or have several possible types (:, |).
    # Also remove those with multiple labels as they tend to include some CV types.
    # Remove those with ZZ in their name.
    aavso_df = aavso_df[(aavso_df['Type'].str.contains(':', regex=True)==False)]
    aavso_df = aavso_df[(aavso_df['Type'].str.contains('\|', regex=True)==False)]
    aavso_df = aavso_df[(aavso_df['Type'].str.contains('\+', regex=True)==False)]
    aavso_df = aavso_df[(aavso_df['Type'].str.contains('ZZ', regex=True)==False)]
    typedrop = ['SXPHE','SXPHE(B)','V361HYA','V1093HER','L','LB','LC',]
    aavso_df = aavso_df[~(aavso_df['Type'].isin(typedrop))]
    aavso_df = aavso_df.reset_index(drop=True)

    aavso_df.to_csv(savepath, index=False)
    # print(f'Number of pulsating variables in AAVSO: {len(aavso_df)}')
    
    return aavso_df

# Function to filter out the aavso pulsating variables from the alerts
def filter_pulsating(alerts_df, aavso_puls_df):

    # Create astropy skycoord objects for each catalogue
    coords_alerts = SkyCoord(ra=alerts_df['ra'].values*u.degree, dec=alerts_df['dec'].values*u.degree)
    coords_aavso = SkyCoord(ra=aavso_puls_df['ra'].values*u.degree, dec=aavso_puls_df['dec'].values*u.degree)

    # Perform coordinate match
    idx, d2d, d3d = match_coordinates_sky(coords_alerts, coords_aavso)

    # Create a pandas dataframe with the results
    matches = pd.DataFrame({'idx_aavso':idx, 'd2d':d2d.arcsecond})
    matches = matches[matches['d2d']<=2] # Only select matches with a separation of less than 5 arcseconds
    # print(f'Number of matches / sources removed: {len(matches)}')

    # Create a new dataframe with the matches
    alerts_aavso = pd.concat([alerts_df, matches], axis=1)
    alerts_aavso = alerts_aavso.merge(aavso_puls_df, left_on='idx_aavso', right_index=True, how='left')

    # Limit to those with a separation of less than 2 arcseconds
    alerts_aavso_matched = alerts_aavso[alerts_aavso['d2d']<=2].reset_index(drop=True)

    # Drop from alerts_df those that have a match
    alerts_df_new = alerts_df[~(alerts_df['objectId'].isin(alerts_aavso_matched['objectId'].to_list()))].reset_index(drop=True)
    print(f'Number of alerts after removing pulsating variables: {len(alerts_df_new)}')

    return alerts_df_new

# Function to filter the alerts based on colour information
def clr_filter(alerts_df, folderpath):

        # Get colours for each light curve
        from featureextractor import FeatureExtractor

        clr_epoch_mean = np.zeros(len(alerts_df))
        clr_epoch_median = np.zeros(len(alerts_df))
        clr_epoch_bright = np.zeros(len(alerts_df))
        clr_epoch_faint = np.zeros(len(alerts_df))
        clr_mean = np.zeros(len(alerts_df))
        clr_median = np.zeros(len(alerts_df))
        npts_g = np.zeros(len(alerts_df))
        npts_r = np.zeros(len(alerts_df))

        objlist = alerts_df['objectId'].to_list()
        folderpath = '../lightcurves_alerts/'
        for count, obj in enumerate(objlist):
                print(count, obj)
                # Load and process lasair light curve
                lc_test = load_lasair_lc(oid=obj, path=folderpath)
                lc_appmag_test = lasair_clean(lc_test, limit=25, magerrlim=1)

                # Create a copy of the light curve
                lc = lc_appmag_test.copy()

                df_g = lc[lc['fid']==1]
                df_r = lc[lc['fid']==2]

                npts_g_x = len(df_g)
                npts_r_x = len(df_r)

                clr_mean_x = df_g['dc_mag'].mean() - df_r['dc_mag'].mean()
                clr_median_x = df_g['dc_mag'].median() - df_r['dc_mag'].median()

                fe_clr = FeatureExtractor(lc)
                clr_epoch_mean[count] = fe_clr.clr(lc)[0]
                clr_epoch_median[count] = fe_clr.clr(lc)[1]
                clr_epoch_bright[count] = fe_clr.clr(lc)[3]
                clr_epoch_faint[count] = fe_clr.clr(lc)[4]
                clr_mean[count] = clr_mean_x
                clr_median[count] = clr_median_x
                npts_g[count] = npts_g_x
                npts_r[count] = npts_r_x


        # Add colour features to dataframe
        alerts_df['clr_epoch_mean'] = clr_epoch_mean
        alerts_df['clr_epoch_median'] = clr_epoch_median
        alerts_df['clr_epoch_bright'] = clr_epoch_bright
        alerts_df['clr_epoch_faint'] = clr_epoch_faint
        alerts_df['clr_mean_new'] = clr_mean
        alerts_df['clr_median_new'] = clr_median
        alerts_df['npts_g'] = npts_g
        alerts_df['npts_r'] = npts_r

        overepochmean = alerts_df[alerts_df['clr_epoch_mean']>0.7]['objectId'].to_list()
        overepochmedian = alerts_df[alerts_df['clr_epoch_median']>0.7]['objectId'].to_list()
        overepochbright = alerts_df[alerts_df['clr_epoch_bright']>0.7]['objectId'].to_list()
        overepochfaint = alerts_df[alerts_df['clr_epoch_faint']>0.7]['objectId'].to_list()
        overmean = alerts_df[alerts_df['clr_mean_new']>0.7]['objectId'].to_list()
        overmedian = alerts_df[alerts_df['clr_median_new']>0.7]['objectId'].to_list()
        overmean_lasair = alerts_df[alerts_df['clr_mean']>0.7]['objectId'].to_list()

        overinall = alerts_df[(alerts_df['clr_epoch_mean']>0.7) &
                                (alerts_df['clr_epoch_median']>0.7) &
                                (alerts_df['clr_epoch_bright']>0.7) &
                                (alerts_df['clr_epoch_faint']>0.7) &
                                (alerts_df['clr_mean_new']>0.7)
                                ]['objectId'].to_list()

        # print(f'Number of objects with clr_epoch_mean > 0.7: {len(overepochmean)}')
        # print(f'Number of objects with clr_epoch_median > 0.7: {len(overepochmedian)}')
        # print(f'Number of objects with clr_epoch_bright > 0.7: {len(overepochbright)}')
        # print(f'Number of objects with clr_epoch_faint > 0.7: {len(overepochfaint)}')
        # print(f'Number of objects with clr_mean_new > 0.7: {len(overmean)}')
        # print(f'Number of objects with clr_median_new > 0.7: {len(overmedian)}')
        # print(f'Number of objects with clr_mean > 0.7: {len(overmean_lasair)}')
        # print(f'Number of objects with clrall > 0.7: {len(overinall)}')

        alerts_df_clrcut = alerts_df[~alerts_df['objectId'].isin(overinall)].reset_index(drop=True)
        alerts_df_clrcut_opposite = alerts_df[alerts_df['objectId'].isin(overinall)].reset_index(drop=True)
        print(f'Number of alerts after clr cuts: {len(alerts_df_clrcut)}')

        return alerts_df_clrcut

# Filter out supernova candidates from alerts
def sn_filtering(alerts_df):

    alerts_df_sncut = alerts_df.copy()
    # Replace 0 values with NaN in physical_separation_kpc, separationArcsec and major_axis_arcsec columns
    alerts_df_sncut.loc[alerts_df_sncut['physical_separation_kpc']==0, 'physical_separation_kpc'] = np.nan
    alerts_df_sncut.loc[alerts_df_sncut['separationArcsec']==0, 'separationArcsec'] = np.nan
    alerts_df_sncut.loc[alerts_df_sncut['major_axis_arcsec']==0, 'major_axis_arcsec'] = np.nan

    pd.options.display.max_rows = 10
    alerts_df_sncut_drop = alerts_df_sncut[(alerts_df_sncut['separationArcsec']<alerts_df_sncut['major_axis_arcsec']) &
                                    (alerts_df_sncut['classification']=='SN') &
                                    (alerts_df_sncut['sgscore1']<=0.5)].reset_index(drop=True)

    alerts_df_sncut_drop_list = alerts_df_sncut_drop['objectId'].to_list()
    alerts_df_sncut = alerts_df_sncut[~alerts_df_sncut['objectId'].isin(alerts_df_sncut_drop_list)].reset_index(drop=True)

    alerts_df_sncut_drop2 = alerts_df_sncut[(alerts_df_sncut['classification']=='SN') &
                                            (alerts_df_sncut['classificationReliability'].isin([1,2])) &
                                            (alerts_df_sncut['physical_separation_kpc']>0) &
                                            (alerts_df_sncut['sgscore1']<=0.5)].reset_index(drop=True)

    alerts_df_sncut_drop2_list = alerts_df_sncut_drop2['objectId'].to_list()
    alerts_df_sncut = alerts_df_sncut[~alerts_df_sncut['objectId'].isin(alerts_df_sncut_drop2_list)].reset_index(drop=True)

    alerts_df_sncut_drop3 = alerts_df_sncut[(alerts_df_sncut['classification']=='SN') &
                                            (alerts_df_sncut['classificationReliability'].isin([1,2])) &
                                            (alerts_df_sncut['sgscore1']<0.15)].reset_index(drop=True)

    alerts_df_sncut_drop3_list = alerts_df_sncut_drop3['objectId'].to_list()
    alerts_df_sncut = alerts_df_sncut[~alerts_df_sncut['objectId'].isin(alerts_df_sncut_drop3_list)].reset_index(drop=True)

    # alerts_df_sncut_drop4 = alerts_df_sncut[(alerts_df_sncut['sgscore1']<0.25)].reset_index(drop=True)
    # alerts_df_sncut = alerts_df_sncut[~alerts_df_sncut['objectId'].isin(alerts_df_sncut_drop4['objectId'].to_list())].reset_index(drop=True)

    print(f'Number of alerts after SN cuts: {len(alerts_df_sncut)}')

    return alerts_df_sncut

# Crossmatch alerts with AAVSO CVs
def xmatchcvs(alerts_df, aavsocvs):
    # Create astropy skycoord objects for each catalogue
    alerts_coords = SkyCoord(ra=alerts_df['ra'].values*u.degree, dec=alerts_df['dec'].values*u.degree)
    cv_coords = SkyCoord(ra=aavsocvs['ra'].values*u.degree, dec=aavsocvs['dec'].values*u.degree)
    # Perform coordinate match
    idx_aavso, d2d_preds, d3d_preds = match_coordinates_sky(alerts_coords, cv_coords)
    # Create a pandas dataframe with the results
    matches = pd.DataFrame({'idx_aavso':idx_aavso, 'd2d':d2d_preds.arcsecond})
    # Create a new dataframe with the matches
    alerts_aavso = pd.concat([alerts_df, matches], axis=1)
    alerts_aavso = alerts_aavso.merge(aavsocvs, left_on='idx_aavso', right_index=True, how='left')
    # If d2d is greater than 2 arcseconds, then there is no match, so set certain columns to NaN
    alerts_aavso.loc[alerts_aavso['d2d']>2, ['Name', 'Const', 'Type', 'Period']] = ''
    # Drop columns
    alerts_aavso = alerts_aavso.drop(columns=['idx_aavso', 'd2d', 'AUID', 'Coords', 'Mag', 'ra_y', 'dec_y', 'Const', 'Period'])
    # Rename columns
    alerts_aavso = alerts_aavso.rename(columns={'ra_x':'ra', 'dec_x':'dec', 'Name':'aavso_name', 'Type':'aavso_type'})

    return alerts_aavso

# Process the coordinates of the aavso data such that they are in decimal degrees and useable
def coords_process(filepath, savepath):
    aavso_df = pd.read_csv(filepath, keep_default_na=False)
    aavso_df['ra'] = aavso_df.apply(lambda x: SkyCoord(f'{x["Coords"]}', unit=(u.hourangle, u.deg), equinox='J2000').ra.deg, axis=1)
    aavso_df['dec'] = aavso_df.apply(lambda x: SkyCoord(f'{x["Coords"]}', unit=(u.hourangle, u.deg), equinox='J2000').dec.deg, axis=1)
    aavso_df.to_csv(savepath, index=False)
    return aavso_df
