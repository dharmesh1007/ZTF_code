# Function to download lasair light curves.
def download_lcs(self, object_list, date, folderpath='../alerts_lcs_lasair'):

    """

    date format: YYYY_MM_DD
    usage refers whether the light curves are from training and testing, or they are alerts light curves.

    """

        # Split list of df_ZTF_CVs CVs into batches of 50 - a list of lists - each sublist is of length 50 elements.
    batch_size=50
    object_lists = [object_list[i:i + batch_size] for i in range(0, len(object_list), batch_size)]   # The list of lists


    # Tokens required for Lasair database access via the API: https://lasair-iris.roe.ac.uk/api.
    token = '4607a33defa78fa20bef98791680574b6cc13b23'

    # Path where I'd like to place the folder of light curves.
    if not self.os.path.exists(folderpath):
        self.os.makedirs(folderpath)

    # Save current working directory for later.
    current_dir = self.os.getcwd()

    # Change directory to light curve folder.
    self.os.chdir(folderpath)

    # Create cache folder for light curves
    if not self.os.path.exists('lc_cache'):
        self.os.makedirs('lc_cache')

    # Initiate lasair client instance with light curves cache.
    L = self.lasair.lasair_client(token, cache='lc_cache')

    # Iterate through lists in object_lists.
    for ls in object_lists:
        # Download light curves for the 50 objects in the list.
        c = L.lightcurves(ls)
        # Iterate through objects in list.
        for j, oid in enumerate(ls, start=0):
            print(j, oid)
            # Store light curves in Lasair_LCs folder in json format.
            if not self.os.path.exists(f'{oid}_{date}.json'):
                print(f'downloading light curve {oid}')
                with open(f'{oid}_{date}.json', 'w') as outfile:
                    self.json.dump(c[j], outfile)

    self.os.chdir(current_dir)



class LasairStuff:

    import numpy as np
    import os
    import lasair
    import json
    import math
    import pandas as pd
    import plotly.express as px

    def __init__(self, lightcurve_df):
        self.lightcurve_df = lightcurve_df


    # Code from Roy Williams to convert magpsf into DCmag.
    def dc_mag(self, fid, magpsf,sigmapsf, magnr,sigmagnr, magzpsci, isdiffpos):
        
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
        dc_sigflux =  self.math.sqrt( difference_sigflux**2 + ref_sigflux**2 )

        # apparent mag and its error from fluxes
        if dc_flux > 0.0:
            dc_mag = magzpsci - 2.5 * self.math.log10(dc_flux)
            dc_sigmag = dc_sigflux/dc_flux*1.0857
        else:
            dc_mag = magzpsci
            dc_sigmag = sigmapsf

        return {'dc_mag':dc_mag, 'dc_sigmag':dc_sigmag}


    # Convert lasair light curve difference magnitudes to apparent magnitudes.
    def convert_to_appmag(self):

        df = self.lightcurve_df.copy()

        df['dc_mag'] = df.apply(lambda x: self.dc_mag(x['fid'],
            x['magpsf'], x['sigmapsf'], x['magnr'], x['sigmagnr'], x['magzpsci'],
            x['isdiffpos'])['dc_mag'], axis=1)

        df['dc_sigmag'] = df.apply(lambda x: self.dc_mag(x['fid'], 
            x['magpsf'], x['sigmapsf'], x['magnr'], x['sigmagnr'], x['magzpsci'],
            x['isdiffpos'])['dc_sigmag'], axis=1)
        
        return df


    # Further cuts for refinement.
    def process_lightcurve(self, limit=22, dropnull=True, dropdup=True):

        df = self.lightcurve_df.copy()

        # Obtain apparent magnitudes.
        df = self.convert_to_appmag(df)
        
        # See https://iopscience.iop.org/article/10.1088/1538-3873/aaecbe for limiting magnitudes curve. 
        df = df[df['dc_mag']<limit]

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

            df = self.pd.concat([dfg, dfr], axis=0).sort_values(by='jd', ascending=True).reset_index(drop=True)

        return df


    # Display light curve
    def display_lightcurve(self, limit=22, dropnull=True, dropdup=True, errorCol=None):     

        df = self.lightcurve_df.copy()

        df = self.process_lightcurve(limit=limit, dropnull=dropnull, dropdup=dropdup)

        fig = self.px.scatter(df, x='jd', y='dc_mag', error_y=errorCol, width=1000, height=250, color='fid',
            color_discrete_sequence=["green", "red", "blue", "goldenrod", "magenta"], opacity=0.5,
            labels={"jd":"Julian Date",
                        "dc_mag": "magnitude",
                        "fid": "Filter"})
    

        fig.update_layout(
            yaxis = dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_tickformat='float',
            font=dict(
            size=10,  # Set the font size here
        ))
    
        fig.show()


class FeatureExtractor:
    import feets
    import numpy as np
    from astropy.timeseries import LombScargle
    import pandas as pd

    def __init__(self, lightcurve_df, col_fid='fid', col_date='jd', col_mag='dc_mag', col_magerr='dc_sigmag'):
        self.lightcurve_df = lightcurve_df
        self.col_fid = col_fid
        self.col_date = col_date
        self.col_mag = col_mag
        self.col_magerr = col_magerr
    
    # Functon for aligning time and magnitude for each band.
    def align(self, time, time2, magnitude, magnitude2, error, error2):
        """Synchronizes the light-curves in the two different bands.

        Input
        -----
        time (list or array of timestamps)
        time2 (list or array of timestamps in another band)
        magnitude (list or array of magnitudes)
        magnitude2 (list or array of magnitudes in another band)
        error (list/array of errors for magnitude)
        error2 (list/array of errors for magnitude2)

        Returns
        -------
        aligned_time
        aligned_magnitude
        aligned_magnitude2
        aligned_error
        aligned_error2
        """
        
        error = self.np.zeros(time.shape) if error is None else error
        error2 = self.np.zeros(time2.shape) if error2 is None else error2

        # this asume that the first series is the short one
        # sserie = pd.DataFrame({"mag": magnitude.to_list(), "error": error.to_list()}, index=time)
        # lserie = pd.DataFrame({"mag": magnitude2.to_list(), "error": error2.to_list()}, index=time2)
        
        sserie = self.pd.DataFrame({"mag": magnitude, "error": error}, index=time)
        lserie = self.pd.DataFrame({"mag": magnitude2, "error": error2}, index=time2)

        # if the second serie is logest then revert
        if len(time) > len(time2):
            sserie, lserie = lserie, sserie
        

        # make the merge. The inner join will only pick the rows (jds) where
        # data is present in both bands.
        # For this we need to set the julien date to integer values.
        # While this does lose the fraction of the day. FOr this purpose it is ok.
        merged = sserie.join(lserie, how="inner", rsuffix="2")


        # recreate columns
        new_time = merged.index.values
        new_mag, new_mag2 = merged.mag.values, merged.mag2.values
        new_error, new_error2 = merged.error.values, merged.error2.values

        if len(time) > len(time2):
            new_mag, new_mag2 = new_mag2, new_mag
            new_error, new_error2 = new_error2, new_error

        return new_time, new_mag, new_mag2, new_error, new_error2

    # Function to extract features provided by feets feature extraction package.
    def feature_extraction_feets(self):

        # Split data by filter.
        dfg = self.lightcurve_df[self.lightcurve_df[self.col_fid]==1]
        dfr = self.lightcurve_df[self.lightcurve_df[self.col_fid]==2]
                
        # Columns need as input for feets package.
        timeExct = dfg[self.col_date].values
        time2Exct = dfr[self.col_date].values
        timeInt = dfg[self.col_date].values.astype('int')
        time2Int = dfr[self.col_date].values.astype('int')
        magnitude = dfg[self.col_mag].values
        magnitude2 = dfr[self.col_mag].values
        error = dfg[self.col_magerr].values
        error2 = dfr[self.col_magerr].values
        
        # Features for individual photometric band.
        single_feets_features = self.feets.FeatureSpace(data=['magnitude', 'time', 'error']).features_as_array_

        # We synchronize the data for multiband features.
        atime, amag, amag2, aerror, aerror2 = self.align(timeInt, time2Int, magnitude, magnitude2, error, error2)

        # Multi-band features.
        multiband_feets_features = self.feets.FeatureSpace(data=['aligned_time','aligned_magnitude','aligned_magnitude2',
                                                            'aligned_error', 'aligned_error2']).features_as_array_

        feature_values = {}
        
        for f_g in single_feets_features:
            fs = self.feets.FeatureSpace(only=[f_g])
            try:
                if len(magnitude)>1:
                    feature_g, value_g = fs.extract(time=timeExct, magnitude=magnitude, error=error)
                    feature_values[f'{f_g}_g'] = value_g[0]
                else:
                    print('mag length <2')
                    feature_values[f'{f_g}_g'] = self.np.nan

            except Exception as e:
                print(e)
                feature_values[f'{f_g}_g'] = self.np.nan
        
        for f_r in single_feets_features:
            try:
                if len(magnitude2)>1:
                    feature_r, value_r = fs.extract(time=time2Exct, magnitude=magnitude2, error=error2)
                    feature_values[f'{f_r}_r'] = value_r[0]
                else:
                    print('mag2 length <2')
                    feature_values[f'{f_r}_r'] = self.np.nan
            except Exception as e:
                print(e)
                feature_values[f'{f_r}_r'] = self.np.nan
        
        for f2 in multiband_feets_features:
            fs2 = self.feets.FeatureSpace(only=[f2])
            try:
                features_mb, values_mb = fs2.extract(aligned_time=atime, aligned_magnitude=amag, aligned_magnitude2=amag2,
                                                aligned_error=aerror, aligned_error2=aerror2)
                feature_values[f'{f2}_comb'] = values_mb[0]
            except Exception as e:
                print(e)
                feature_values[f'{f2}_comb'] = self.np.nan

        fs3 = self.feets.FeatureSpace(only=['Color'])
        try:
            features_col, values_col = fs3.extract(magnitude=magnitude, magnitude2=magnitude2)
            feature_values['g-r_color'] = values_col[0]
        except Exception as e:
            print(e)
            feature_values['g-r_color'] = (self.np.nan)

        return feature_values

    # Function to find the deviations from a rolling average.
    def deviations_from_RollAv(self, win_size, delta=1, rollType='median', deviation_type='mag'):
        
        magCol = self.lightcurve_df[self.col_mag]

        # variable for number of instances where condition is met.
        instances = 0

        if rollType == 'median':
            # Threshold value to find value equal to or above. Index of median calculation is set by center.
            rollMed = magCol.rolling(win_size, center=True).median()
            # print(rollMed)
            if deviation_type == 'mag':
                threshold = rollMed + delta
                # print(threshold)
            elif deviation_type == 'std':
                threshold = rollMed + delta * magCol.rolling(win_size, center=True).std()
            # print(threshold)
        
        elif rollType == 'mean':
            # Threshold value to find value equal to or above. Index of median calculation is set by center.
            rollMed = magCol.rolling(win_size, center=True).mean()
            # print(rollMed)
            if deviation_type == 'mag':
                threshold = rollMed + delta
                # print(threshold)
            elif deviation_type == 'std':
                threshold = rollMed + delta * magCol.rolling(win_size, center=True).std()
            # print(threshold)

        # Index of first rolling median value
        indFirst = int(self.np.floor(win_size/2))

        # If the magnitude column size is greater than or equal to the window size we extract other wise return null.
        if len(magCol)>win_size:

            if delta > 0:

                for mag in range(indFirst):
                    # print(mag, 'a', magCol[mag],  threshold[indFirst])
                    if (magCol[mag] >=  threshold[indFirst]):
                        instances+=1 # log instances where  thresholds are met for each epoch

                for mag in range(indFirst, len(magCol)-indFirst):
                    # print(mag,'b', magCol[mag],   threshold[mag])
                    if (magCol[mag] >=  threshold[mag]):
                        instances+=1 # log instances where  thresholds are met for each epoch
                        
                for mag in range(len(magCol)-indFirst, len(magCol)):
                    # print(mag,'c', magCol[mag],   threshold[len(magCol)-1-indFirst])
                    if (magCol[mag] >=  threshold[len(magCol)-1-indFirst]):
                        instances+=1 # log instances where  thresholds are met for each epoch
        
            elif delta < 0:

                for mag in range(indFirst):
                    # print(mag, 'a', magCol[mag],  threshold[indFirst])
                    if (magCol[mag] <=  threshold[indFirst]):
                        instances+=1 # log instances where  thresholds are met for each epoch

                for mag in range(indFirst, len(magCol)-indFirst):
                    # print(mag,'b', magCol[mag],   threshold[mag])
                    if (magCol[mag] <=  threshold[mag]):
                        instances+=1 # log instances where  thresholds are met for each epoch
                        
                for mag in range(len(magCol)-indFirst, len(magCol)):
                    # print(mag,'c', magCol[mag],   threshold[len(magCol)-1-indFirst])
                    if (magCol[mag] <=  threshold[len(magCol)-1-indFirst]):
                        instances+=1 # log instances where  thresholds are met for each epoch
        
        return instances/len(magCol) if len(magCol)>win_size else self.np.nan

    # Number of data points above or below delta mags or stds of the median or mean of the column.
    def deviations_from_average(self, avType='median', deviation_type='mag', delta=1):
        
        magCol = self.lightcurve_df[self.col_mag]

        instances = 0

        if avType == 'median':
            for mag in range(len(magCol)):
                if deviation_type == 'mag':
                    if delta > 0:
                        if magCol[mag] >= (magCol.median() + delta):
                            instances+=1
                    elif delta < 0:
                        if magCol[mag] <= (magCol.median() + delta):
                            instances+=1
                if deviation_type == 'std':
                    if delta > 0:
                        if magCol[mag] >= (magCol.median() + delta * magCol.std()):
                            instances+=1
                    if delta < 0:
                        if magCol[mag] <= (magCol.median() + delta * magCol.std()):
                            instances+=1
        
        elif avType == 'mean':
            for mag in range(len(magCol)):
                if deviation_type == 'mag':
                    if delta > 0:
                        if magCol[mag] >= (magCol.mean() + delta):
                            instances+=1
                    if delta < 0:
                        if magCol[mag] <= (magCol.mean() + delta):
                            instances+=1
                if deviation_type == 'std':
                    if delta > 0:
                        if magCol[mag] >= (magCol.mean() + delta * magCol.std()):
                            instances+=1
                    if delta < 0:
                        if magCol[mag] <= (magCol.mean() + delta * magCol.std()):
                            instances+=1

        if len(magCol)>0:
            return instances/len(magCol)
        else:
            return self.np.nan

    # Median absolute deviation function.
    def mad(self, array):
        a = self.np.abs(array-self.np.median(array))
        b = self.np.median(a)
        return b

    # Extract custom features.
    def feature_extraction_custom(self):

        objFeatures = {}

        for fid, clr in enumerate(['g','r'], start=1):
            # Separate light curves into diferent filters
            df = self.lightcurve_df[self.lightcurve_df[self.col_fid]==fid].reset_index(drop=True)

            # Acquire statistical information
            
            objFeatures[f'{clr}_mean'] = df[self.col_mag].mean()
            objFeatures[f'{clr}_median'] = df[self.col_mag].median()
            objFeatures[f'{clr}_std'] = df[self.col_mag].std()
            objFeatures[f'{clr}_MAD'] = self.mad(df[self.col_mag])
            objFeatures[f'{clr}_min_mag'] = df[self.col_mag].min()
            objFeatures[f'{clr}_max_mag'] = df[self.col_mag].max()
            objFeatures[f'{clr}_n_obs'] = df[self.col_mag].count()
            
            objFeatures[f'{clr}_dif_min_mean'] = abs(objFeatures[f'{clr}_min_mag'] - objFeatures[f'{clr}_mean'])
            objFeatures[f'{clr}_dif_min_median'] = abs(objFeatures[f'{clr}_min_mag'] - objFeatures[f'{clr}_median'])
            objFeatures[f'{clr}_dif_max_mean'] = abs(objFeatures[f'{clr}_max_mag'] - objFeatures[f'{clr}_mean'])
            objFeatures[f'{clr}_dif_max_median'] = abs(objFeatures[f'{clr}_max_mag'] - objFeatures[f'{clr}_median'])
            objFeatures[f'{clr}_temporal_baseline'] = df[self.col_date].max() - df[self.col_date].min()
            objFeatures[f'{clr}_amplitude'] = abs(df[self.col_mag].max() - df[self.col_mag].min())
            
            # Extract number of points brighter than x standard deviations brigher than the rolling median of window size y.
            win_size = [10,20]
            delta = [-1,-2,-3]
            for w in win_size:
                for d in delta:
                    objFeatures[f'{clr}_pnts_leq_rollMedWin{w}{d}std'] = \
                        self.deviations_from_RollAv(
                            win_size=w,
                            delta=d,
                            rollType='median',
                            deviation_type='std'
                            )
            
            # Extract number of points fainter than x standard deviations fainter than the rolling median of window size y.
            win_size = [10,20]
            delta = [1,2,3]
            for w in win_size:
                for d in delta:
                    objFeatures[f'{clr}_pnts_geq_rollMedWin{w}+{d}std'] = \
                        self.deviations_from_RollAv(
                            win_size=w,
                            delta=d,
                            rollType='median',
                            deviation_type='std'
                            )
            
            # Extract number of points brighter than x magnitudes brighter than the rolling median of window size y.
            win_size = [10,20]
            delta = [-1,-3,-5]
            for w in win_size:
                for d in delta:
                    objFeatures[f'{clr}_pnts_leq_rollMedWin{w}{d}mag'] = \
                        self.deviations_from_RollAv(
                            win_size=w,
                            delta=d,
                            rollType='median',
                            deviation_type='mag'
                            )
            
            # Extract number of points fainter than x magnitudes fainter than the rolling median of window size y.
            win_size = [10,20]
            delta = [1,2,3]
            for w in win_size:
                for d in delta:
                    objFeatures[f'{clr}_pnts_geq_rollMedWin{w}+{d}mag'] = \
                        self.deviations_from_RollAv(
                            win_size=w,
                            delta=d,
                            rollType='median',
                            deviation_type='mag'
                            )
            
            # Extract number of points brighter than x standard deviations brigher than median.
            delta = [-1,-2,-3]
            for d in delta:
                objFeatures[f'{clr}_pnts_leq_median{d}std'] = \
                        self.deviations_from_average(
                        avType='median',
                        deviation_type='std',
                        delta=d
                        )
            
            # Extract number of points fainter than x standard deviations fainter than median.
            delta = [1,2,3]
            for d in delta:
                objFeatures[f'{clr}_pnts_geq_median+{d}std'] = \
                        self.deviations_from_average(
                        avType='median',
                        deviation_type='std',
                        delta=d
                        )
            
            # Extract number of points brighter than x magnitudes brigher than median.
            delta = [-1,-3,-5]
            for d in delta:
                objFeatures[f'{clr}_pnts_leq_median{d}mag'] = \
                        self.deviations_from_average(
                        avType='median',
                        deviation_type='mag',
                        delta=d
                        )
            
            # Extract number of points fainter than x magnitudes fainter than median.
            delta = [1,2,3]
            for d in delta:
                objFeatures[f'{clr}_pnts_geq_median+{d}mag'] = \
                        self.deviations_from_average(
                        avType='median',
                        deviation_type='mag',
                        delta=d
                        )
            
            objFeatures[f'{clr}_kurtosis'] = df[self.col_mag].kurtosis()
            objFeatures[f'{clr}_skewness'] = df[self.col_mag].skew()
            
            # Lomb Scargle based features
            if objFeatures[f'{clr}_n_obs']>2:
                ls = self.LombScargle(df[self.col_date], df[self.col_mag])
                frequency, power = ls.autopower()
                objFeatures[f'{clr}_pwr_max'] = power.max()
                objFeatures[f'{clr}_freq_pwr_max'] = frequency[self.np.where(power==power.max())][0]
                objFeatures[f'{clr}_FalseAlarm_prob'] = ls.false_alarm_probability(power.max())
            else:
                objFeatures[f'{clr}_pwr_max'] = self.np.nan
                objFeatures[f'{clr}_freq_pwr_max'] = self.np.nan
                objFeatures[f'{clr}_FalseAlarm_prob'] = self.np.nan

        # Extract colour
        objFeatures['clr_median'] = objFeatures['g_median'] - objFeatures['r_median']
        objFeatures['clr_mean'] = objFeatures['g_mean'] - objFeatures['r_mean']
        objFeatures['clr_minmag'] = objFeatures['g_min_mag'] - objFeatures['r_min_mag']
        objFeatures['clr_maxmag'] = objFeatures['g_max_mag'] - objFeatures['r_max_mag']

        return objFeatures



def load_lc_json(path, object, date):
    import pandas as pd
    import json
    with open(f'{path}/{object}_{date}.json') as json_file:
        df = pd.DataFrame(json.load(json_file))
    return df


