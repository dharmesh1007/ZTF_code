import numpy as np
import pandas as pd
import feets
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.stats import linregress
import time
from lcfunctions import load_lasair_lc, lasair_clean
from outlier import outlier_thresholds_skewed, apply_thresholds
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor:
    def __init__(self, lc):
        self.lc = lc

    # Functon for aligning time and magnitude for each band.
    def align(self, time, time2, magnitude, magnitude2, error, error2):
        """Synchronizes the light-curves in the two different bands.
        Returns
        -------
        aligned_time
        aligned_magnitude
        aligned_magnitude2
        aligned_error
        aligned_error2
        """
        error = np.zeros(time.shape) if error is None else error
        error2 = np.zeros(time2.shape) if error2 is None else error2

        # this asume that the first series is the short one
        # sserie = pd.DataFrame({"mag": magnitude.to_list(), "error": error.to_list()}, index=time)
        # lserie = pd.DataFrame({"mag": magnitude2.to_list(), "error": error2.to_list()}, index=time2)
        
        sserie = pd.DataFrame({"mag": magnitude, "error": error}, index=time)
        lserie = pd.DataFrame({"mag": magnitude2, "error": error2}, index=time2)


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

        # print(new_time, new_mag, new_mag2, new_error, new_error2)
        return new_time, new_mag, new_mag2, new_error, new_error2

    # Number of data points above or below delta mags or standard deviations of a rolling median or mean.
    def deviations_from_RollAv(self, magCol, win_size, delta=1, rollType='median', deviation_type='mag'):
        magCol = magCol.reset_index(drop=True)
        # variable for number of instances where condition is met.
        instances = 0

        if rollType == 'median':
            # Threshold value to find value equal to or above. Index of median calculation is set by center.
            rollMed = magCol.rolling(win_size, center=True).median()
            # print(rollMed.max())
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
        indFirst = int(np.floor(win_size/2))

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
        
        return instances if len(magCol)>win_size else np.nan


    # Number of data points above or below delta mags or stds of the median or mean of the column.
    def deviations_from_average(self, magCol, avType='median', deviation_type='mag', delta=1):
        magCol = magCol.reset_index(drop=True)
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
            return instances
        else:
            return np.nan

    # Median absolute deviation function.
    def mad(self, array):
        a = np.abs(array-np.median(array))
        b = np.median(a)
        return b

    # Method for finding standstills in Z_Cam light curves
    def standstill_finder(self, lc, pnt_threshold=20, window_size=10):
        # Reverse light curve
        lc_rev = lc.copy()
        # subtract median magnitude from each filter and multiply by -1 to flip the light curve
        lc_rev['dc_mag'] = (lc_rev['dc_mag'] - lc_rev['dc_mag'].median())*(-1)

        # # Convert jd to ingeter
        # lc['jd'] = lc['jd'].astype(int)
        # # for a given filter, if the same jd is repeated, keep the first one
        # lc = lc.drop_duplicates(subset=['jd', 'fid'], keep='first')

        # Rolling standard deviation for light curves with equal to or more than pnt_threshold
        if len(lc_rev) >= pnt_threshold:
            # Rolling standard deviation for window size
            roll_std = lc_rev['dc_mag'].rolling(window_size).std()
            # Rolling mean for window size
            roll_mean = lc_rev['dc_mag'].rolling(window_size).mean()
            # Maximum and minimum of light curve
            lc_max = lc_rev['dc_mag'].max()
            lc_min = lc_rev['dc_mag'].min()
            # index of minimum of rolling standard deviation
            min_roll_std = roll_std.idxmin()
            # mean of the window with the minimum rolling standard deviation
            min_roll_std_mean = roll_mean[min_roll_std]
            # fraction of the maximum light curve amplitude that min_roll_std_mean is, i.e., location relative to the maximum of the standstill.
            standstill_level = (min_roll_std_mean-lc_min)/(lc_max-lc_min)
            # Ratio of max and min of rolling std
            rollstd_maxminratio = roll_std.max()/roll_std.min()
            # print(f'Rolling std max: {roll_std.max()}',
            #     f'\nRolling std min: {roll_std.min()}',
            #     f'\nRatio of max and min of rolling std: {roll_std.max()/roll_std.min()}',
            #     f'\nMean brightness of window with minimum rolling std: {min_roll_std_mean}'
            #     f'\nStandstill level: {standstill_level}')
            
            return rollstd_maxminratio, standstill_level
        else:
            return np.nan, np.nan

    def peak_finder(self, lc, height=None, threshold=None, distance=None, prominence=(2,5), 
                        width=None, wlen=None, rel_height=0.5, plateau_size=None):
        
        # Reverse light curve
        lc_rev = lc.copy()
        # subtract median magnitude from each filter and multiply by -1 to flip the light curve
        lc_rev['dc_mag'] = (lc_rev['dc_mag'] - lc_rev['dc_mag'].median())*(-1)

        # Find peaks in the light curve
        peaks, properties = find_peaks(lc_rev['dc_mag'], 
                                    height=height, 
                                    threshold=threshold, 
                                    distance=distance, 
                                    prominence=prominence, 
                                    width=width, 
                                    wlen=wlen,
                                    rel_height=rel_height, 
                                    plateau_size=plateau_size)


        if len(peaks)>0:
            # Get the times of the peaks and the times of the left and right edges of the peaks
            peak_times = lc_rev.iloc[peaks]['jd'].values
            n_peaks = len(peak_times)
            times_left_bases = lc_rev.iloc[properties['left_bases']]['jd'].values
            times_right_bases = lc_rev.iloc[properties['right_bases']]['jd'].values
            max_prominence = properties['prominences'].max()
            rise_rate = np.mean(properties['prominences']/(peak_times-times_left_bases))
            decline_rate = np.mean(properties['prominences']/(times_right_bases-peak_times))
        
        elif len(peaks)==0:
            try:
                linreg = linregress(lc['jd'], lc['dc_mag'])
                if (linreg.pvalue < 0.05) & (linreg.slope < 0):
                    n_peaks = 0
                    rise_rate = 0
                    decline_rate = linreg.slope*(-1)
                    max_prominence = 0
                else:
                    n_peaks = 0
                    rise_rate = 0
                    decline_rate = 0
                    max_prominence = 0
                
            except:
                n_peaks = 0
                rise_rate = 0
                decline_rate = 0
                max_prominence = 0


        # print(f'Number of peaks: {n_peaks}',
        #         f'\nRise rate: {rise_rate}',
        #         f'\nDecline rate: {decline_rate}',
        #         f'\nMax prominence: {max_prominence}')

        ## Plot location of peaks
        # import matplotlib.pyplot as plt 
        # plt.figure(figsize=(12,2.5))
        # plt.scatter(lc_rev['jd'], lc_rev['dc_mag'],s=10)
        # plt.scatter(lc_rev.iloc[peaks]['jd'], lc_rev.iloc[peaks]['dc_mag'], s=10,c='r')
        # plt.show()

        return n_peaks, rise_rate, decline_rate, max_prominence

    def clr(self, lc):

        df = lc.copy()
        # Dates as integers
        df['jd'] = df['jd'].astype(int)
        # Split into different filters
        df_g = df[df['fid']==1]
        df_r = df[df['fid']==2]

        # Drop duplicated dates from each band
        df_g = df_g.drop_duplicates(subset=['jd'], keep='first')
        df_r = df_r.drop_duplicates(subset=['jd'], keep='first')

        # Identify dates that are in both bands
        df_g['in_both'] = df_g['jd'].isin(df_r['jd'])
        df_r['in_both'] = df_r['jd'].isin(df_g['jd'])

        # Drop dates that are not in both bands
        df_g_inboth = df_g[df_g['in_both']==True].reset_index(drop=True)
        df_r_inboth = df_r[df_r['in_both']==True].reset_index(drop=True)

        # If sufficient data points in both bands, calculate the colour
        if (len(df_g_inboth)>0) & (len(df_r_inboth)>0):
            # Calculate the difference between the two bands
            clr_per_epoch = df_g_inboth['dc_mag'] - df_r_inboth['dc_mag']
            
            # Calculate the mean difference
            clr_mean = clr_per_epoch.mean()
            clr_median = clr_per_epoch.median()
            clr_std = clr_per_epoch.std()
            clr_bright = df_g_inboth['dc_mag'].min() - df_r_inboth['dc_mag'].min()
            clr_faint = df_g_inboth['dc_mag'].max() - df_r_inboth['dc_mag'].max()

            # print(f'CLR mean: {clr_mean}\nCLR median: {clr_median}\nCLR std: {clr_std}')
        
        else:
            clr_mean = df_g['dc_mag'].mean() - df_r['dc_mag'].mean()
            clr_median = df_g['dc_mag'].median() - df_r['dc_mag'].median()
            clr_std = np.nan
            clr_bright = df_g['dc_mag'].min() - df_r['dc_mag'].min()
            clr_faint = df_g['dc_mag'].max() - df_r['dc_mag'].max()

            # print(f'CLR mean: {clr_mean}\nCLR median: {clr_median}\nCLR std: {clr_std}')

        return clr_mean, clr_median, clr_std, clr_bright, clr_faint

    # Missing data imputation function
    def impute_column(self, dataframe, col_orig, col_replace, reversed=True):
        df = dataframe.copy()
        for col1, col2 in zip(col_orig, col_replace):
            # If value in col_orig is missing, replace with value in col_replace if not missing, else replace with nan
            df[col1] = np.where(df[col1].isnull(), df[col2], df[col1])
            if reversed == True:
                # Do the same the other way around
                df[col2] = np.where(df[col2].isnull(), df[col1], df[col2])
        return df
    
    
    # Function to calculate features from feets package.
    def extract_feets(self, custom_remove=None, 
                      timeCol='jd', magCol='dc_mag', errCol='dc_sigmag', fieldCol='fid'):
        
        df = self.lc.copy()
        # Feature lists
        single_feets_features = feets.FeatureSpace(data=['magnitude', 'time', 'error']).features_as_array_

        multi_feets_features = feets.FeatureSpace(data=['aligned_time', 'aligned_magnitude', 'aligned_magnitude2', 
                                                        'aligned_error', 'aligned_error2']).features_as_array_
        
        # Features to remove from fs.extract due to insufficient data.
        feat_remove = {'1':['Autocor_length', 'Beyond1Std', 'Con', 'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 
                            'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80', 
                            'Freq1_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_1', 'Freq1_harmonics_amplitude_2', 
                            'Freq1_harmonics_amplitude_3', 'Freq1_harmonics_rel_phase_0', 'Freq1_harmonics_rel_phase_1', 
                            'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3', 'Freq2_harmonics_amplitude_0', 
                            'Freq2_harmonics_amplitude_1', 'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3', 
                            'Freq2_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_1', 'Freq2_harmonics_rel_phase_2', 
                            'Freq2_harmonics_rel_phase_3', 'Freq3_harmonics_amplitude_0', 'Freq3_harmonics_amplitude_1', 
                            'Freq3_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_3', 'Freq3_harmonics_rel_phase_0', 
                            'Freq3_harmonics_rel_phase_1', 'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3', 
                            'MaxSlope', 'PercentDifferenceFluxPercentile', 'PeriodLS', 'Period_fit', 'Psi_CS', 'Psi_eta', 
                            'SmallKurtosis', 'StetsonK', 'StructureFunction_index_21', 'StructureFunction_index_31', 
                            'StructureFunction_index_32'],
                        '2':['Con', 'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 
                            'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80', 
                            'Freq1_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_1', 'Freq1_harmonics_amplitude_2', 
                            'Freq1_harmonics_amplitude_3', 'Freq1_harmonics_rel_phase_0', 'Freq1_harmonics_rel_phase_1', 
                            'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3', 'Freq2_harmonics_amplitude_0', 
                            'Freq2_harmonics_amplitude_1', 'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3', 
                            'Freq2_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_1', 'Freq2_harmonics_rel_phase_2', 
                            'Freq2_harmonics_rel_phase_3', 'Freq3_harmonics_amplitude_0', 'Freq3_harmonics_amplitude_1', 
                            'Freq3_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_3', 'Freq3_harmonics_rel_phase_0', 
                            'Freq3_harmonics_rel_phase_1', 'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3', 
                            'PercentDifferenceFluxPercentile', 'SmallKurtosis'],
                        '3': ['FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 
                                'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80', 'PercentDifferenceFluxPercentile', 
                                'SmallKurtosis'],
                        '4to19':['FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 
                                'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80', 'PercentDifferenceFluxPercentile'],
                        }

        if custom_remove is not None:
            # remove custom features and drop duplicates
            single_feets_features = np.delete(single_feets_features, np.where(np.isin(single_feets_features, custom_remove)))
            multi_feets_features = np.delete(multi_feets_features, np.where(np.isin(multi_feets_features, custom_remove)))
            
            # Append _g and _r to custom_remove for later use
            custom_remove_g = [x + '_g' for x in custom_remove]
            custom_remove_r = [x + '_r' for x in custom_remove]
            custom_remove_gandr = custom_remove_g + custom_remove_r

        # Some pre-processing
        dfg = df[df[fieldCol]==1].copy()
        dfr = df[df[fieldCol]==2].copy()
        dfg = dfg[~dfg[magCol].isnull()]
        dfr = dfr[~dfr[magCol].isnull()]

        # Not necessary if using lasair_clean function.
        # dfg = dfg.drop_duplicates(subset=timeCol, keep='first', inplace=False, ignore_index=True)
        # dfr = dfr.drop_duplicates(subset=timeCol, keep='first', inplace=False, ignore_index=True)

        # Data for align function
        timeInt = dfg[timeCol].values.astype('int')
        time2Int = dfr[timeCol].values.astype('int')
        magnitude = dfg[magCol].values
        magnitude2 = dfr[magCol].values
        error = dfg[errCol].values
        error2 = dfr[errCol].values

        # We synchronize the data for multiband features.
        atime, amag, amag2, aerror, aerror2 = self.align(timeInt, time2Int, magnitude, magnitude2, error, error2)

        # Extract features for dfg and dfr
        def extract(df, filter):
            if df.shape[0]==0:
                # Generate a dataframe with all features and NaN values.
                df = pd.DataFrame(columns=list(single_feets_features))
                # Append a row with NaN values.
                df_feets_single = df.append(pd.Series(), ignore_index=True)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]

            elif df.shape[0]==1:
                # Extract features for df
                if custom_remove is not None:
                    notextract = feat_remove['1'] + custom_remove
                    notextract = list(set(notextract))
                else:
                    notextract = feat_remove['1']
                
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, notextract)))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                if custom_remove is not None:
                    add_cols = [f for f in feat_remove['1'] if f not in custom_remove]
                else:
                    add_cols = feat_remove['1']

                for f in add_cols:
                    df_feets_single[f+f'_{filter}'] = np.nan
            
            elif df.shape[0]==2:
                # Extract features for dfg
                if custom_remove is not None:
                    notextract = feat_remove['2'] + custom_remove
                    notextract = list(set(notextract))
                else:
                    notextract = feat_remove['2']
                
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, notextract)))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                if custom_remove is not None:
                    add_cols = [f for f in feat_remove['2'] if f not in custom_remove]
                else:
                    add_cols = feat_remove['2']

                for f in add_cols:
                    df_feets_single[f+f'_{filter}'] = np.nan
            
            elif df.shape[0]==3:
                # Extract features for df
                if custom_remove is not None:
                    notextract = feat_remove['3'] + custom_remove
                    notextract = list(set(notextract))
                else:
                    notextract = feat_remove['3']
                
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, notextract)))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                if custom_remove is not None:
                    add_cols = [f for f in feat_remove['3'] if f not in custom_remove]
                else:
                    add_cols = feat_remove['3']

                for f in add_cols:
                    df_feets_single[f+f'_{filter}'] = np.nan
            
            elif (df.shape[0]>3) & (df.shape[0]<20):
                # Extract features for df
                if custom_remove is not None:
                    notextract = feat_remove['4to19'] + custom_remove
                    notextract = list(set(notextract))
                else:
                    notextract = feat_remove['4to19']
                
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, notextract)))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)

                # Add '_filter' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                if custom_remove is not None:
                    add_cols = [f for f in feat_remove['4to19'] if f not in custom_remove]
                else:
                    add_cols = feat_remove['4to19']

                for f in add_cols:
                    df_feets_single[f+f'_{filter}'] = np.nan

            else:
                # Extract features for dfg
                fs = feets.FeatureSpace(only=list(single_feets_features))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )

                df_feets_single = pd.DataFrame([values], columns=features)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]

            return df_feets_single
        
        # Extract single-band features
        df_feets_g = extract(dfg, 'g')
        df_feets_r = extract(dfr, 'r')
        
        # Extract multi-band features
        fs_multi = feets.FeatureSpace(only=list(multi_feets_features))
        
        try:
            features_multi, values_multi = fs_multi.extract(aligned_magnitude=amag,
                                                aligned_magnitude2=amag2,
                                                aligned_error=aerror,
                                                aligned_error2=aerror2,
                                                aligned_time=atime)
        
            df_feets_multi = pd.DataFrame([values_multi], columns=features_multi)
        except:
            df_feets_multi = pd.DataFrame(np.nan, index=[0], columns=multi_feets_features)

        # Concatenate the single and multi-band features
        df_feets = pd.concat([df_feets_g, df_feets_r, df_feets_multi], axis=1)

        # Impute based on subject knowledge
        original = ['Amplitude_g', 'AndersonDarling_g', 'Autocor_length_g', 'Beyond1Std_g', 'CAR_mean_g', 'CAR_sigma_g', 
            'CAR_tau_g', 'Con_g', 'Eta_e_g', 'FluxPercentileRatioMid20_g', 'FluxPercentileRatioMid35_g', 'FluxPercentileRatioMid50_g', 
            'FluxPercentileRatioMid65_g', 'FluxPercentileRatioMid80_g', 'Freq1_harmonics_amplitude_0_g', 'Freq1_harmonics_amplitude_1_g', 
            'Freq1_harmonics_amplitude_2_g', 'Freq1_harmonics_amplitude_3_g', 'Freq1_harmonics_rel_phase_0_g', 'Freq1_harmonics_rel_phase_1_g', 
            'Freq1_harmonics_rel_phase_2_g', 'Freq1_harmonics_rel_phase_3_g', 'Freq2_harmonics_amplitude_0_g', 'Freq2_harmonics_amplitude_1_g', 
            'Freq2_harmonics_amplitude_2_g', 'Freq2_harmonics_amplitude_3_g', 'Freq2_harmonics_rel_phase_0_g', 'Freq2_harmonics_rel_phase_1_g', 
            'Freq2_harmonics_rel_phase_2_g', 'Freq2_harmonics_rel_phase_3_g', 'Freq3_harmonics_amplitude_0_g', 'Freq3_harmonics_amplitude_1_g', 
            'Freq3_harmonics_amplitude_2_g', 'Freq3_harmonics_amplitude_3_g', 'Freq3_harmonics_rel_phase_0_g', 'Freq3_harmonics_rel_phase_1_g', 
            'Freq3_harmonics_rel_phase_2_g', 'Freq3_harmonics_rel_phase_3_g', 'Gskew_g', 'LinearTrend_g', 'MaxSlope_g',
            'Meanvariance_g', 'MedianAbsDev_g', 'MedianBRP_g', 'PairSlopeTrend_g', 'PercentAmplitude_g', 'PercentDifferenceFluxPercentile_g', 
            'PeriodLS_g', 'Period_fit_g', 'Psi_CS_g', 'Psi_eta_g', 'Q31_g', 'Rcs_g', 'Skew_g', 'SlottedA_length_g', 'SmallKurtosis_g', 
            'Std_g', 'StetsonK_g', 'StetsonK_AC_g', 'StructureFunction_index_21_g', 'StructureFunction_index_31_g', 
            'StructureFunction_index_32_g']

        replace = ['Amplitude_r', 'AndersonDarling_r', 'Autocor_length_r', 'Beyond1Std_r', 'CAR_mean_r', 'CAR_sigma_r', 'CAR_tau_r', 
                'Con_r', 'Eta_e_r', 'FluxPercentileRatioMid20_r', 'FluxPercentileRatioMid35_r', 'FluxPercentileRatioMid50_r', 
                'FluxPercentileRatioMid65_r', 'FluxPercentileRatioMid80_r', 'Freq1_harmonics_amplitude_0_r', 'Freq1_harmonics_amplitude_1_r', 
                'Freq1_harmonics_amplitude_2_r', 'Freq1_harmonics_amplitude_3_r', 'Freq1_harmonics_rel_phase_0_r', 'Freq1_harmonics_rel_phase_1_r',
                    'Freq1_harmonics_rel_phase_2_r', 'Freq1_harmonics_rel_phase_3_r', 'Freq2_harmonics_amplitude_0_r', 'Freq2_harmonics_amplitude_1_r', 
                    'Freq2_harmonics_amplitude_2_r', 'Freq2_harmonics_amplitude_3_r', 'Freq2_harmonics_rel_phase_0_r', 'Freq2_harmonics_rel_phase_1_r', 
                    'Freq2_harmonics_rel_phase_2_r', 'Freq2_harmonics_rel_phase_3_r', 'Freq3_harmonics_amplitude_0_r', 'Freq3_harmonics_amplitude_1_r', 
                    'Freq3_harmonics_amplitude_2_r', 'Freq3_harmonics_amplitude_3_r', 'Freq3_harmonics_rel_phase_0_r', 'Freq3_harmonics_rel_phase_1_r', 
                    'Freq3_harmonics_rel_phase_2_r', 'Freq3_harmonics_rel_phase_3_r', 'Gskew_r', 'LinearTrend_r', 'MaxSlope_r', 
                    'Meanvariance_r', 'MedianAbsDev_r', 'MedianBRP_r', 'PairSlopeTrend_r', 'PercentAmplitude_r', 'PercentDifferenceFluxPercentile_r', 
                    'PeriodLS_r', 'Period_fit_r', 'Psi_CS_r', 'Psi_eta_r', 'Q31_r', 'Rcs_r', 'Skew_r', 'SlottedA_length_r', 'SmallKurtosis_r', 
                    'Std_r', 'StetsonK_r', 'StetsonK_AC_r', 'StructureFunction_index_21_r', 'StructureFunction_index_31_r', 
                    'StructureFunction_index_32_r']
        
        if custom_remove is not None:
            original = [x for x in original if x not in custom_remove_gandr]
            replace = [x for x in replace if x not in custom_remove_gandr]
        
        newdf = self.impute_column(df_feets, original, replace)

        # Lets replace inf values with nan
        newdf = newdf.replace([np.inf, -np.inf], np.nan)

        return newdf


    # Extract custom features
    def extract_custom(self, remove=None, timeCol='jd', magCol='dc_mag', errCol='dc_sigmag', fieldCol='fid'):
        df = self.lc.copy()
        # Some pre-processing
        dfg = df[df[fieldCol]==1].copy()
        dfr = df[df[fieldCol]==2].copy()
        dfg = dfg[~dfg[magCol].isnull()]
        dfr = dfr[~dfr[magCol].isnull()]

        # Not necessary if using lasair_clean function.
        # dfg = dfg.drop_duplicates(subset=timeCol, keep='first', inplace=False, ignore_index=True)
        # dfr = dfr.drop_duplicates(subset=timeCol, keep='first', inplace=False, ignore_index=True)
    
        def extract(df, filter):
            df_cust = pd.DataFrame()
            df_cust[f'mean_{filter}'] = [df[magCol].mean()] # Can omit on return
            df_cust[f'median_{filter}'] = [df[magCol].median()] 
            # df_cust[f'std_{filter}'] = [df[magCol].std()] # Can omit on return
            # df_cust[f'MAD_{filter}'] = [self.mad(df[magCol])] # Can omit on return
            df_cust[f'min_mag_{filter}'] = [df[magCol].min()]
            df_cust[f'max_mag_{filter}'] = [df[magCol].max()]
            df_cust[f'n_obs_{filter}'] = [df[magCol].count()]

            df_cust[f'dif_min_mean_{filter}'] = abs(df_cust[f'min_mag_{filter}'] - df_cust[f'mean_{filter}']).iloc[0]
            df_cust[f'dif_min_median_{filter}'] = abs(df_cust[f'min_mag_{filter}'] - df_cust[f'median_{filter}']).iloc[0]
            df_cust[f'dif_max_mean_{filter}'] = abs(df_cust[f'max_mag_{filter}'] - df_cust[f'mean_{filter}']).iloc[0]
            df_cust[f'dif_max_median_{filter}'] = abs(df_cust[f'max_mag_{filter}'] - df_cust[f'median_{filter}']).iloc[0]
            df_cust[f'dif_max_min_{filter}'] = abs(df[magCol].max() - df[magCol].min())
            df_cust[f'temporal_baseline_{filter}'] = df[timeCol].max() - df[timeCol].min()
            df_cust[f'kurtosis_{filter}'] = df[magCol].kurtosis()

            # Lomb Scargle based features
            if df_cust[f'n_obs_{filter}'].iloc[0]>2:
                ls = LombScargle(df[timeCol], df[magCol])
                frequency, power = ls.autopower()
                df_cust[f'pwr_max_{filter}'] = [power.max()]
                df_cust[f'freq_pwr_max_{filter}'] = [frequency[np.where(power==power.max())][0]]
                df_cust[f'FalseAlarm_prob_{filter}'] = [ls.false_alarm_probability(power.max())]
                df_cust[f'pwr_maxovermean_{filter}'] = df_cust[f'pwr_max_{filter}']/power.mean()
            else:
                df_cust[f'pwr_max_{filter}'] = [np.nan]
                df_cust[f'freq_pwr_max_{filter}'] = [np.nan]
                df_cust[f'FalseAlarm_prob_{filter}'] = [np.nan]
                df_cust[f'pwr_maxovermean_{filter}'] = [np.nan]

            # Peak finder features
            df_cust[f'npeaks_pt5to1_{filter}'] = [self.peak_finder(df, prominence=(0.5, 1))[0]]
            df_cust[f'rrate_pt5to1_{filter}'] = [self.peak_finder(df, prominence=(0.5, 1))[1]]
            df_cust[f'drate_pt5to1_{filter}'] = [self.peak_finder(df, prominence=(0.5, 1))[2]]
            df_cust[f'amp_pt5to1_{filter}'] = [self.peak_finder(df, prominence=(0.5, 1))[3]]

            df_cust[f'npeaks_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[0]]
            df_cust[f'rrate_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[1]]
            df_cust[f'drate_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[2]]
            df_cust[f'amp_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[3]]

            df_cust[f'npeaks_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[0]]
            df_cust[f'rrate_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[1]]
            df_cust[f'drate_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[2]]
            df_cust[f'amp_1to2_{filter}'] = [self.peak_finder(df, prominence=(1, 2))[3]]

            df_cust[f'npeaks_2to5_{filter}'] = [self.peak_finder(df, prominence=(2, 5))[0]]
            df_cust[f'rrate_2to5_{filter}'] = [self.peak_finder(df, prominence=(2, 5))[1]]
            df_cust[f'drate_2to5_{filter}'] = [self.peak_finder(df, prominence=(2, 5))[2]]
            df_cust[f'amp_2to5_{filter}'] = [self.peak_finder(df, prominence=(2, 5))[3]]

            df_cust[f'npeaks_above5_{filter}'] = [self.peak_finder(df, prominence=(5, None))[0]]
            df_cust[f'rrate_above5_{filter}'] = [self.peak_finder(df, prominence=(5, None))[1]]
            df_cust[f'drate_above5_{filter}'] = [self.peak_finder(df, prominence=(5, None))[2]]
            df_cust[f'amp_above5_{filter}'] = [self.peak_finder(df, prominence=(5, None))[3]]

            # Standstill finder features
            df_cust[f'rollstd_ratio_t20s10_{filter}'] = [self.standstill_finder(df, pnt_threshold=20, window_size=10)[0]]
            df_cust[f'stdstilllev_t20s10_{filter}'] = [self.standstill_finder(df, pnt_threshold=20, window_size=10)[1]]
            df_cust[f'rollstd_ratio_t10s5_{filter}'] = [self.standstill_finder(df, pnt_threshold=10, window_size=5)[0]]
            df_cust[f'stdstilllev_t10s5{filter}'] = [self.standstill_finder(df, pnt_threshold=10, window_size=5)[1]]

            
            # Extract number of points brighter than x magnitudes brighter than the rolling median of window size y.
            win_size = [20]
            delta = [-1,-2,-5]
            for w in win_size:
                for d in delta:
                    df_cust[f'pnts_leq_rollMedWin{w}{d}mag_{filter}'] = \
                        [self.deviations_from_RollAv(magCol=df[magCol],
                            win_size=w,
                            delta=d,
                            rollType='median',
                            deviation_type='mag'
                            )]
                
            
            # Extract number of points fainter than x magnitudes fainter than the rolling median of window size y.
            win_size = [20]
            delta = [1,2,3]
            for w in win_size:
                for d in delta:
                    df_cust[f'pnts_geq_rollMedWin{w}+{d}mag_{filter}'] = \
                        [self.deviations_from_RollAv(magCol=df[magCol],
                            win_size=w,
                            delta=d,
                            rollType='median',
                            deviation_type='mag'
                            )]
            
            
            # Extract number of points brighter than x magnitudes brigher than median.
            delta = [-1,-2,-5]
            for d in delta:
                df_cust[f'pnts_leq_median{d}mag_{filter}'] = \
                        [self.deviations_from_average(
                        magCol=df[magCol],
                        avType='median',
                        deviation_type='mag',
                        delta=d
                        )]
                

            # Extract number of points fainter than x magnitudes fainter than median.
            delta = [1,2,3]
            for d in delta:
                df_cust[f'pnts_geq_median+{d}mag_{filter}'] = [\
                        self.deviations_from_average(
                        magCol=df[magCol],
                        avType='median',
                        deviation_type='mag',
                        delta=d
                        )]
                
            return df_cust
                
        
        df_cust_g = extract(dfg, 'g')
        df_cust_r = extract(dfr, 'r')
        df_custom = pd.concat([df_cust_g, df_cust_r], axis=1)

        # Colour extractor
        df_custom['clr_mean'] = [self.clr(df)[0]]
        df_custom['clr_median'] = [self.clr(df)[1]]
        df_custom['clr_std'] = [self.clr(df)[2]]
        df_custom['clr_bright'] = [self.clr(df)[3]]
        df_custom['clr_faint'] = [self.clr(df)[4]]

        df_custom.drop(['mean_g', 'mean_r'], axis=1, inplace=True)

        original = ['dif_min_mean_g', 'dif_min_median_g', 'dif_max_mean_g', 'dif_max_median_g', 'dif_max_min_g', 
                         'kurtosis_g', 'pwr_max_g', 'freq_pwr_max_g', 'FalseAlarm_prob_g', 'pwr_maxovermean_g']

        replace = ['dif_min_mean_r', 'dif_min_median_r', 'dif_max_mean_r', 'dif_max_median_r', 'dif_max_min_r', 
                    'kurtosis_r', 'pwr_max_r', 'freq_pwr_max_r', 'FalseAlarm_prob_r', 'pwr_maxovermean_r']

        original2 = ['pnts_leq_rollMedWin20-1mag_g', 'pnts_leq_rollMedWin20-2mag_g', 'pnts_leq_rollMedWin20-5mag_g', 
                    'pnts_geq_rollMedWin20+1mag_g', 'pnts_geq_rollMedWin20+2mag_g', 'pnts_geq_rollMedWin20+3mag_g', 
                    'pnts_leq_rollMedWin20-1mag_r', 'pnts_leq_rollMedWin20-2mag_r', 'pnts_leq_rollMedWin20-5mag_r', 
                    'pnts_geq_rollMedWin20+1mag_r', 'pnts_geq_rollMedWin20+2mag_r', 'pnts_geq_rollMedWin20+3mag_r',
                    'rollstd_ratio_t20s10_g','stdstilllev_t20s10_g','rollstd_ratio_t20s10_r','stdstilllev_t20s10_r']

        replace2 = ['pnts_leq_median-1mag_g', 'pnts_leq_median-2mag_g', 'pnts_leq_median-5mag_g', 
                    'pnts_geq_median+1mag_g', 'pnts_geq_median+2mag_g', 'pnts_geq_median+3mag_g',
                    'pnts_leq_median-1mag_r', 'pnts_leq_median-2mag_r', 'pnts_leq_median-5mag_r',
                    'pnts_geq_median+1mag_r', 'pnts_geq_median+2mag_r', 'pnts_geq_median+3mag_r',
                    'rollstd_ratio_t10s5_g', 'stdstilllev_t10s5g','rollstd_ratio_t10s5_r', 'stdstilllev_t10s5r']
        
        original3 = ['pnts_leq_rollMedWin20-1mag_g', 'pnts_leq_rollMedWin20-2mag_g', 'pnts_leq_rollMedWin20-5mag_g', 
                     'pnts_geq_rollMedWin20+1mag_g', 'pnts_geq_rollMedWin20+2mag_g', 'pnts_geq_rollMedWin20+3mag_g', 
                     'pnts_leq_median-1mag_g', 'pnts_leq_median-2mag_g', 'pnts_leq_median-5mag_g', 
                     'pnts_geq_median+1mag_g', 'pnts_geq_median+2mag_g', 'pnts_geq_median+3mag_g',
                     'rollstd_ratio_t20s10_g','stdstilllev_t20s10_g','rollstd_ratio_t10s5_g', 'stdstilllev_t10s5g']

        replace3 = ['pnts_leq_rollMedWin20-1mag_r', 'pnts_leq_rollMedWin20-2mag_r', 'pnts_leq_rollMedWin20-5mag_r', 
                    'pnts_geq_rollMedWin20+1mag_r', 'pnts_geq_rollMedWin20+2mag_r', 'pnts_geq_rollMedWin20+3mag_r',
                    'pnts_leq_median-1mag_r', 'pnts_leq_median-2mag_r', 'pnts_leq_median-5mag_r',
                    'pnts_geq_median+1mag_r', 'pnts_geq_median+2mag_r', 'pnts_geq_median+3mag_r',
                    'rollstd_ratio_t20s10_r','stdstilllev_t20s10_r','rollstd_ratio_t10s5_r', 'stdstilllev_t10s5r']

        newdf = self.impute_column(df_custom, original, replace)
        newdf2 = self.impute_column(newdf, original2, replace2, reversed=False)
        newdf3 = self.impute_column(newdf2, original3, replace3, reversed=True)

        # Drop columns that are in remove list
        if remove is not None:
            newdf3.drop(remove, axis=1, inplace=True)

        return newdf3
