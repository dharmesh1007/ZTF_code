import numpy as np
import pandas as pd
import feets
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.stats import linregress

class FeatureExtractor:
    def __init__(self, lc):
        self.lc = lc

    # Functon for aligning time and magnitude for each band.
    def align(time, time2, magnitude, magnitude2, error, error2):
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
        
        return instances/len(magCol) if len(magCol)>win_size else np.nan


    # Number of data points above or below delta mags or stds of the median or mean of the column.
    def deviations_from_average(self, magCol, avType='median', deviation_type='mag', delta=1):
        
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
            return np.nan

    # Median absolute deviation function.
    def mad(array):
        a = np.abs(array-np.median(array))
        b = np.median(a)
        return b

    # Method for finding standstills in Z_Cam light curves
    def standstill_finder(lc, pnt_threshold=20, window_size=10):
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

            # print(f'Rolling std max: {roll_std.max()}',
            #     f'\nRolling std min: {roll_std.min()}',
            #     f'\nRatio of max and min of rolling std: {roll_std.max()/roll_std.min()}',
            #     f'\nMean brightness of window with minimum rolling std: {min_roll_std_mean}'
            #     f'\nStandstill level: {standstill_level}')
            
            return roll_std.max(), roll_std.min(), roll_std.max()/roll_std.min(), min_roll_std_mean, standstill_level
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan

    def peak_definer(lc, height=None, threshold=None, distance=None, prominence=(2,5), 
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
                    rise_rate = np.nan
                    decline_rate = linreg.slope*(-1)
                    max_prominence = np.nan
                else:
                    n_peaks = 0
                    rise_rate = np.nan
                    decline_rate = np.nan
                    max_prominence = np.nan
                
            except:
                n_peaks = 0
                rise_rate = np.nan
                decline_rate = np.nan
                max_prominence = np.nan


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

    

    # Function to calculate features from feets package.
    def extract_feets(self, timeCol='jd', magCol='dc_mag', errCol='dc_sigmag', fieldCol='fid'):
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
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, feat_remove['1'])))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                for f in feat_remove['1']:
                    df_feets_single[f+f'_{filter}'] = np.nan

            elif df.shape[0]==2:
                # Extract features for dfg
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, feat_remove['2'])))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                for f in feat_remove['2']:
                    df_feets_single[f+f'_{filter}'] = np.nan
            
            elif df.shape[0]==3:
                # Extract features for dfg
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, feat_remove['3'])))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)
                # Add '_g' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                for f in feat_remove['3']:
                    df_feets_single[f+f'_{filter}'] = np.nan
            
            elif (df.shape[0]>3) & (df.shape[0]<20):
                # Extract features for df
                single_feets_features_revised = np.delete(single_feets_features, np.where(np.isin(single_feets_features, feat_remove['4to19'])))
                fs = feets.FeatureSpace(only=list(single_feets_features_revised))
                features, values = fs.extract(time=df[timeCol],
                                            magnitude=df[magCol], 
                                            error=df[errCol]
                                            )
                df_feets_single = pd.DataFrame([values], columns=features)

                # Add '_filter' to the column names.
                df_feets_single.columns = [col+f'_{filter}' for col in df_feets_single.columns]
                # Add in the dropped features as columns with NaN values.
                for f in feat_remove['4to19']:
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
        df_feets_g = extract(df, 'g')
        df_feets_r = extract(df, 'r')
        
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

        return df_feets


    # Extract custom features
    def extract_custom(self, df, timeCol='jd', magCol='dc_mag', errCol='dc_sigmag', fieldCol='fid'):
            
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
            df_cust[f'std_{filter}'] = [df[magCol].std()] # Can omit on return
            df_cust[f'MAD_{filter}'] = [self.mad(df[magCol])] # Can omit on return
            df_cust[f'min_mag_{filter}'] = [df[magCol].min()]
            df_cust[f'max_mag_{filter}'] = [df[magCol].max()]
            df_cust[f'n_obs_{filter}'] = [df[magCol].count()]

            df_cust[f'dif_min_mean_{filter}'] = abs(df_cust[f'min_mag_{filter}'] - df_cust[f'mean_{filter}']).iloc[0]
            df_cust[f'dif_min_median_{filter}'] = abs(df_cust[f'min_mag_{filter}'] - df_cust[f'median_{filter}']).iloc[0]
            df_cust[f'dif_max_mean_{filter}'] = abs(df_cust[f'max_mag_{filter}'] - df_cust[f'mean_{filter}']).iloc[0]
            df_cust[f'dif_max_median_{filter}'] = abs(df_cust[f'max_mag_{filter}'] - df_cust[f'median_{filter}']).iloc[0]
            df_cust[f'dif_max_min_{filter}'] = abs(df[magCol].max() - df[magCol].min())
            df_cust[f'temporal_baseline_{filter}'] = df[timeCol].max() - df[timeCol].min()
            df_cust[f'avg_obs_per_day_{filter}'] = df_cust[f'n_obs_{filter}']/df_cust[f'temporal_baseline_{filter}']
            df_cust[f'kurtosis_{filter}'] = df[magCol].kurtosis()

            # Lomb Scargle based features
            if df_cust[f'n_obs_{filter}'].iloc[0]>2:
                ls = LombScargle(df[timeCol], df[magCol])
                frequency, power = ls.autopower()
                df_cust[f'pwr_max_{filter}'] = [power.max()]
                df_cust[f'freq_pwr_max_{filter}'] = [frequency[np.where(power==power.max())][0]]
                df_cust[f'FalseAlarm_prob_{filter}'] = [ls.false_alarm_probability(power.max())]
            else:
                df_cust[f'pwr_max_{filter}'] = [np.nan]
                df_cust[f'freq_pwr_max_{filter}'] = [np.nan]
                df_cust[f'FalseAlarm_prob_{filter}'] = [np.nan]


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
        
        # Extract colour
        df_custom['clr_median'] = df_custom['median_g'] - df_custom['median_r']
        df_custom['clr_mean'] = df_custom['mean_g'] - df_custom['mean_r']
        df_custom['clr_minmag'] = df_custom['min_mag_g'] - df_custom['min_mag_r']
        df_custom['clr_maxmag'] = df_custom['max_mag_g'] - df_custom['max_mag_r']

        df_custom.drop(['mean_g', 'mean_r', 'std_g', 'std_r', 'MAD_g', 'MAD_r'], axis=1, inplace=True)

        return df_custom

