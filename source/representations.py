from scipy.signal import find_peaks
import numpy as np

def getdiffs(array):
    if len(array):
        diffs = np.diff(array)
        return diffs
    else:
        return np.array([])

def backfilled_magnitudes(lightcurves, n_pts=1000, backfill_value=np.nan):
    X_gmag = np.empty((0, n_pts))
    X_rmag = np.empty((0, n_pts))
    X_gjd = np.empty((0, n_pts))
    X_rjd = np.empty((0, n_pts))

    for obj in lightcurves.keys():
        for key in ['g_mag', 'r_mag', 'g_jd', 'r_jd']:
            lc = lightcurves[obj][key]

            if key == 'g_jd' or key == 'r_jd':
                try:
                    lc = (lc - lc[0])
                except:
                    lc = []

            if len(lc) < n_pts:
                lc = np.pad(lc, (n_pts-len(lc), 0), 'constant', constant_values=(backfill_value, backfill_value))
            elif len(lc) >= n_pts:
                lc = lc[-n_pts:]
            
            if key == 'g_mag':
                X_gmag = np.vstack((X_gmag, lc))
            elif key == 'r_mag':
                X_rmag = np.vstack((X_rmag, lc))
            elif key == 'g_jd':
                X_gjd = np.vstack((X_gjd, lc))
            elif key == 'r_jd':
                X_rjd = np.vstack((X_rjd, lc))
            
    return X_gmag, X_rmag, X_gjd, X_rjd


def dmdt(lightcurves, n_pts=500, backfill_value=np.nan):

    X_gdm = np.empty((0, n_pts))
    X_gdt = np.empty((0, n_pts))
    X_rdm = np.empty((0, n_pts))
    X_rdt = np.empty((0, n_pts))

    for obj in lightcurves.keys():
        for key in ['g_mag', 'r_mag', 'g_jd', 'r_jd']:
            lc = lightcurves[obj][key]

            if len(lc) > 0:
                diff = np.diff(lc)
                diff = np.insert(diff, 0, 0)
            elif len(lc) == 0:
                diff = np.array([])
            
            if len(diff) < n_pts:
                diff = np.pad(diff, (n_pts-len(diff), 0), 'constant', constant_values=(backfill_value, backfill_value))
                
            elif len(diff) >= n_pts:
                diff = diff[-n_pts:]
            
            if key == 'g_mag':
                X_gdm = np.vstack((X_gdm, diff))
            elif key == 'r_mag':
                X_rdm = np.vstack((X_rdm, diff))
            elif key == 'g_jd':
                X_gdt = np.vstack((X_gdt, diff))
            elif key == 'r_jd':
                X_rdt = np.vstack((X_rdt, diff))

    return X_gdm, X_rdm, X_gdt, X_rdt


def interpolated_lcs(lightcurves, interp_pts=1000, true_cadence=True, backfill_value=0, cadence=1):
        X_ginterp = np.empty((0, interp_pts))
        X_rinterp = np.empty((0, interp_pts))

    
        for obj in list(lightcurves.keys()):

            for magkey, timekey in zip(['g_mag', 'r_mag'], ['g_jd', 'r_jd']):
                mag = lightcurves[obj][magkey]
                time = lightcurves[obj][timekey]

                # Convert jd to days since first observation
                try:
                    time = (time - time[0])
                except:
                    time = []

                 # Interpolate the light curve to 1 day cadence or a set number of points
                try:
                    if true_cadence==True:
                        lc_interp = np.interp(np.arange(0, time[-1], cadence), time, mag)
                    elif true_cadence==False:
                        lc_interp = np.interp(np.arange(0, time[-1], time[-1]/interp_pts), time, mag)
                except:
                    lc_interp = np.array([])
                
                # Pad the light curve with zeros if it is shorter than the desired length, or trim it if it is longer
                if len(lc_interp) < interp_pts:
                    lc_interp = np.pad(lc_interp, (interp_pts-len(lc_interp), 0), 'constant', constant_values=(backfill_value, backfill_value))
                elif len(lc_interp) >= interp_pts:
                    lc_interp = lc_interp[-interp_pts:]
                
                # Append the interpolated light curve to the appropriate array
                if magkey == 'g_mag':
                    X_ginterp = np.vstack((X_ginterp, lc_interp))
                elif magkey == 'r_mag':
                    X_rinterp = np.vstack((X_rinterp, lc_interp))

        return X_ginterp, X_rinterp


def gminusr(lightcurves, interp_pts=500, true_cadence=True, backfill_value=0, cadence=1):

    clr = np.empty((0, interp_pts))

    for obj in list(lightcurves.keys()):
        # print(obj)
        # df_g = pd.DataFrame(lightcurves[obj]['g_mag', 'g_jd'].copy())
        gmag = lightcurves[obj]['g_mag'].copy()
        gjd = lightcurves[obj]['g_jd'].copy()
        rmag = lightcurves[obj]['r_mag'].copy()
        rjd = lightcurves[obj]['r_jd'].copy()

        gjd = gjd.astype(int)
        rjd = rjd.astype(int)

        # Drop the duplicate jd and corresponding mag.
        g_jd_unique0 = np.unique(gjd, return_index=True)    # The unique function also returns the indices of unique values
        g_jd_unique = g_jd_unique0[0]                       # The unique jd values
        g_jd_unique_idx = g_jd_unique0[1]                   # The indices of the unique jd values can be used as a mask
        gmag_unique = gmag[g_jd_unique_idx]                 # The unique mag values

        r_jd_unique0 = np.unique(rjd, return_index=True)
        r_jd_unique = r_jd_unique0[0]
        r_jd_unique_idx = r_jd_unique0[1]
        rmag_unique = rmag[r_jd_unique_idx]

        # Get a boolean array of whether each g_jd_unique is in r_jd_unique. Use it as a mask to get gmag and rmag
        # values that were taken at the same time.
        gmask = np.in1d(g_jd_unique, r_jd_unique)
        gjd_inr = g_jd_unique[gmask]    
        # print(gjd_inr.shape)  
        gmag_inr = gmag_unique[gmask]  
        # print(gmag_inr.shape)        


        # Get a boolean array of whether each r_jd is in g_jd
        rmask = np.in1d(r_jd_unique, g_jd_unique)
        rjd_ing = r_jd_unique[rmask]
        # print(rjd_ing.shape)
        rmag_ing = rmag_unique[rmask]
        # print(rmag_ing.shape)

        # Convert jd to days since first observation
        try:
            time = (gjd_inr - gjd_inr[0])
        except:
            time = []
        
        clr_source = gmag_inr - rmag_ing

        # Interpolate the light curve to 1 day cadence or a set number of points
        try:
            if true_cadence==True:
                clr_interp = np.interp(np.arange(0, time[-1], cadence), time, clr_source)
            elif true_cadence==False:
                clr_interp = np.interp(np.arange(0, time[-1], time[-1]/interp_pts), time, clr_source)
        except:
            clr_interp = np.array([])
        
        # Pad the light curve with zeros if it is shorter than the desired length, or trim it if it is longer
        if len(clr_interp) < interp_pts:
            clr_interp = np.pad(clr_interp, (interp_pts-len(clr_interp), 0), 'constant', constant_values=(backfill_value, backfill_value))
        elif len(clr_interp) >= interp_pts:
            clr_interp = clr_interp[-interp_pts:]
        
        clr_interp = clr_interp.reshape(1, clr_interp.shape[0])
        # print(clr_source.shape)
        clr = np.vstack((clr, clr_interp))

    return clr, clr_source, time

def dmdt_hist(lightcurves, bins, limit, dm=False, usepeaks=False):

    g_all = np.array([])
    r_all = np.array([])
    g_hist_all = np.empty((0, bins))
    r_hist_all = np.empty((0, bins))

    rand = np.random.randint(0, len(lightcurves))
    for obj in list(lightcurves.keys()):#[rand:rand+1]:
        lc = lightcurves[obj]
        lc_g = lc['g_mag']
        lc_r = lc['r_mag']
        lc_g_jd = lc['g_jd']
        lc_r_jd = lc['r_jd']

        if usepeaks == True:
            # Find the peaks and the troughs of lc_g and lc_r
            g_peaks, _ = find_peaks(-lc_g, prominence=(None,None))
            g_troughs, _ = find_peaks(lc_g, prominence=(None,None))
            r_peaks, _ = find_peaks(-lc_r, prominence=(None,None))
            r_troughs, _ = find_peaks(lc_r, prominence=(None,None))

            lc_g = lc_g[np.sort(np.concatenate((g_peaks, g_troughs)))]
            lc_r = lc_r[np.sort(np.concatenate((r_peaks, r_troughs)))]
            lc_g_jd = lc_g_jd[np.sort(np.concatenate((g_peaks, g_troughs)))]
            lc_r_jd = lc_r_jd[np.sort(np.concatenate((r_peaks, r_troughs)))]
        

        # Get the dm and dt arrays
        g_dm = getdiffs(lc_g)
        r_dm = getdiffs(lc_r)
        g_dt = getdiffs(lc_g_jd)
        r_dt = getdiffs(lc_r_jd)

        if dm == True:
            # Just use the dms
            g_processed = g_dm
            r_processed = r_dm
        else:
            # Get the dmdt arrays
            g_dmdt = g_dm / g_dt
            g_processed = np.nan_to_num(g_dmdt, nan=0)
            r_dmdt = r_dm / r_dt
            r_processed = np.nan_to_num(r_dmdt, nan=0)

        # Add the g and r processed arrays to the all arrays
        g_all = np.append(g_all, g_processed, axis=0)
        r_all = np.append(r_all, r_processed, axis=0)

        # Get the histograms. First clip the arrays to the limit.
        g_processed[g_processed > limit] = limit
        g_processed[g_processed < -limit] = -limit
        r_processed[r_processed > limit] = limit
        r_processed[r_processed < -limit] = -limit

        g_hist, g_bins = np.histogram(g_processed, bins=np.linspace(-limit, limit, bins+1))
        r_hist, r_bins = np.histogram(r_processed, bins=np.linspace(-limit, limit, bins+1))

        # print(g_hist)

        # Add the histogram to the array
        g_hist_all = np.vstack((g_hist_all, g_hist))
        r_hist_all = np.vstack((r_hist_all, r_hist))

        # # plot the peaks and troughs and indicate them
        # plt.figure(figsize=(20, 5))
        # plt.plot(lc_g_jd, lc_g, '.')
        # plt.plot(lc_g_jd[g_peaks], lc_g[g_peaks], "x")
        # plt.plot(lc_g_jd[g_troughs], lc_g[g_troughs], "x")
        # plt.gca().invert_yaxis()
        # plt.show()

        # plt.figure(figsize=(20, 5))
        # plt.plot(lc_g_peaksandtroughs_jd, lc_g_peaksandtroughs, '.')
        # plt.gca().invert_yaxis()
        # plt.show()

        # print(np.diff(lc_g_peaksandtroughs))

    return g_hist_all, r_hist_all, g_all, r_all

def multi_channel(array_list):
    reshaped_arrays = []
    for array in array_list:
        reshaped_arrays.append(np.reshape(array, (array.shape[0], array.shape[1], 1)))

    new_arrray = np.concatenate(reshaped_arrays, axis=2)
    return new_arrray

