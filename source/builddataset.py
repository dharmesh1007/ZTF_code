from lcfunctions import load_lasair_lc, lasair_clean
from featureextractor import FeatureExtractor
from metadatafeatures import gaiadr3append
import pandas as pd
import numpy as np


def build_dataset(df, objcol, folderpath='../lightcurves_dataset/lasair_2023_03_25'):
    
    metadata = gaiadr3append(df=df, objcol=objcol)

    # Metadata columns to drop
    drop = ['parallax_over_error', 'pmra', 'pmdec', 'ra_dec_corr', 'ra_parallax_corr', 
            'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 
            'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 'astrometric_n_obs_al', 
            'astrometric_n_obs_ac', 'astrometric_n_good_obs_al', 'astrometric_n_bad_obs_al', 
            'astrometric_gof_al', 'astrometric_chi2_al', 'astrometric_excess_noise', 
            'astrometric_excess_noise_sig', 'astrometric_params_solved', 'astrometric_primary_flag', 
            'pseudocolour', 'pseudocolour_error', 'ra_pseudocolour_corr', 'dec_pseudocolour_corr', 
            'parallax_pseudocolour_corr', 'pmra_pseudocolour_corr', 'pmdec_pseudocolour_corr', 
            'astrometric_matched_transits', 'visibility_periods_used', 'matched_transits', 
            'new_matched_transits', 'matched_transits_removed', 'ipd_gof_harmonic_amplitude', 
            'ipd_gof_harmonic_phase', 'ipd_frac_multi_peak', 'ipd_frac_odd_win', 'ruwe', 
            'scan_direction_strength_k1', 'scan_direction_strength_k2', 'scan_direction_strength_k3', 
            'scan_direction_strength_k4', 'scan_direction_mean_k1', 'scan_direction_mean_k2', 
            'scan_direction_mean_k3', 'scan_direction_mean_k4', 'duplicated_source', 'phot_g_mean_flux_over_error', 
            'phot_bp_mean_flux_over_error', 'phot_rp_mean_flux_over_error', 'phot_bp_rp_excess_factor', 
            'phot_bp_n_contaminated_transits', 'phot_bp_n_blended_transits', 'phot_rp_n_contaminated_transits', 
            'phot_rp_n_blended_transits', 'phot_proc_mode', 'radial_velocity', 'radial_velocity_error', 
            'rv_method_used', 'rv_nb_transits', 'rv_nb_deblended_transits', 'rv_visibility_periods_used', 
            'rv_expected_sig_to_noise', 'rv_renormalised_gof', 'rv_chisq_pvalue', 'rv_time_duration', 
            'rv_amplitude_robust', 'rv_template_teff', 'rv_template_logg', 'rv_template_fe_h', 
            'rv_atm_param_origin', 'vbroad', 'vbroad_error', 'vbroad_nb_transits', 'grvs_mag', 'grvs_mag_error', 
            'grvs_mag_nb_transits', 'rvs_spec_sig_to_noise', 'phot_variable_flag', 'in_qso_candidates', 
            'in_galaxy_candidates', 'non_single_star', 'has_xp_continuous', 'has_xp_sampled', 'has_rvs', 
            'has_epoch_photometry', 'has_epoch_rv', 'has_mcmc_gspphot', 'has_mcmc_msc', 'in_andromeda_survey', 
            'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy', 'classprob_dsc_combmod_star', 
            'teff_gspphot', 'teff_gspphot_lower', 'teff_gspphot_upper', 'logg_gspphot', 'logg_gspphot_lower', 
            'logg_gspphot_upper', 'mh_gspphot', 'mh_gspphot_lower', 'mh_gspphot_upper', 'distance_gspphot', 
            'distance_gspphot_lower', 'distance_gspphot_upper', 'azero_gspphot', 'azero_gspphot_lower', 
            'azero_gspphot_upper', 'ag_gspphot', 'ag_gspphot_lower', 'ag_gspphot_upper', 'ebpminrp_gspphot', 
            'ebpminrp_gspphot_lower', 'ebpminrp_gspphot_upper', 'libname_gspphot', 'ra_prop', 'dec_prop']
    
    # Drop metadata columns
    metadata = metadata.drop(drop, axis=1)

    # Extract light curve features
    objlist = df[objcol].tolist()
    feature_df = pd.DataFrame()
    
    for count, obj in enumerate(objlist):
        print(count, obj)
        # Load and process lasair light curve
        lc_test = load_lasair_lc(oid=obj, path=folderpath)
        lc_appmag_test = lasair_clean(lc_test, limit=25, magerrlim=1)

        # Create a copy of the light curve
        lc = lc_appmag_test.copy()

        # Extract features
        fe = FeatureExtractor(lc=lc)
        feets = fe.extract_feets(outliercap=True, custom_remove=['FalseAlarm_prob','Eta_color',
                                                                 'Freq1_harmonics_rel_phase_0', 
                                                                 'Freq2_harmonics_rel_phase_0', 
                                                                 'Freq3_harmonics_rel_phase_0'])
        
        custom = fe.extract_custom()
        
        # Conactenate custom features to feets
        features_single = pd.concat([feets, custom], axis=1)
        # Add features to dataframe
        feature_df = feature_df.append(features_single, ignore_index=True)

    # Add oid to dataframe as the first column
    feature_df.insert(0, 'oid_ztf', objlist)
    # feature_df['oid_ztf'] = objlist # This is another way to do it

    # Replace inf values with nan
    for col in metadata.iloc[:, 1:]:
        metadata.loc[(metadata[col]==np.inf)|(metadata[col]==-np.inf), col] = np.nan
    
    for col in feature_df.iloc[:, 1:]:
        feature_df.loc[(feature_df[col]==np.inf)|(feature_df[col]==-np.inf), col] = np.nan
    
    # Merge the two datasets
    merged = pd.merge(feature_df, metadata, left_on='oid_ztf', right_on=objcol, how='left')

    if objcol != 'oid_ztf':
        # Drop the Xmatch_obj column
        merged = merged.drop(columns=[objcol])

    return feature_df, metadata, merged

