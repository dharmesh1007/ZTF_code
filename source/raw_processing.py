import pandas as pd

def label_schemes(cvs_df):

    # This function is just for the manually labelled csv file, that is the input.

    # Select columns; rename columns.
    cvs_df = cvs_df[['Xmatch_obj','Name','Type','ra','dec','Eclipsing','CV_Types','CV_subtypes',
                 'CV_subsubtypes','eclipse_clear','manual_label','Clarity']]
    
    cvs_df.rename(columns={'Xmatch_obj':'oid_ztf', 'Name':'oid_aavso','Type':'type_aavso'}, inplace=True)

    # Apply label schemes.

    # Label scheme 1
    cvs_df.loc[(cvs_df.manual_label.isin(['SU_Uma','SU_UMa'])), 'labels_1'] = 'dwarf_nova_SU_UMa'
    cvs_df.loc[(cvs_df.manual_label.isin(['WZ_Sge'])), 'labels_1'] = 'dwarf_nova_WZ_Sge'
    cvs_df.loc[(cvs_df.manual_label.isin(['ER_UMa'])), 'labels_1'] = 'dwarf_nova_ER_UMa'
    cvs_df.loc[(cvs_df.manual_label.isin(['U_Gem'])), 'labels_1'] = 'dwarf_nova_U_Gem'
    cvs_df.loc[(cvs_df.manual_label.isin(['Z_Cam'])), 'labels_1'] = 'dwarf_nova_Z_Cam'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova'])), 'labels_1'] = 'nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_decline'])), 'labels_1'] = 'nova_decline'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_remnant'])), 'labels_1'] = 'nova_remnant'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_like'])), 'labels_1'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['VY Scl','VY_Scl'])), 'labels_1'] = 'nova_like_VY_Scl'
    cvs_df.loc[(cvs_df.manual_label.isin(['AMCVn'])), 'labels_1'] = 'AMCVn'
    cvs_df.loc[(cvs_df.manual_label.isin(['DQ_Her','DQ_Her+U_Gem'])), 'labels_1'] = 'int_polar'
    cvs_df.loc[(cvs_df.manual_label.isin(['AM_Her'])), 'labels_1'] = 'polar'

    # Label scheme 2
    cvs_df.loc[(cvs_df.manual_label.isin(['SU_Uma','SU_UMa'])), 'labels_2'] = 'dwarf_nova_SU_UMa'
    cvs_df.loc[(cvs_df.manual_label.isin(['WZ_Sge'])), 'labels_2'] = 'dwarf_nova_SU_UMa'
    cvs_df.loc[(cvs_df.manual_label.isin(['ER_UMa'])), 'labels_2'] = 'dwarf_nova_SU_UMa'
    cvs_df.loc[(cvs_df.manual_label.isin(['U_Gem'])), 'labels_2'] = 'dwarf_nova_U_Gem'
    cvs_df.loc[(cvs_df.manual_label.isin(['Z_Cam'])), 'labels_2'] = 'dwarf_nova_Z_Cam'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova'])), 'labels_2'] = 'nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_decline'])), 'labels_2'] = 'nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_remnant'])), 'labels_2'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_like'])), 'labels_2'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['VY Scl','VY_Scl'])), 'labels_2'] = 'nova_like_VY_Scl'
    cvs_df.loc[(cvs_df.manual_label.isin(['AMCVn'])), 'labels_2'] = 'AMCVn'
    cvs_df.loc[(cvs_df.manual_label.isin(['DQ_Her','DQ_Her+U_Gem'])), 'labels_2'] = 'int_polar'
    cvs_df.loc[(cvs_df.manual_label.isin(['AM_Her'])), 'labels_2'] = 'polar'


    # Label scheme 3
    cvs_df.loc[(cvs_df.manual_label.isin(['SU_Uma','SU_UMa'])), 'labels_3'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['WZ_Sge'])), 'labels_3'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['ER_UMa'])), 'labels_3'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['U_Gem'])), 'labels_3'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['Z_Cam'])), 'labels_3'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova'])), 'labels_3'] = 'nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_decline'])), 'labels_3'] = 'nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_remnant'])), 'labels_3'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_like'])), 'labels_3'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['VY Scl','VY_Scl'])), 'labels_3'] = 'nova_like_VY_Scl'
    cvs_df.loc[(cvs_df.manual_label.isin(['AMCVn'])), 'labels_3'] = 'AMCVn'
    cvs_df.loc[(cvs_df.manual_label.isin(['DQ_Her','DQ_Her+U_Gem'])), 'labels_3'] = 'int_polar'
    cvs_df.loc[(cvs_df.manual_label.isin(['AM_Her'])), 'labels_3'] = 'polar'

    # Label scheme 4
    cvs_df.loc[(cvs_df.manual_label.isin(['SU_Uma','SU_UMa'])), 'labels_4'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['WZ_Sge'])), 'labels_4'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['ER_UMa'])), 'labels_4'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['U_Gem'])), 'labels_4'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['Z_Cam'])), 'labels_4'] = 'dwarf_nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova'])), 'labels_4'] = 'nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_decline'])), 'labels_4'] = 'nova'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_remnant'])), 'labels_4'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['nova_like'])), 'labels_4'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['VY Scl','VY_Scl'])), 'labels_4'] = 'nova_like'
    cvs_df.loc[(cvs_df.manual_label.isin(['AMCVn'])), 'labels_4'] = 'AMCVn'
    cvs_df.loc[(cvs_df.manual_label.isin(['DQ_Her','DQ_Her+U_Gem'])), 'labels_4'] = 'magnetic'
    cvs_df.loc[(cvs_df.manual_label.isin(['AM_Her'])), 'labels_4'] = 'magnetic'

    # print(cvs_df.labels_4.value_counts())

    return cvs_df