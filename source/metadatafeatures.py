from astroquery.gaia import Gaia
from astropy.table import Table
import pandas as pd
import numpy as np
import math

# Get distance and absolute magnitude from Gaia parallax and apparent magnitude.
def distance(parallax):
    """Convert Gaia parallax to distance in parsecs."""
    return 1000./parallax

def absmag(appmag, distance):
    """Convert apparent magnitude to absolute magnitude."""
    return appmag - 5*np.log10(distance) + 5

# Functions to acquire Gaia metadata.

def getGaiaData(target_df, radius):

    # Astropy table with coordinates
    alerts_table = Table([target_df['ra'], target_df['dec']], names=['RA', 'Dec'], meta={'meta':'CVs'})
    
    # Login to archive
    Gaia.login(user='dmistry', password='X641oVxh8kLj#')

    # Delete tables if present
    try:
        Gaia.delete_user_table("ast_table")
    except:
        pass
    try:
        Gaia.delete_user_table("xmatch_table")
    except:
        pass

    # Upload table
    Gaia.upload_table(upload_resource=alerts_table, table_name='ast_table')

    # Update table
    Gaia.update_user_table(table_name="user_dmistry.ast_table",
                        list_of_changes=[["ra", "flags", "ra"],
                                            ["dec","flags","dec"]])


    # Perform cross-match of coordinates with Gaia data release 3 data using cone search, collect results.
    # Radius is set to 1 originally but may increase to 2 arcseconds.
    full_qualified_table_name = 'user_dmistry.ast_table'
    xmatch_table_name = 'xmatch_table'
    Gaia.cross_match(full_qualified_table_name_a=full_qualified_table_name,
                    full_qualified_table_name_b='gaiadr3.gaia_source',
                    results_table_name=xmatch_table_name, radius=radius, verbose=False)
    

    # Collect meta data for each of the successfully cross matched targets from Gaia DR3.

    # Cross matched targets.
    xmatch_table = 'user_dmistry.' + xmatch_table_name
    
    # Query. Separation is in degrees, need to convert to arcseconds so * 3600
    query = ('SELECT c."separation"*3600 as dist, a.*, b.* FROM gaiadr3.gaia_source AS a, '
            '%s AS b,'
            '%s AS c '
            'WHERE (c.gaia_source_source_id = a.source_id AND '
            'c.ast_table_ast_table_oid = b.ast_table_oid)' %(full_qualified_table_name, xmatch_table))
    
    # Launch query
    job = Gaia.launch_job(query=query)
    results = job.get_results()
        
    # Place results into dataframe.  
    df_archive=results.to_pandas()

    # Clear tables and jobs from Gaia workspace and log out.
    Gaia.delete_user_table("ast_table")
    Gaia.delete_user_table("xmatch_table")
    jobs = [job for job in Gaia.list_async_jobs()]
    job_ids = [job.jobid for job in jobs]
    Gaia.remove_jobs(job_ids)
    Gaia.logout()

    return df_archive
        
    # Function to acquire Gaia metadata.



# Function to upload data and retrieve targets via a cross match.
def gaia_xmatch(target_df, query, racol, deccol):

    # Astropy table with coordinates
    alerts_table = Table([target_df['ra'], target_df['dec']], names=[racol, deccol], meta={'meta':'CVs'})

    # Login to archive
    Gaia.login(user='dmistry', password='X641oVxh8kLj#')

    # Delete tables if present
    try:
        Gaia.delete_user_table("ast_table")
    except:
        pass
    try:
        Gaia.delete_user_table("xmatch_table")
    except:
        pass

    # Upload table
    Gaia.upload_table(upload_resource=alerts_table, table_name='ast_table')

    # Update table
    Gaia.update_user_table(table_name="user_dmistry.ast_table",
                           list_of_changes=[[racol, "flags", "ra"],
                                            [deccol,"flags", "dec"]]
    )
    
    # Launch query
    job = Gaia.launch_job_async(query=query)
    
    results = job.get_results()
        
    # Place results into dataframe.  
    df_archive=results.to_pandas()

    # Clear tables and jobs from Gaia workspace and log out.
    Gaia.delete_user_table("ast_table")
    jobs = [job for job in Gaia.list_async_jobs()]
    job_ids = [job.jobid for job in jobs]
    Gaia.remove_jobs(job_ids)
    Gaia.logout()

    return df_archive


# Queries and columns to use.


def gaiadr3append(df, objcol):

    # Column names to assign for right ascension and declination of our table.
    racol = 'raj2000'
    deccol = 'decj2000'

    # Basic query not taking into account proper motion. This is for where no proper motion is available.
    query = ("SELECT *," + " "
                "ra + 1. / 3600000 * pmra * (2000.0 - ref_epoch) / cos(radians(dec)) as ra_prop," + " "
                "dec + 1. / 3600000 * pmdec * (2000.0 - ref_epoch) as dec_prop," + " "
                f"DISTANCE(POINT({racol}, {deccol}), POINT(ra, dec)) * 3600. AS dist_arcsec" + " "
                "FROM user_dmistry.ast_table AS dmistry "
                "JOIN gaiadr3.gaia_source AS gaia "
                f"ON 1 = CONTAINS(POINT({racol}, {deccol}), CIRCLE(ra, dec, 2 / 3600.))"
            )

    # Query that accounts for proper motion of the targets in the cross match.
    query2 = ("SELECT *," + " "
                f"DISTANCE(POINT({racol}, {deccol}), POINT(ra_prop, dec_prop)) * 3600. AS dist_arcsec2" + " "
                "FROM ("
                f"SELECT *," + " "
                "ra + 1. / 3600000 * pmra * (2000.0 - ref_epoch) / cos(radians(dec)) as ra_prop," + " "
                "dec + 1. / 3600000 * pmdec * (2000.0 - ref_epoch) as dec_prop," + " "
                f"DISTANCE(POINT({racol}, {deccol}), POINT(ra, dec)) * 3600. AS dist_arcsec" + " "
                "FROM user_dmistry.ast_table AS dmistry" + " "
                "JOIN gaiadr3.gaia_source AS gaia" + " "
                f"ON 1 = CONTAINS(POINT({racol}, {deccol}), CIRCLE(ra, dec, 20 / 3600.))" + " "
                "OFFSET 0"
                ") AS subquery" + " "
                f"WHERE 1 = CONTAINS(POINT({racol}, {deccol}), CIRCLE(ra_prop, dec_prop, 2 / 3600.))"
                )

    ZTF_CVs = df.copy()
    # Run queries within Gaia archive and return dataframe.
    xmatch_basic = gaia_xmatch(ZTF_CVs, query, racol, deccol)
    xmatch_pm = gaia_xmatch(ZTF_CVs, query2, racol, deccol)

    # From the basic query generate a dataframe of those without pm measurements.
    xmatch_basic_nopm = xmatch_basic[xmatch_basic['pmra'].isnull()]

    # Add the above to the xmatch_pm dataframe.
    metadata_df = pd.concat([xmatch_basic_nopm, xmatch_pm]).reset_index(drop=True)

    # This dataframe contains all the cross matches, whether proper motion was present or not.
    metadata_df = metadata_df.drop(columns=['solution_id','DESIGNATION','source_id','random_index','designation'])

    # Attach ZTF object IDs to the metadata table.
    # We need to reset the index of the ZTF_CVs dataframe so it starts at 1.
    # This can be used to match with the ast_table_oid column from the metadata table.
    ZTF_CVs_cpy = ZTF_CVs.copy()
    ZTF_CVs_cpy.index = ZTF_CVs_cpy.index+1

    # Join the ZTF_CVs dataframe with the metadata dataframe on index and ast_table_oid.
    ztf_gaia_df = ZTF_CVs_cpy.merge(metadata_df, how='outer', left_index=True, right_on='ast_table_oid')


    # Identify and remove duplicates.
    # Load cross matched dataset.
    ztf_gaia_df = ztf_gaia_df.copy()

    # Create new column indicating duplicates.
    ztf_gaia_df = ztf_gaia_df.assign(duplicated = lambda row: ztf_gaia_df.duplicated(subset=objcol, keep=False))

    # Before removing duplicates, we need to order the them so we can keep the first.
    # We create a information rank column that ranks examples based on the data present.
    ztf_gaia_df['info_rank']= ztf_gaia_df.apply(lambda x: 1 if (~np.isnan(x['parallax'])) &
                                                                                                (~pd.isnull(x['pm'])) &
                                                                                                (~pd.isnull(x['bp_rp'])) &
                                                                                                (~pd.isnull(x['bp_g'])) &
                                                                                                (~pd.isnull(x['g_rp'])) else
                                                                                2 if (((pd.isnull(x['parallax']))|
                                                                                                (pd.isnull(x['pm']))) &
                                                                                                (~pd.isnull(x['bp_rp']))&
                                                                                                (~pd.isnull(x['bp_g']))&
                                                                                                (~pd.isnull(x['g_rp']))) else
                                                                                3 if ((~pd.isnull(x['parallax']))&
                                                                                                (~pd.isnull(x['pm']))&
                                                                                                ((pd.isnull(x['bp_rp']))|
                                                                                                (pd.isnull(x['bp_g']))|
                                                                                                (pd.isnull(x['g_rp'])))) else 4, axis=1)             
                                                                                                

    # Now we sort the dataframe by the information rank.
    # sorting by info_rank column.
    ztf_gaia_df.sort_values(['info_rank', 'dist_arcsec2', 'dist_arcsec'], inplace=True)
    ztf_gaia_df.reset_index(drop=True, inplace=True)

    # Print number of duplicates before drop.
    print(f"Number of entries that are duplicated: {len(ztf_gaia_df[ztf_gaia_df['duplicated']==True])}")

    # ztf_gaia_df[ztf_gaia_df['duplicated']==True][[objcol,'ra_x', 'dec_x', 'pm','bp_rp','bp_g','g_rp','parallax','dist_arcsec','dist_arcsec2','info_rank']].sort_values(by=objcol)

    # Now drop the duplicates based on objcol.
    ztf_gaia_df.drop_duplicates(subset=[objcol], inplace=True, keep='first')

    # Print number of duplicates after drop.
    print(f"Number of duplicate entries after removal: {len(ztf_gaia_df[ztf_gaia_df['duplicated']==True])}")

    # Display formerly duplicated examples.
    # ztf_gaia_df[ztf_gaia_df['duplicated']==True].sort_values(by=objcol)[[objcol,'ra_x', 'dec_x', 'pm','bp_rp','bp_g','g_rp','parallax','dist_arcsec','dist_arcsec2','info_rank']]

    # Final bit of housekeeping
    ztf_gaia_df.reset_index(drop=True, inplace=True)
    ztf_gaia_df = ztf_gaia_df.drop(columns=['ast_table_oid','raj2000','decj2000','ref_epoch','ra_y','dec_y',
                                            'dec_y','dist_arcsec','dist_arcsec2','duplicated','info_rank'])
    ztf_gaia_df.rename(columns={'ra_x':'ra','dec_x':'dec'}, inplace=True)

    # Add distance column
    ztf_gaia_df['distance'] = distance(ztf_gaia_df['parallax'])
    # Add absolute g magnitude column
    ztf_gaia_df['absmag_g'] = absmag(ztf_gaia_df['phot_g_mean_mag'], ztf_gaia_df['distance'])
    # Add absolute bp magnitude column
    ztf_gaia_df['absmag_bp'] = absmag(ztf_gaia_df['phot_bp_mean_mag'], ztf_gaia_df['distance'])
    # Add absolute rp magnitude column
    ztf_gaia_df['absmag_rp'] = absmag(ztf_gaia_df['phot_rp_mean_mag'], ztf_gaia_df['distance'])

    return ztf_gaia_df


def esa_archive(target_df, radius, table='gaiadr3.gaia_source', shortname='gaia_source_'):

    # Astropy table with coordinates
    alerts_table = Table([target_df['ra'], target_df['dec']], names=['RA', 'Dec'], meta={'meta':'CVs'})
    
    # Login to archive
    Gaia.login(user='dmistry', password='X641oVxh8kLj#')

    # Delete tables if present
    try:
        Gaia.delete_user_table("ast_table")
    except:
        pass
    try:
        Gaia.delete_user_table("xmatch_table")
    except:
        pass

    # Upload table
    Gaia.upload_table(upload_resource=alerts_table, table_name='ast_table')

    # Update table
    Gaia.update_user_table(table_name="user_dmistry.ast_table",
                        list_of_changes=[["ra", "flags", "ra"],
                                            ["dec","flags","dec"]])


    # Perform cross-match of coordinates with Gaia data release 3 data using cone search, collect results.
    # Radius is set to 1 originally but may increase to 2 arcseconds.
    full_qualified_table_name = 'user_dmistry.ast_table'
    xmatch_table_name = 'xmatch_table'
    Gaia.cross_match(full_qualified_table_name_a=full_qualified_table_name,
                    full_qualified_table_name_b=table,
                    results_table_name=xmatch_table_name, radius=radius, verbose=False)
    

    # Collect meta data for each of the successfully cross matched targets from Gaia DR3.

    # Cross matched targets.
    xmatch_table = 'user_dmistry.' + xmatch_table_name
    
    # Query. Separation is in degrees, need to convert to arcseconds so * 3600
    query = (f'SELECT c."separation"*3600 as dist, a.*, b.* FROM {table} AS a, '
            '%s AS b,'
            '%s AS c '
            f'WHERE (c.{shortname}_source_id = a.source_id AND '
            'c.ast_table_ast_table_oid = b.ast_table_oid)' %(full_qualified_table_name, xmatch_table))
    
    # Launch query
    job = Gaia.launch_job_async(query=query)
    results = job.get_results()
        
    # Place results into dataframe.  
    df_archive=results.to_pandas()

    # Clear tables and jobs from Gaia workspace and log out.
    Gaia.delete_user_table("ast_table")
    Gaia.delete_user_table("xmatch_table")
    jobs = [job for job in Gaia.list_async_jobs()]
    job_ids = [job.jobid for job in jobs]
    Gaia.remove_jobs(job_ids)
    Gaia.logout()

    return df_archive
