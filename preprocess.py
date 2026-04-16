import pandas as pd
import os

df1 = pd.read_csv('data/raw/phishing.csv')
df2 = pd.read_csv('data/raw/dataset3.csv')
df3 = pd.read_csv('data/raw/Phishing_Websites_Data.csv')

# Manually map columns to a common name
# Format: 'common_name': (df1_col, df2_col, df3_col)
column_map = {
    'using_ip':                 ('UsingIP',              'ip',                        'having_IP_Address'),
    'short_url':                ('ShortURL',             'shortening_service',         'Shortining_Service'),
    'symbol_at':                ('Symbol@',              'nb_at',                     'having_At_Symbol'),
    'redirecting':              ('Redirecting//',        'nb_redirection',            'double_slash_redirecting'),
    'prefix_suffix':            ('PrefixSuffix-',        'prefix_suffix',             'Prefix_Suffix'),
    'sub_domains':              ('SubDomains',           'nb_subdomains',             'having_Sub_Domain'),
    'domain_reg_len':           ('DomainRegLen',         'domain_registration_length','Domain_registeration_length'),
    'favicon':                  ('Favicon',              'external_favicon',          'Favicon'),
    'port':                     ('NonStdPort',           'port',                      'port'),
    'https_token':              ('HTTPSDomainURL',       'https_token',               'HTTPS_token'),
    'request_url':              ('RequestURL',           'ratio_extHyperlinks',       'Request_URL'),
    'anchor_url':               ('AnchorURL',            'ratio_nullHyperlinks',      'URL_of_Anchor'),
    'links_in_tags':            ('LinksInScriptTags',    'links_in_tags',             'Links_in_tags'),
    'sfh':                      ('ServerFormHandler',    'sfh',                       'SFH'),
    'info_email':               ('InfoEmail',            'submit_email',              'Submitting_to_email'),
    'abnormal_url':             ('AbnormalURL',          'statistical_report',        'Abnormal_URL'),
    'website_forwarding':       ('WebsiteForwarding',    'nb_external_redirection',   'Redirect'),
    'on_mouseover':             ('StatusBarCust',        'onmouseover',               'on_mouseover'),
    'right_click':              ('DisableRightClick',    'right_clic',                'RightClick'),
    'popup_window':             ('UsingPopupWindow',     'popup_window',              'popUpWidnow'),
    'iframe':                   ('IframeRedirection',    'iframe',                    'Iframe'),
    'age_of_domain':            ('AgeofDomain',         'domain_age',                'age_of_domain'),
    'dns_record':               ('DNSRecording',         'dns_record',                'DNSRecord'),
    'web_traffic':              ('WebsiteTraffic',       'web_traffic',               'web_traffic'),
    'page_rank':                ('PageRank',             'page_rank',                 'Page_Rank'),
    'google_index':             ('GoogleIndex',          'google_index',              'Google_Index'),
    'links_pointing_to_page':   ('LinksPointingToPage',  'nb_hyperlinks',             'Links_pointing_to_page'),
    'stats_report':             ('StatsReport',          'suspecious_tld',            'Statistical_report'),
}

# Build aligned dataframes
def extract_cols(df, col_index, col_map):
    extracted = {}
    for common_name, sources in col_map.items():
        extracted[common_name] = df[sources[col_index]]
    return pd.DataFrame(extracted)

aligned1 = extract_cols(df1, 0, column_map)
aligned2 = extract_cols(df2, 1, column_map)
aligned3 = extract_cols(df3, 2, column_map)

# Standardize labels → 1=Phishing, 0=Legitimate
aligned1['label'] = df1['class'].apply(lambda x: 1 if x == -1 else 0)
aligned2['label'] = df2['status'].apply(lambda x: 1 if x == 'phishing' else 0)
aligned3['label'] = df3['Result'].apply(lambda x: 1 if x == -1 else 0)

merged = pd.concat([aligned1, aligned2, aligned3], ignore_index=True)
print(f"Merged shape: {merged.shape}")
print(f"\nLabel distribution:\n{merged['label'].value_counts()}")

# Clean the data
merged.drop_duplicates(inplace=True)
merged.fillna(merged.median(numeric_only=True), inplace=True)

print(f"\nMissing values after cleaning:\n{merged.isnull().sum().sum()} total")
print(f"Final shape: {merged.shape}")

os.makedirs('data/processed', exist_ok=True)
merged.to_csv('data/processed/processed_data.csv', index=False)
print("\nSaved to data/processed/processed_data.csv")
