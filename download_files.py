#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:28:47 2026

@author: alam
"""

#!/usr/bin/env python3

import urllib.request
#from gnss_utils import gpsweek_from_doy_and_year, date_from_doy

infos = {"IBGE" : 'https://geoftp.ibge.gov.br/informacoes_sobre_posicionamento_geodesico/rbmc/dados', 
         "IGS": 'https://igs.bkg.bund.de/root_ftp/IGS/products/'}

regions = {"region1": ['alar', 'bair', 'brft', 'ceeu', 
                       'ceft', 'cesb', 'crat', 'pbcg', 
                       'pbjp', 'peaf', 'pepe', 'recf',
                       'rnmo', 'rnna', 'seaj']}


url='https://geoftp.ibge.gov.br/informacoes_sobre_posicionamento_geodesico/rbmc/dados/'#rinex_url(2025,355)
ano='2025'
doy='354'
fnm='pbjp'+doy+'1.zip'

import os
cwd = os.getcwd()

LOCAL_ZIP_FILENAME=fnm
DATA_URL=url+ano+'/'+doy+'/'+fnm


urllib.request.urlretrieve(DATA_URL,LOCAL_ZIP_FILENAME)



# def main():
#     """Fetch the Iris dataset from UCI."""
#     urllib.request.urlretrieve(
#         IRIS_DATA_URL,
#         LOCAL_ZIP_FILENAME
#     )
#     print(f"Downloaded {LOCAL_ZIP_FILENAME}")

# if __name__ == "__main__":
#     main()
