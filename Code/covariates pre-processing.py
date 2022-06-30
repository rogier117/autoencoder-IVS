from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal  # CHANGE ALL USFEDERALHOLIDAY THINGS WITH THIS ONE!!
from functools import reduce

SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')

# Import covariates

# Function to let all dates correspond to SPX
def match_dates(good, good_colname, new, new_colname):
    nyse = mcal.get_calendar('NYSE')
    first_date = nyse.valid_days(start_date=good[good_colname][0] - timedelta(days=40), end_date=good[good_colname][0]).tz_localize(
        None)[-22]
    pre_sample = nyse.valid_days(start_date=first_date, end_date=good[good_colname][0]).tz_localize(None)
    new = new[new[new_colname] >= pre_sample[0]].reset_index(drop=True)
    keep = np.full(new.shape[0], False)
    for _ in range(new.shape[0]):
        if new[new_colname][_] in pre_sample or new[new_colname][_] in good[good_colname].unique():
            keep[_] = True
    new = new[keep].reset_index(drop=True)
    check = good[good_colname].isin(new[new_colname])
    check = np.array([not elem for elem in check]).nonzero()[0]
    if new.shape[0] < good.shape[0] + 21:
        positions = np.arange(new.shape[0], good.shape[0] + 21)
        for _ in range(good.shape[0] + 21 - new.shape[0]):
            new.loc[positions[_]] = float("nan")
            new.at[positions[_], new_colname] = good[good_colname][check[_]]
    new[new_colname] = pd.to_datetime(new[new_colname], format='%Y-%m-%d')
    new = new.sort_values(by=new_colname, ignore_index=True)
    return new


# Volatility Index (VIX)
VIX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\VIX_History.csv')
VIX = VIX.rename(columns={'DATE': 'Date', 'CLOSE': 'VIX'})
VIX = VIX[['Date','VIX']]
VIX['Date'] = pd.to_datetime(VIX['Date'], format='%m/%d/%Y')
VIX = match_dates(good=SPX, good_colname='Date', new=VIX, new_colname='Date')
VIX['VIX'] = VIX['VIX'].interpolate(method='linear', axis=0)

# Left Tail Volatility (LTV) (ONLY GOES TO END OF 2019!!) (THEREFORE PROBABLY WON'T INCLUDE)
LTV = pd.read_html(r'D:\Master Thesis\autoencoder-IVS\Data\LTV.xls', header=0, decimal=',', thousands='.')
LTV = LTV[0]
LTV = LTV.rename(columns={'Index': 'LTV'})
LTV['Date'] = pd.to_datetime(LTV['Date'], format='%Y-%m-%d')
LTV = match_dates(good=SPX, good_colname='Date', new=LTV, new_colname='Date')
LTV['LTV'] = LTV['LTV'].interpolate(method='linear', axis=0)

# Left Tail Porbability (LTP) (ONLY GOES TO END OF 2019!!) (THEREFORE PROBABLY WON'T INCLUDE)
LTP = pd.read_html(r'D:\Master Thesis\autoencoder-IVS\Data\LTP.xls', header=0, decimal=',', thousands='.')
LTP = LTP[0]
LTP = LTP.rename(columns={'Index': 'LTP'})
LTP['Date'] = pd.to_datetime(LTP['Date'], format='%Y-%m-%d')
LTP = match_dates(good=SPX, good_colname='Date', new=LTP, new_colname='Date')
LTP['LTP'] = LTP['LTP'].interpolate(method='linear', axis=0)

# Realized Volatility (RVOL)
RVOL = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\RVOL.csv')
RVOL = RVOL.rename(columns={'Unnamed: 0': 'Date', 'rv5': 'RVOL'})
RVOL = RVOL[RVOL['Symbol'] == '.SPX'].reset_index(drop=True)
RVOL = RVOL[['Date', 'RVOL']]
# Remove Timezone as it is of no importance
for _ in range(RVOL.shape[0]):
    RVOL.at[_, 'Date'] = RVOL.Date.iloc[_][:-6]
RVOL['Date'] = pd.to_datetime(RVOL['Date'], format="%Y-%m-%d")
RVOL = match_dates(good=SPX, good_colname='Date', new=RVOL, new_colname='Date')
RVOL['RVOL'] = RVOL['RVOL'].interpolate(method='linear', axis=0)

# Economic Policy Uncertainty (EPU) (monthly)
EPU = pd.read_excel(r'D:\Master Thesis\autoencoder-IVS\Data\EPU.xlsx')
EPU = EPU.rename(columns={'Three_Component_Index': 'EPU'})
EPU = EPU[:-1]
# EPU is shifted by one month because the information about a certain month is only available at the end of it
EPU['EPU'] = EPU['EPU'].shift(1)
EPU = EPU.iloc[1:, :]
EPU = EPU.astype({'Month': 'int32'})
EPU['Year'] = EPU.Year.astype(str)
EPU['Month'] = EPU.Month.astype(str)
EPU['Date'] = EPU.Year + '-' + EPU.Month + '-1'
EPU['Date'] = pd.to_datetime(EPU['Date'], format='%Y-%m-%d')
EPU = EPU[['Date', 'EPU']].reset_index(drop=True)
nyse = mcal.get_calendar('NYSE')
dates = nyse.valid_days(start_date=EPU.Date[0], end_date=EPU.Date[EPU.shape[0] - 1]).tz_localize(None)
dates = list(set(dates) - set(EPU.Date))
for _ in range(len(dates)):
    el = EPU.shape[0]
    EPU.loc[el] = float("nan")
    EPU.at[el, 'Date'] = dates[_]
EPU = EPU.sort_values(by='Date').reset_index(drop=True)
EPU = EPU.fillna(method='ffill')
EPU = match_dates(good=SPX, good_colname='Date', new=EPU, new_colname='Date')
EPU['EPU'] = EPU['EPU'].interpolate(method='linear', axis=0)

# US News Index (USNI)
USNI = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\USNI.csv')
USNI = USNI.rename(columns={'daily_policy_index': 'USNI'})
USNI['year'] = USNI.year.astype(str)
USNI['month'] = USNI.month.astype(str)
USNI['day'] = USNI.day.astype(str)
USNI['Date'] = USNI.year + '-' + USNI.month + '-' + USNI.day
USNI['Date'] = pd.to_datetime(USNI['Date'], format='%Y-%m-%d')
# USNI is shifted by one day because the information about a certain day is only available at the end of it
USNI['USNI'] = USNI['USNI'].shift(1)
USNI = USNI.iloc[1:, :]
USNI = USNI[['Date', 'USNI']].reset_index(drop=True)
USNI = match_dates(good=SPX, good_colname='Date', new=USNI, new_colname='Date')
USNI['USNI'] = USNI['USNI'].interpolate(method='linear', axis=0)

# Aruoba, Diebold and Scotti (2009) (ADS) business conditions index
ADS = pd.read_excel(r'D:\Master Thesis\autoencoder-IVS\Data\ADS.xlsx')
ADS = ADS.rename(columns={'Unnamed: 0': 'Date', 'ADS_Index': 'ADS'})
ADS['Date'] = pd.to_datetime(ADS['Date'], format='%Y:%m:%d')
ADS = match_dates(good=SPX, good_colname='Date', new=ADS, new_colname='Date')
ADS['ADS'] = ADS['ADS'].interpolate(method='linear', axis=0)

# Term Spread first difference
TMS = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\TMS.csv')
TMS = TMS.rename(columns={'DATE': 'Date', 'T10Y2Y': 'TMS'})
TMS['Date'] = pd.to_datetime(TMS['Date'], format='%Y-%m-%d')
TMS['TMS'] = pd.to_numeric(TMS['TMS'], errors='coerce')
TMS['TMS'] = TMS['TMS'].interpolate(method='linear', axis=0)
TMS['TMS'] = TMS.TMS.diff(1)
TMS = match_dates(good=SPX, good_colname='Date', new=TMS, new_colname='Date')
TMS['TMS'] = TMS['TMS'].interpolate(method='linear', axis=0)

# Credit Spread first difference
CRS = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\CRS.csv')
CRS = CRS.rename(columns={'DATE': 'Date', 'BAMLH0A0HYM2': 'CRS'})
CRS['Date'] = pd.to_datetime(CRS['Date'], format='%Y-%m-%d')
CRS['CRS'] = pd.to_numeric(CRS['CRS'], errors='coerce')
CRS['CRS'] = CRS['CRS'].interpolate(method='linear', axis=0)
CRS['CRS'] = CRS.CRS.diff(1)
CRS = match_dates(good=SPX, good_colname='Date', new=CRS, new_colname='Date')
CRS['CRS'] = CRS['CRS'].interpolate(method='linear', axis=0)

# Federal Funds Effective Rate (FFER) (from FRED)
FFER = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\FFER.csv')
FFER = FFER.rename(columns={'DATE': 'Date', 'DFF': 'FFER'})
FFER['Date'] = pd.to_datetime(FFER['Date'], format='%Y-%m-%d')
FFER = match_dates(good=SPX, good_colname='Date', new=FFER, new_colname='Date')
FFER['FFER'] = FFER['FFER'].interpolate(method='linear', axis=0)

# Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity (US10YMY) (from FRED)
US10YMY = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\US10YMY.csv')
US10YMY = US10YMY.rename(columns={'DATE': 'Date', 'DGS10': 'US10YMY'})
US10YMY['Date'] = pd.to_datetime(US10YMY['Date'], format='%Y-%m-%d')
US10YMY['US10YMY'] = pd.to_numeric(US10YMY['US10YMY'], errors='coerce')
US10YMY = match_dates(good=SPX, good_colname='Date', new=US10YMY, new_colname='Date')
US10YMY['US10YMY'] = US10YMY['US10YMY'].interpolate(method='linear', axis=0)

# Monthly US Inflation (USCPI) (from OECD)
USCPI = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\USCPI.csv')
USCPI = USCPI.rename(columns={'TIME': 'Date', 'Value': 'USCPI'})
USCPI = USCPI[['Date', 'USCPI']]
USCPI['Date'] = pd.to_datetime(USCPI['Date'], format='%Y-%m')
# USCPI is shifted by one day because the information about a certain day is only available at the end of it
USCPI['USCPI'] = USCPI['USCPI'].shift(1)
nyse = mcal.get_calendar('NYSE')
dates = nyse.valid_days(start_date=USCPI.Date[0], end_date=USCPI.Date[USCPI.shape[0] - 1]).tz_localize(None)
dates = list(set(dates) - set(USCPI.Date))
for _ in range(len(dates)):
    el = USCPI.shape[0]
    USCPI.loc[el] = float("nan")
    USCPI.at[el, 'Date'] = dates[_]
USCPI = USCPI.sort_values(by='Date').reset_index(drop=True)
USCPI = USCPI.fillna(method='ffill')
USCPI = match_dates(good=SPX, good_colname='Date', new=USCPI, new_colname='Date')
USCPI['USCPI'] = USCPI['USCPI'].interpolate(method='linear', axis=0)

# Real GDP growth Brave-Butters-Kelley (GDPBBK) (monthly)
GDPBBK = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\GDPBBK.csv')
GDPBBK = GDPBBK.rename(columns={'DATE': 'Date', 'BBKMGDP': 'GDPBBK'})
GDPBBK['Date'] = pd.to_datetime(GDPBBK['Date'], format='%Y-%m-%d')
# GDPBBK is shifted by one day because the information about a certain day is only available at the end of it
GDPBBK['GDPBBK'] = GDPBBK['GDPBBK'].shift(1)
nyse = mcal.get_calendar('NYSE')
dates = nyse.valid_days(start_date=GDPBBK.Date[0], end_date=GDPBBK.Date[GDPBBK.shape[0] - 1]).tz_localize(None)
dates = list(set(dates) - set(GDPBBK.Date))
for _ in range(len(dates)):
    el = GDPBBK.shape[0]
    GDPBBK.loc[el] = float("nan")
    GDPBBK.at[el, 'Date'] = dates[_]
GDPBBK = GDPBBK.sort_values(by='Date').reset_index(drop=True)
GDPBBK = GDPBBK.fillna(method='ffill')
GDPBBK = match_dates(good=SPX, good_colname='Date', new=GDPBBK, new_colname='Date')
GDPBBK['GDPBBK'] = GDPBBK['GDPBBK'].interpolate(method='linear', axis=0)

# Last month's (21 trading days) total return SPX (SPXM)
SPX2001 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2001.csv', decimal='.', thousands=',')
SPX2001['Date'] = pd.to_datetime(SPX2001['Date'], format='%m/%d/%Y')
SPX2001 = SPX2001.sort_values(by='Date').reset_index(drop=True)
SPXM = pd.concat([SPX2001, SPX]).reset_index(drop=True)
SPXM['SPXM'] = (SPXM.Close[21:].reset_index(drop=True) - SPXM.Close[:-21].reset_index(drop=True))/SPXM.Close[:-21].reset_index(drop=True)
SPXM['SPXM'] = SPXM['SPXM'].shift(21)
SPXM = SPXM[['Date', 'SPXM']]
SPXM = match_dates(good=SPX, good_colname='Date', new=SPXM, new_colname='Date')
SPXM['SPXM'] = SPXM['SPXM'].interpolate(method='linear', axis=0)


data_frames = [ADS,CRS,EPU,FFER,GDPBBK,LTP,LTV,RVOL,SPXM,TMS,US10YMY,USCPI,USNI,VIX]
covariates = reduce(lambda left, right: pd.merge(left,right,on=['Date'],
                                            how='outer'), data_frames)
# covariates = pd.merge(ADS, [CRS,EPU,FFER,GDPBBK,LTP,LTV,RVOL,SPXM,TMS,US10YMY,USCPI,USNI,VIX], on='Date')
#object dtype for: VIX, RVOL, LTV, LTP