import streamlit as st
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
import json
import requests
from bs4 import BeautifulSoup
from typing import List
import xgboost as xgb
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# @st.cache_data
def get_data():
    # f = open('settings.json')
    # j = json.load(f)
    # API_KEY_FRED = j["API_KEY_FRED"]

    API_KEY_FRED = st.secrets["API_KEY_FRED"]
    
    def parse_release_dates(release_id: str) -> List[str]:
        release_dates_url = f'https://api.stlouisfed.org/fred/release/dates?release_id={release_id}&realtime_start=2015-01-01&include_release_dates_with_no_data=true&api_key={API_KEY_FRED}'
        r = requests.get(release_dates_url)
        text = r.text
        soup = BeautifulSoup(text, 'xml')
        dates = []
        for release_date_tag in soup.find_all('release_date', {'release_id': release_id}):
            dates.append(release_date_tag.text)
        return dates

    def parse_release_dates_obs(series_id: str) -> List[str]:
        obs_url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&realtime_start=2015-01-01&include_release_dates_with_no_data=true&api_key={API_KEY_FRED}'
        r = requests.get(obs_url)
        text = r.text
        soup = BeautifulSoup(text, 'xml')
        observations  = []
        for observation_tag in soup.find_all('observation'):
            date = observation_tag.get('date')
            value = observation_tag.get('value')
            observations.append((date, value))
        return observations

    econ_dfs = {}

    econ_tickers = [
        'WALCL',
        'NFCI',
        'WRESBAL'
    ]

    for et in tqdm(econ_tickers, desc='getting econ tickers'):
        # p = parse_release_dates_obs(et)
        # df = pd.DataFrame(columns = ['ds',et], data = p)
        df = pdr.get_data_fred(et)
        df.index = df.index.rename('ds')
        # df.index = pd.to_datetime(df.index.rename('ds')).dt.tz_localize(None)
        # df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        econ_dfs[et] = df

    # walcl = pd.DataFrame(columns = ['ds','WALCL'], data = p)
    # walcl['ds'] = pd.to_datetime(walcl['ds']).dt.tz_localize(None)

    # nfci = pd.DataFrame(columns = ['ds','NFCI'], data = p2)
    # nfci['ds'] = pd.to_datetime(nfci['ds']).dt.tz_localize(None)

    release_ids = [
        "10", # "Consumer Price Index"
        "46", # "Producer Price Index"
        "50", # "Employment Situation"
        "53", # "Gross Domestic Product"
        "103", # "Discount Rate Meeting Minutes"
        "180", # "Unemployment Insurance Weekly Claims Report"
        "194", # "ADP National Employment Report"
        "323" # "Trimmed Mean PCE Inflation Rate"
    ]

    release_names = [
        "CPI",
        "PPI",
        "NFP",
        "GDP",
        "FOMC",
        "UNEMP",
        "ADP",
        "PCE"
    ]

    releases = {}

    for rid, n in tqdm(zip(release_ids, release_names), total = len(release_ids), desc='Getting release dates'):
        releases[rid] = {}
        releases[rid]['dates'] = parse_release_dates(rid)
        releases[rid]['name'] = n 

    # Create a DF that has all dates with the name of the col as 1
    # Once merged on the main dataframe, days with econ events will be 1 or None. Fill NA with 0
    # This column serves as the true/false indicator of whether there was economic data released that day.
    for rid in tqdm(release_ids, desc='Making indicators'):
        releases[rid]['df'] = pd.DataFrame(
            index=releases[rid]['dates'],
            data={
            releases[rid]['name']: 1
            })
        releases[rid]['df'].index = pd.DatetimeIndex(releases[rid]['df'].index)
        # releases[rid]['df']['ds'] = pd.to_datetime(releases[rid]['df']['ds']).dt.tz_localize(None)
        # releases[rid]['df'] = releases[rid]['df'].set_index('ds')

    vix = yf.Ticker('^VIX')
    spx = yf.Ticker('^GSPC')

    prices_vix = vix.history(start='2020-01-01', interval='1d')
    prices_spx = spx.history(start='2020-01-01', interval='1d')
    prices_spx['index'] = [str(x).split()[0] for x in prices_spx.index]
    prices_spx['index'] = pd.to_datetime(prices_spx['index']).dt.date
    prices_spx.index = prices_spx['index']
    prices_spx = prices_spx.drop(columns='index')

    prices_vix['index'] = [str(x).split()[0] for x in prices_vix.index]
    prices_vix['index'] = pd.to_datetime(prices_vix['index']).dt.date
    prices_vix.index = prices_vix['index']
    prices_vix = prices_vix.drop(columns='index')

    data = prices_spx.merge(prices_vix[['Open','High','Low','Close']], left_index=True, right_index=True, suffixes=['','_VIX'])
    data.index = pd.DatetimeIndex(data.index)

    # Features
    data['PrevClose'] = data['Close'].shift(1)
    data['Perf5Day'] = data['Close'] > data['Close'].shift(5)
    data['Perf5Day_n1'] = data['Perf5Day'].shift(1)
    data['Perf5Day_n1'] = data['Perf5Day_n1'].astype(bool)
    data['GreenDay'] = (data['Close'] > data['PrevClose']) * 1
    data['RedDay'] = (data['Close'] <= data['PrevClose']) * 1

    data['VIX5Day'] = data['Close_VIX'] > data['Close_VIX'].shift(5)
    data['VIX5Day_n1'] = data['VIX5Day'].astype(bool)

    data['Range'] = data[['Open','High']].max(axis=1) - data[['Low','Open']].min(axis=1) # Current day range in points
    data['RangePct'] = data['Range'] / data['Close']
    data['VIXLevel'] = pd.qcut(data['Close_VIX'], 4)
    data['OHLC4_VIX'] = data[['Open_VIX','High_VIX','Low_VIX','Close_VIX']].mean(axis=1)
    data['OHLC4'] = data[['Open','High','Low','Close']].mean(axis=1)
    data['OHLC4_Trend'] = data['OHLC4'] > data['OHLC4'].shift(1)
    data['OHLC4_Trend_n1'] = data['OHLC4_Trend'].shift(1)
    data['OHLC4_Trend_n1'] = data['OHLC4_Trend_n1'].astype(float)
    data['OHLC4_Trend_n2'] = data['OHLC4_Trend'].shift(1)
    data['OHLC4_Trend_n2'] = data['OHLC4_Trend_n2'].astype(float)
    data['RangePct_n1'] = data['RangePct'].shift(1)
    data['RangePct_n2'] = data['RangePct'].shift(2)
    data['OHLC4_VIX_n1'] = data['OHLC4_VIX'].shift(1)
    data['OHLC4_VIX_n2'] = data['OHLC4_VIX'].shift(2)
    data['CurrentGap'] = (data['Open'] - data['PrevClose']) / data['PrevClose']
    data['CurrentGap'] = data['CurrentGap'].shift(-1)
    data['DayOfWeek'] = pd.to_datetime(data.index)
    data['DayOfWeek'] = data['DayOfWeek'].dt.day

    # Target -- the next day's low
    data['Target'] = (data['OHLC4'] / data['PrevClose']) - 1
    data['Target'] = data['Target'].shift(-1)
    # data['Target'] = data['RangePct'].shift(-1)

    # Target for clf -- whether tomorrow will close above or below today's close
    data['Target_clf'] = data['Close'] > data['PrevClose']
    data['Target_clf'] = data['Target_clf'].shift(-1)
    data['DayOfWeek'] = pd.to_datetime(data.index)
    data['Quarter'] = data['DayOfWeek'].dt.quarter
    data['DayOfWeek'] = data['DayOfWeek'].dt.weekday

    for rid in tqdm(release_ids, desc='Merging econ data'):
        # Get the name of the release
        n = releases[rid]['name']
        # Merge the corresponding DF of the release
        data = data.merge(releases[rid]['df'], how = 'left', left_index=True, right_index=True)
        # Create a column that shifts the value in the merged column up by 1
        data[f'{n}_shift'] = data[n].shift(-1)
        # Fill the rest with zeroes
        data[n] = data[n].fillna(0)
        data[f'{n}_shift'] = data[f'{n}_shift'].fillna(0)
        
    data['BigNewsDay'] = data[[x for x in data.columns if '_shift' in x]].max(axis=1)

    def cumul_sum(col):
        nums = []
        s = 0
        for x in col:
            if x == 1:
                s += 1
            elif x == 0:
                s = 0
            nums.append(s)
        return nums

    consec_green = cumul_sum(data['GreenDay'].values)
    consec_red = cumul_sum(data['RedDay'].values)

    data['DaysGreen'] = consec_green
    data['DaysRed'] = consec_red

    final_row = data.index[-2]

    df_final = data.loc[:final_row,
    [
        'BigNewsDay',
        'Quarter',
        'Perf5Day',
        'Perf5Day_n1',
        'DaysGreen',
        'DaysRed',
        # 'OHLC4_Trend',
        # 'OHLC4_Trend_n1',
        # 'OHLC4_Trend_n2',
        # 'VIX5Day',
        # 'VIX5Day_n1',
        'CurrentGap',
        'RangePct',
        'RangePct_n1',
        'RangePct_n2',
        'OHLC4_VIX',
        'OHLC4_VIX_n1',
        'OHLC4_VIX_n2',
        'Target',
        'Target_clf'
        ]]
    df_final = df_final.dropna(subset=['Target','Target_clf','Perf5Day_n1'])
    return data, df_final, final_row

st.set_page_config(
    page_title="Gameday Model for $SPX",
    page_icon="🎮"
)

st.title('🎮 Gameday Model for $SPX')
st.markdown('**PLEASE NOTE:** Model should be run at or after market open.')

if st.button("CLEAR IT"):
    st.cache_data.clear()

if st.button('RUN IT'):
    with st.spinner('Loading data...'):
        data, df_final, final_row = get_data()
    st.success("✅ Historical data")

    def walk_forward_validation(df, target_column, num_training_rows, num_periods):
        
        # Create an XGBRegressor model
        # model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state = 42)
        model = LinearRegression()
        
        overall_results = []
        # Iterate over the rows in the DataFrame, one step at a time
        for i in tqdm(range(num_training_rows, df.shape[0] - num_periods + 1)):
            # Split the data into training and test sets
            X_train = df.drop(target_column, axis=1).iloc[:i]
            y_train = df[target_column].iloc[:i]
            X_test = df.drop(target_column, axis=1).iloc[i:i+num_periods]
            y_test = df[target_column].iloc[i:i+num_periods]
            
            # Fit the model to the training data
            model.fit(X_train, y_train)
            
            # Make a prediction on the test data
            predictions = model.predict(X_test)
            
            # Create a DataFrame to store the true and predicted values
            result_df = pd.DataFrame({'True': y_test, 'Predicted': predictions}, index=y_test.index)
            
            overall_results.append(result_df)

        df_results = pd.concat(overall_results)
        # model.save_model('model.bin')
        # Return the true and predicted values, and fitted model
        return df_results, model

    def walk_forward_validation_clf(df, target_column, num_training_rows, num_periods):
        
        # Create an XGBRegressor model
        model = xgb.XGBClassifier(n_estimators=10, random_state = 42)
        
        overall_results = []
        # Iterate over the rows in the DataFrame, one step at a time
        for i in tqdm(range(num_training_rows, df.shape[0] - num_periods + 1)):
            # Split the data into training and test sets
            X_train = df.drop(target_column, axis=1).iloc[:i]
            y_train = df[target_column].iloc[:i]
            X_test = df.drop(target_column, axis=1).iloc[i:i+num_periods]
            y_test = df[target_column].iloc[i:i+num_periods]
            
            # Fit the model to the training data
            model.fit(X_train, y_train)
            
            # Make a prediction on the test data
            predictions = model.predict_proba(X_test)[:,-1]
            
            # Create a DataFrame to store the true and predicted values
            result_df = pd.DataFrame({'True': y_test, 'Predicted': predictions}, index=y_test.index)
            
            overall_results.append(result_df)

        df_results = pd.concat(overall_results)
        # model.save_model('model_clf.bin')
        # Return the true and predicted values, and fitted model
        return df_results, model

    with st.spinner("Training models..."):
        def train_models():
            res, xgbr = walk_forward_validation(df_final.drop(columns=['Target_clf']).dropna(), 'Target', 100, 1)
            res1, xgbc = walk_forward_validation_clf(df_final.drop(columns=['Target']).dropna(), 'Target_clf', 100, 1)
            return res, xgbr, res1, xgbc
        res, xgbr, res1, xgbc = train_models()
    st.success("✅ Models trained")

    with st.spinner("Getting new prediction..."):

        # Get last row
        new_pred = data.loc[final_row, ['BigNewsDay',
            'Quarter',
            'Perf5Day',
            'Perf5Day_n1',    
            'DaysGreen',    
            'DaysRed',    
            # 'OHLC4_Trend',    
            # 'OHLC4_Trend_n1',    
            # 'OHLC4_Trend_n2',    
            # 'VIX5Day',
            # 'VIX5Day_n1',
            'CurrentGap',
            'RangePct',
            'RangePct_n1',
            'RangePct_n2',
            'OHLC4_VIX',
            'OHLC4_VIX_n1',
            'OHLC4_VIX_n2']]

        new_pred = pd.DataFrame(new_pred).T
        # new_pred_show = pd.DataFrame(index=[new_pred.columns], columns=[new_pred.index], data=[[v] for v in new_pred.values])

        new_pred['BigNewsDay'] = new_pred['BigNewsDay'].astype(float)
        new_pred['Quarter'] = new_pred['Quarter'].astype(int)
        new_pred['Perf5Day'] = new_pred['Perf5Day'].astype(bool)
        new_pred['Perf5Day_n1'] = new_pred['Perf5Day_n1'].astype(bool)
        new_pred['DaysGreen'] = new_pred['DaysGreen'].astype(float)
        new_pred['DaysRed'] = new_pred['DaysRed'].astype(float)
        # new_pred['OHLC4_Trend'] = new_pred['OHLC4_Trend'].astype(float)
        # new_pred['OHLC4_Trend_n1'] = new_pred['OHLC4_Trend_n1'].astype(float)
        # new_pred['OHLC4_Trend_n2'] = new_pred['OHLC4_Trend_n2'].astype(float)
        # new_pred['VIX5Day'] = new_pred['VIX5Day'].astype(bool)
        # new_pred['VIX5Day_n1'] = new_pred['VIX5Day_n1'].astype(bool)
        new_pred['CurrentGap'] = new_pred['CurrentGap'].astype(float)
        new_pred['RangePct'] = new_pred['RangePct'].astype(float)
        new_pred['RangePct_n1'] = new_pred['RangePct_n1'].astype(float)
        new_pred['RangePct_n2'] = new_pred['RangePct_n2'].astype(float)
        new_pred['OHLC4_VIX'] = new_pred['OHLC4_VIX'].astype(float)
        new_pred['OHLC4_VIX_n1'] = new_pred['OHLC4_VIX_n1'].astype(float)
        new_pred['OHLC4_VIX_n2'] = new_pred['OHLC4_VIX_n2'].astype(float)

    st.success("✅ Data for new prediction")
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "✨ New Data", "🗄 Historical"])

    clf_proba = xgbc.predict_proba(new_pred)[:,-1]
    clf_pred = clf_proba > 0.5
    reg_pred = xgbr.predict(new_pred) > 0

    results = pd.DataFrame(index=[
        'CLF Proba',
        'CLF Pred',
        'REG Pred'
    ], data = [clf_proba, clf_pred, reg_pred])

    results.columns = ['Outputs']

    # st.subheader('New Prediction')

    df_ensemble = res[['Predicted']].merge(res1[['Predicted','True']], left_index=True, right_index=True)
    df_ensemble = df_ensemble.merge(data[['PrevClose','Open','High','Low','Close']], left_index=True, right_index=True)
    df_ensemble['ActualDirection'] = df_ensemble['Close'] > df_ensemble['PrevClose']
    df_ensemble['OpenDirection'] = df_ensemble['Close'] > df_ensemble['Open']
    df_ensemble['ActualDirection'] = df_ensemble['ActualDirection'].shift(-1)

    df_ensemble.columns = ['RegModelOut','ClfModelOut','Target','PrevClose','Open','High','Low','Close','ActualDirection','OpenDirection']
    df_ensemble['RegModelOut'] = df_ensemble['RegModelOut'] > 0
    df_ensemble['RegModelOut_n1'] = df_ensemble['RegModelOut'].shift(1)
    df_ensemble['ClfModelOut_n1'] = df_ensemble['ClfModelOut'].shift(1)
    df_ensemble['RegModelOut_n2'] = df_ensemble['RegModelOut'].shift(2)
    df_ensemble['ClfModelOut_n2'] = df_ensemble['ClfModelOut'].shift(2)

    df_ensemble = df_ensemble.dropna(subset=['ClfModelOut_n2'])
    df_ensemble['ClfModelOut_tf'] = df_ensemble['ClfModelOut'] > 0.5
    df_probas = df_ensemble.groupby(['RegModelOut','ClfModelOut_tf'])[['ActualDirection','OpenDirection']].mean()

    tab1.subheader('Preds and Probabilities')
    tab1.write(results)
    tab1.write(df_probas)

    tab2.subheader('Latest Data for Pred')
    tab2.write(new_pred)

    tab3.subheader('Historical Data')
    tab3.write(df_final)


# The only variable you can play with as the other ones are historical
# new_pred.loc[:,'CurrentGap'] = -0.01 / 100
# new_pred.loc[:,'BigNewsDay'] = 0

# st.subheader('Subset')
# st.write(data.iloc[-1])

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)