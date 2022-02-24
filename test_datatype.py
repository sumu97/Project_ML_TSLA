""" Data Crawling """
# 목표 주식, 시작일, 종료일 설정
from datetime import datetime, timedelta
target = 'TSLA'
daily = True
start_date = '2022-01-01'
end_date = str(datetime.today().strftime("%Y-%m-%d"))
time_range = start_date + " " + end_date

# 구글트렌드 검색 데이터 크롤링 (Pytrend API)
import pandas as pd
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US')
kw_list = [target]
pytrends.build_payload(kw_list, cat=0, timeframe = time_range, geo='', gprop='')
df_a = pytrends.interest_over_time()

# 주식 데이터 크롤링 (Finance Data Reader API)
import FinanceDataReader as fdr
df_ndq = fdr.StockListing('NASDAQ')
df_b = fdr.DataReader(target, start_date, end_date)

""" Data EDA """
# 검색 데이터 정제
df_a1 = df_a.reset_index(drop = False)
df_a1 = df_a1[['date',target]]
df_a1.columns = ['Date',target]

# 주식 데이터 정제
if daily == False:
    df_b1 = df_b.resample(rule = 'W').last()
else:
    df_b1 = df_b
df_b1 = df_b1.reset_index(drop = True)
df_b1 = df_b1#[:-1]

# Change 값을 %로 표기하도록 변경
temp_list = []
for i in df_b1['Change']:
  temp_list.append(i*100)
df_b1['Change'] = temp_list

# Volume 값 000 제거
temp_list = []
for i in df_b1['Volume']:
  temp_list.append(i/1000)
df_b1['Volume'] = temp_list
df_b1

# 검색, 주가 데이터 합치기
df = pd.concat([df_a1,df_b1], axis = 1, ignore_index = True)

# 컬럼명 재설정
df.columns = ['Date','Search','Price','Open','High','Low','Vol','Change']

# Datetime64 -> datetime.datetime
import pandas as pd
import numpy as np
temp_list = []
for i in df['Date']:
    temp_list.append(datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').date())
df['Date'] = temp_list

print(df)
print(df.columns)
print(df.dtypes)
print(df.shape)