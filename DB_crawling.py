""" Data Crawling """
# 목표 주식, 시작일, 종료일 설정
from datetime import datetime, timedelta
target = 'TSLA'
start_date = '2017-01-01'
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
df_b1 = df_b.resample(rule = 'W').last()
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

"""
# 자료형 변경
import datetime
import numpy as np
for col in df.columns:
  temp_list = []
  for i in df[col]:
    try: # datetime 우선 처리
      temp = datetime.datetime.strptime(str(i), '%Y-%m-%d').date()
      temp_list.append(temp) # datetime 로 변경된 자료형을 list에 저장
    except:
      try: # 데이터 조정
        temp = i.replace('M','')
        temp = temp.replace(',','')
        temp = temp.replace('B','1000')
        temp = temp.replace('%','')
        temp = float(temp)
        temp_list.append(temp)
      except: # 기존 int 형 데이터
          temp_list.append(i)
  df[col] = temp_list # 변경된 리스트를 col 에 저장
"""

""" Feature Engineering """
# 이동평균선 함수(몇 주 치, 비교할 특성) / Feature
def moving_avg(much,target_feature): 
  target_list = []
  for i in range(len(target_feature)):
    feature_list = []
    try:
      for l in range(much+1):
        feature_list.append(target_feature[i-l])
      target_list.append(round(np.mean(feature_list),2))
    except:
      target_list.append(np.nan)
  return target_list # 이동평균선 List 반환

# 가격변화 함수(몇 주 뒤, 비교할 특성) / Label, Target Data
def price_go(much,target_feature): 
  week = []
  for i in range(len(target_feature)):
    try:
      if target_feature[i] < target_feature[i+much]:
        week.append(1)
      elif target_feature[i] >= target_feature[i+much]:
        week.append(0)
    except:
      week.append(np.nan)
  return week # 몇 주 뒤 커지면 1, 아니면 0, 모르면 Nan 반환

# 이동평균선 함수 적용
df['avg_one'] = moving_avg(1, df.Price)
df['avg_thr'] = moving_avg(3, df.Price)
df['avg_fiv'] = moving_avg(5, df.Price)
#df['avg_ten'] = moving_avg(10, df.Price)

# 가격변화 함수 적용
df['one'] = price_go(1,df.Price)
df['two'] = price_go(2,df.Price)
df['thr'] = price_go(3,df.Price)
df['for'] = price_go(4,df.Price)
df['fiv'] = price_go(5,df.Price)
df['six'] = price_go(6,df.Price)
df['sev'] = price_go(7,df.Price)
df['eig'] = price_go(8,df.Price)
df['nin'] = price_go(9,df.Price)
df['ten'] = price_go(10,df.Price)

#print(df)

""" Store Data """
# Pickle화 해서 저장
import joblib
import os
base = os.path.dirname(os.path.abspath(__file__))
joblib.dump(df, os.path.join(base,'data',target))