# Pickle Test
import os
import joblib
base = '/Users/yutaejun/Desktop/code/p1'
pipe_1 = joblib.load(os.path.join(base,'model','pipe_1'))
pipe_3 = joblib.load(os.path.join(base,'model','pipe_3'))
pipe_5 = joblib.load(os.path.join(base,'model','pipe_5'))
pipe_10 = joblib.load(os.path.join(base,'model','pipe_10'))
# picked base dataframe
df = joblib.load(os.path.join(base,'model','df'))

from sklearn.metrics import accuracy_score
import datetime
import numpy as np
def test_case(start_date, end_date, df): # 테스트 함수 : 시작날짜, 종료날짜, 데이터프레임
  condition = (datetime.datetime.strptime(end_date, '%Y-%m-%d').date() >= df['Date']) & (df['Date'] >= datetime.datetime.strptime(start_date, '%Y-%m-%d').date())
  features = ['Search','Price','Open','High','Low','Vol','Change','avg_one','avg_thr','avg_fiv','avg_ten']
  temp_df = df[condition][features]
  pred_1 = pipe_1.predict(temp_df)
  pred_3 = pipe_3.predict(temp_df)
  pred_5 = pipe_5.predict(temp_df)
  pred_10 = pipe_10.predict(temp_df)
  real_1 = df[condition]['one']
  real_3 = df[condition]['thr']
  real_5 = df[condition]['fiv']
  real_10 = df[condition]['one']
  pred = np.concatenate((pred_1, pred_3, pred_5, pred_10), axis=0)
  real = np.concatenate((real_1, real_3, real_5, real_10), axis=0)
  print(start_date,"~",end_date,"예측 정확도 :",accuracy_score(real, pred) * 100,"%")

#test_case('2017-1-1','2017-12-31',df)
test_case('2018-1-1','2018-12-31',df)
#test_case('2019-1-1','2019-12-31',df)
#test_case('2020-1-1','2020-12-31',df)
#test_case('2021-1-1','2021-12-31',df)

#test_case('2021-1-1','2021-6-30',df)
#test_case('2021-7-1','2021-12-31',df)
print(df.columns)
print(df.dtypes)
print(df.shape)