"""  Key Feature  """
target = 'TSLA'

"""  Collect Data  """
import joblib
import os
# 주소
base = os.path.dirname(os.path.abspath(__file__))
# 데이터
df = joblib.load(os.path.join(base,'data',target))
df = df[10:] # 이동평균선으로 생기는 결측치 제거
df = df.reset_index(drop=True) # 인덱스 초기화

# 모델
pipe_1 = joblib.load(os.path.join(base,'model','pipe_1'))
pipe_2 = joblib.load(os.path.join(base,'model','pipe_2'))
pipe_3 = joblib.load(os.path.join(base,'model','pipe_3'))
pipe_4 = joblib.load(os.path.join(base,'model','pipe_4'))
pipe_5 = joblib.load(os.path.join(base,'model','pipe_5'))
pipe_6 = joblib.load(os.path.join(base,'model','pipe_6'))
pipe_7 = joblib.load(os.path.join(base,'model','pipe_7'))
pipe_8 = joblib.load(os.path.join(base,'model','pipe_8'))
pipe_9 = joblib.load(os.path.join(base,'model','pipe_9'))
pipe_10 = joblib.load(os.path.join(base,'model','pipe_10'))

"""  Prediction"""
import numpy as np
features = ['Search', 'Price', 'Open', 'High', 'Low', 'Vol', 'Change', 'avg_one', 'avg_thr', 'avg_fiv']#, 'avg_ten']
temp_df = df[features]

df['p_one'] = pipe_1.predict(temp_df)
df['p_two'] = pipe_2.predict(temp_df)
df['p_thr'] = pipe_3.predict(temp_df)
df['p_for'] = pipe_4.predict(temp_df)
df['p_fiv'] = pipe_5.predict(temp_df)
df['p_six'] = pipe_6.predict(temp_df)
df['p_sev'] = pipe_7.predict(temp_df)
df['p_eig'] = pipe_8.predict(temp_df)
df['p_nin'] = pipe_9.predict(temp_df)
df['p_ten'] = pipe_10.predict(temp_df)

# 예상 투자 수익 조건
# 1,3,5,10주 후 가격이 모두 상승할 때
#condition = (df['p_one'] == 1  & (df['p_thr']==1)) & ((df['p_fiv']==1) & (df['p_ten']==1))
# 1~10주 중 가격 오르는게 9 초과일 때
condition = (df['p_one'] + df['p_two'] + df['p_thr'] + df['p_for'] + df['p_fiv'] + df['p_six'] + \
            df['p_sev'] + df['p_eig'] + df['p_nin'] + df['p_ten']) > 9
# 특성 추가 - 4주 뒤 매도시 수익
target_feature = df.Price
much = 4
Income_list = []
Income_rate = []

for i in range(len(target_feature)):
  try:
    income = target_feature[i+much] - target_feature[i]
    Income_rate.append(round((income/target_feature[i]*100),2)) # 수익률
    Income_list.append(round((income),2)) # 수익

  except:
    Income_list.append(np.nan)
    Income_rate.append(np.nan)

df['Income'] = Income_list
df['Incrat'] = Income_rate

dfk = df[condition]
print("DATA----------------------------------------------------------")
print("전체 데이터: ",df.shape)
print("추출 데이터: ", dfk.shape)
print("추출 비율 :", dfk.shape[0]/df.shape[0]*100 ,"%")
print("PRED----------------------------------------------------------")


temp_list = []
df['sum_p'] = df['p_one'] + df['p_two'] + df['p_thr'] + df['p_for'] + df['p_fiv'] + df['p_six'] + df['p_sev'] + df['p_eig'] + df['p_nin'] + df['p_ten']
df['sum_r'] = df['one'] + df['two'] + df['thr'] + df['for'] + df['fiv'] + df['six'] + df['sev'] + df['eig'] + df['nin'] + df['ten']
dfp = df[['Date','Search','Price','sum_r','sum_p','Income','Incrat']]

print(dfp.tail(10))
#print(dfp[condition].tail(10))
# 시간 조건

print("Pred_Money----------------------------------------------------")
# 예측된 모든 지표가 1일 경우 ( 모델 예측 시행 )
print('예측 투자 평균 수익률 (전체) :',round(np.mean(dfk.Incrat),3),'%')
# 최근 5주 예측 수익률
print('평균 예측 수익률 (최근 05주) :',round(np.mean(dfk.Incrat[-8:-3]),3),'%')
# 최근 10주 예측 수익률
print('평균 예측 수익률 (최근 10주) :',round(np.mean(dfk.Incrat[-14:-3]),3),'%')
# 최근 25주 예측 수익률
print('평균 예측 수익률 (최근 25주) :',round(np.mean(dfk.Incrat[-29:-3]),3),'%')

print("Base_Money----------------------------------------------------")
# 무지성 투자했을경우의 수익률 (Base Model)
print('기본 투자 평균 수익률 (전체) :',round(np.mean(df.Incrat),3),'%')
# 최근 5주 기본 수익률
print('평균 기본 수익률 (최근 05주) :',round(np.mean(df.Incrat[-8:-3]),3),'%')
# 최근 10주 기본 수익률
print('평균 기본 수익률 (최근 10주) :',round(np.mean(df.Incrat[-14:-3]),3),'%')
# 최근 25주 기본 수익률
print('평균 기본 수익률 (최근 25주) :',round(np.mean(df.Incrat[-29:-3]),3),'%')



print("--------------------------------------------------------------")
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%matplotlib inline

ftr_importances_values = pipe_2.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = temp_df.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Top Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
#plt.show()
"""

"""
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
  #pred = np.concatenate((pred_1, pred_3, pred_5, pred_10), axis=0)
  #real = np.concatenate((real_1, real_3, real_5, real_10), axis=0)
  #print(start_date,"~",end_date,"예측 정확도 :",accuracy_score(real, pred) * 100,"%")
  print(start_date,"~",end_date,"예측 정확도 1 :",accuracy_score(real_1, pred_1) * 100,"%")
  print(start_date,"~",end_date,"예측 정확도 3 :",accuracy_score(real_3, pred_3) * 100,"%")
  print(start_date,"~",end_date,"예측 정확도 5 :",accuracy_score(real_5, pred_5) * 100,"%")
  print(start_date,"~",end_date,"예측 정확도 10 :",accuracy_score(real_10, pred_10) * 100,"%")



test_case('2017-2-2','2021-2-2',df)
print(pdf)

"""