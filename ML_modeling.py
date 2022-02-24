import pandas as pd
import os

"""  1. 데이터 불러오기  """
# 현재파일 절대경로 위 디렉토리 추출
base = os.path.dirname(os.path.abspath(__file__))
# 검색 데이터 [날짜, 검색량]
df_a = pd.read_csv(os.path.join(base,'data','tsla_search.csv'))
# 주가 데이터 [날짜, 종가, 시가, 고가, 저가, 거래량, 변동률 ]
df_b = pd.read_csv(os.path.join(base,'data','tsla_stock.csv'))

"""  2. 데이터 정렬  """
# 검색 데이터를 거꾸로
df_a = df_a.sort_values(by=['Date'],axis = 0,ascending=False)
# 주가 데이터를 거꾸로
df_a = df_a.reset_index(drop=True, inplace = False)
# 검색, 주가 데이터 합치기
df = pd.concat([df_a,df_b], axis = 1, ignore_index = True)
# 컬럼명 재설정
df.columns = ['Date','Search','Day(notuse)','Price','Open','High','Low','Vol','Change']
# 필요없는 컬럼 제거
df = df.drop(['Day(notuse)'],axis = 1)
# 데이터 시간순 정렬
df = df.sort_values(by=['Date'],axis = 0,ascending=True)

# 자료형 변경
import datetime
import numpy as np
for col in df.columns:
  temp_list = []
  for i in df[col]:
    try: # datetime 우선 처리
      temp = datetime.datetime.strptime(i, '%Y-%m-%d').date()
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

# 인덱스 초기화
df = df.reset_index(drop = True)

"""  3. 데이터 조작  """
# Label (가격변화) 생성 함수 [몇 주 뒤, 비교할 특성]
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

# Label 생성
df['one'] = price_go(1,df.Price)
df['thr'] = price_go(3,df.Price)
df['fiv'] = price_go(5,df.Price)
df['ten'] = price_go(10,df.Price)

# Feature (이동평균선) 생성 함수 [몇 주 치, 비교할 특성]
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
  return target_list

# Feature 생성
df['avg_one'] = moving_avg(1, df.Price)
df['avg_thr'] = moving_avg(3, df.Price)
df['avg_fiv'] = moving_avg(5, df.Price)
df['avg_ten'] = moving_avg(10, df.Price)

# Label, Feature 으로 생성된 결측치 제거 (20개)
df = df.dropna(axis = 0)

# Train Test Split * 0.7
train = df[0:int(len(df)*0.7)]
test = df[int(len(df)*0.7):]
features = ['Search','Price','Open','High','Low','Vol','Change','avg_one','avg_thr','avg_fiv','avg_ten']
# Train Data Set
X_train = train[features]
y_train_1 = train['one']
y_train_3 = train['thr']
y_train_5 = train['fiv']
y_train_10 = train['ten']
# Test Data Set_1
X_test_1 = test[features]
X_test_1 = X_test_1[:-1]
y_test_1 = test['one']
y_test_1 = y_test_1[:-1]

# Test Data Set_3
X_test_3 = test[features]
X_test_3 = X_test_3[:-3]
y_test_3 = test['thr']
y_test_3 = y_test_3[:-3]

# Test Data Set_5
X_test_5 = test[features]
X_test_5 = X_test_5[:-5]
y_test_5 = test['fiv']
y_test_5 = y_test_5[:-5]

# Test Data Set_10
X_test_10 = test[features]
X_test_10 = X_test_10[:-10]
y_test_10 = test['ten']
y_test_10 = y_test_10[:-10]

"""  4. 데이터 모델링  """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pipe_1 = RandomForestClassifier(n_estimators = 10, 
                                max_depth = 2,
                                min_samples_leaf = 32,
                                min_samples_split = 2,
                                random_state = 0,
                                n_jobs = -1)
pipe_1.fit(X_train, y_train_1)

pipe_3 = RandomForestClassifier(n_estimators = 10, 
                                max_depth = 8,
                                min_samples_leaf = 4,
                                min_samples_split = 2,
                                random_state = 0,
                                n_jobs = -1)
pipe_3.fit(X_train, y_train_3)

pipe_5 = RandomForestClassifier(n_estimators = 10, 
                                max_depth = 2,
                                min_samples_leaf = 32,
                                min_samples_split = 2,
                                random_state = 0,
                                n_jobs = -1)
pipe_5.fit(X_train, y_train_5)

pipe_10 = RandomForestClassifier(n_estimators = 100, 
                                max_depth = 2,
                                min_samples_leaf = 16,
                                min_samples_split = 2,
                                random_state = 0,
                                n_jobs = -1)
pipe_10.fit(X_train, y_train_10)

"""  5. 생성된 머신러닝 모델 저장  """
import joblib
joblib.dump(pipe_1, os.path.join(base,'model','pipe_1'))
joblib.dump(pipe_3, os.path.join(base,'model','pipe_3'))
joblib.dump(pipe_5, os.path.join(base,'model','pipe_5'))
joblib.dump(pipe_10, os.path.join(base,'model','pipe_10'))
joblib.dump(df, os.path.join(base,'model','df'))

""" 저장된 모델 불러오기
import joblib
base = '/Users/yutaejun/Desktop/code/p1'
pipe_1 = joblib.load(os.path.join(base,'model','pipe_1'))
pipe_3 = joblib.load(os.path.join(base,'model','pipe_3'))
pipe_5 = joblib.load(os.path.join(base,'model','pipe_5'))
pipe_10 = joblib.load(os.path.join(base,'model','pipe_10'))
"""