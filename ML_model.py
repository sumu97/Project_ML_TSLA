"""  Key Feature  """
target = 'TSLA'

"""  Collect Data  """
import joblib
import os
# 주소
base = os.path.dirname(os.path.abspath(__file__))
# 데이터
df = joblib.load(os.path.join(base,'data',target))
# 데이터 결측치 제거
df = df.dropna(axis = 0)

p1 = joblib.load(os.path.join(base,'param','p1'))
p2 = joblib.load(os.path.join(base,'param','p2'))
p3 = joblib.load(os.path.join(base,'param','p3'))
p4 = joblib.load(os.path.join(base,'param','p4'))
p5 = joblib.load(os.path.join(base,'param','p5'))
p6 = joblib.load(os.path.join(base,'param','p6'))
p7 = joblib.load(os.path.join(base,'param','p7'))
p8 = joblib.load(os.path.join(base,'param','p8'))
p9 = joblib.load(os.path.join(base,'param','p9'))
p10 = joblib.load(os.path.join(base,'param','p10'))

print(p1['max_depth'],p1['min_samples_leaf'],p1['min_samples_split'],p1['n_estimators'])

"""  Train - Test Split  """
# Train Test Split * 0.7
train = df[0:int(len(df)*0.7)]
test = df[int(len(df)*0.7):]
features = ['Search','Price','Open','High','Low','Vol','Change','avg_one','avg_thr','avg_fiv']#,'avg_ten']
# Train Data Set
X_train = train[features]
y_train_1 = train['one']
y_train_2 = train['two']
y_train_3 = train['thr']
y_train_4 = train['for']
y_train_5 = train['fiv']
y_train_6 = train['six']
y_train_7 = train['sev']
y_train_8 = train['eig']
y_train_9 = train['nin']
y_train_10 = train['ten']
# Test Data Set_1
X_test_1 = test[features]
X_test_1 = X_test_1[:-1]
y_test_1 = test['one']
y_test_1 = y_test_1[:-1]

# Test Data Set_2
X_test_2 = test[features]
X_test_2 = X_test_2[:-2]
y_test_2 = test['two']
y_test_2 = y_test_2[:-2]

# Test Data Set_3
X_test_3 = test[features]
X_test_3 = X_test_3[:-3]
y_test_3 = test['thr']
y_test_3 = y_test_3[:-3]

# Test Data Set_4
X_test_4 = test[features]
X_test_4 = X_test_4[:-4]
y_test_4 = test['for']
y_test_4 = y_test_4[:-4]

# Test Data Set_5
X_test_5 = test[features]
X_test_5 = X_test_5[:-5]
y_test_5 = test['fiv']
y_test_5 = y_test_5[:-5]


# Test Data Set_6
X_test_6 = test[features]
X_test_6 = X_test_6[:-6]
y_test_6 = test['six']
y_test_6 = y_test_6[:-6]

# Test Data Set_7
X_test_7 = test[features]
X_test_7 = X_test_7[:-7]
y_test_7 = test['sev']
y_test_7 = y_test_7[:-7]

# Test Data Set_8
X_test_8 = test[features]
X_test_8 = X_test_8[:-8]
y_test_8 = test['eig']
y_test_8 = y_test_8[:-8]

# Test Data Set_9
X_test_9 = test[features]
X_test_9 = X_test_9[:-9]
y_test_9 = test['nin']
y_test_9 = y_test_9[:-9]

# Test Data Set_10
X_test_10 = test[features]
X_test_10 = X_test_10[:-10]
y_test_10 = test['ten']
y_test_10 = y_test_10[:-10]

"""   Model  """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1주 뒤 예측
pipe_1 = RandomForestClassifier(n_estimators = p1['n_estimators'], 
                                max_depth = p1['max_depth'],
                                min_samples_leaf = p1['min_samples_leaf'],
                                min_samples_split = p1['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_1.fit(X_train, y_train_1)
pred = pipe_1.predict(X_test_1)
print('[1주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_1,pred)))
print('훈련 정확도: ', pipe_1.score(X_train, y_train_1))
print('검증 정확도: ', pipe_1.score(X_test_1, y_test_1))


# 2주 뒤 예측
pipe_2 = RandomForestClassifier(n_estimators = p2['n_estimators'], 
                                max_depth = p2['max_depth'],
                                min_samples_leaf = p2['min_samples_leaf'],
                                min_samples_split = p2['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_2.fit(X_train, y_train_2)
pred = pipe_2.predict(X_test_2)
print('[2주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_2,pred)))
print('훈련 정확도: ', pipe_2.score(X_train, y_train_2))
print('검증 정확도: ', pipe_2.score(X_test_2, y_test_2))

# 3주 뒤 예측
pipe_3 = RandomForestClassifier(n_estimators = p3['n_estimators'], 
                                max_depth = p3['max_depth'],
                                min_samples_leaf = p3['min_samples_leaf'],
                                min_samples_split = p3['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_3.fit(X_train, y_train_3)
pred = pipe_3.predict(X_test_3)
print('[3주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_3,pred)))
print('훈련 정확도: ', pipe_3.score(X_train, y_train_3))
print('검증 정확도: ', pipe_3.score(X_test_3, y_test_3))

# 4주 뒤 예측
pipe_4 = RandomForestClassifier(n_estimators = p4['n_estimators'], 
                                max_depth = p4['max_depth'],
                                min_samples_leaf = p4['min_samples_leaf'],
                                min_samples_split = p4['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_4.fit(X_train, y_train_4)
pred = pipe_4.predict(X_test_4)
print('[4주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_4,pred)))
print('훈련 정확도: ', pipe_4.score(X_train, y_train_4))
print('검증 정확도: ', pipe_4.score(X_test_4, y_test_4))

# 5주 뒤 예측
pipe_5 = RandomForestClassifier(n_estimators = p5['n_estimators'], 
                                max_depth = p5['max_depth'],
                                min_samples_leaf = p5['min_samples_leaf'],
                                min_samples_split = p5['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_5.fit(X_train, y_train_5)
pred = pipe_5.predict(X_test_5)
print('[5주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_5,pred)))
print('훈련 정확도: ', pipe_5.score(X_train, y_train_5))
print('검증 정확도: ', pipe_5.score(X_test_5, y_test_5))


# 6주 뒤 예측
pipe_6 = RandomForestClassifier(n_estimators = p6['n_estimators'], 
                                max_depth = p6['max_depth'],
                                min_samples_leaf = p6['min_samples_leaf'],
                                min_samples_split = p6['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_6.fit(X_train, y_train_6)
pred = pipe_6.predict(X_test_6)
print('[6주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_6,pred)))
print('훈련 정확도: ', pipe_6.score(X_train, y_train_6))
print('검증 정확도: ', pipe_6.score(X_test_6, y_test_6))

# 7주 뒤 예측
pipe_7 = RandomForestClassifier(n_estimators = p7['n_estimators'], 
                                max_depth = p7['max_depth'],
                                min_samples_leaf = p7['min_samples_leaf'],
                                min_samples_split = p7['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_7.fit(X_train, y_train_7)
pred = pipe_7.predict(X_test_7)
print('[7주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_7,pred)))
print('훈련 정확도: ', pipe_7.score(X_train, y_train_7))
print('검증 정확도: ', pipe_7.score(X_test_7, y_test_7))

# 8주 뒤 예측
pipe_8 = RandomForestClassifier(n_estimators = p8['n_estimators'], 
                                max_depth = p8['max_depth'],
                                min_samples_leaf = p8['min_samples_leaf'],
                                min_samples_split = p8['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_8.fit(X_train, y_train_8)
pred = pipe_8.predict(X_test_8)
print('[2주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_8,pred)))
print('훈련 정확도: ', pipe_8.score(X_train, y_train_8))
print('검증 정확도: ', pipe_8.score(X_test_8, y_test_8))

# 9주 뒤 예측
pipe_9 = RandomForestClassifier(n_estimators = p9['n_estimators'], 
                                max_depth = p9['max_depth'],
                                min_samples_leaf = p9['min_samples_leaf'],
                                min_samples_split = p9['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_9.fit(X_train, y_train_9)
pred = pipe_9.predict(X_test_9)
print('[9주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_9,pred)))
print('훈련 정확도: ', pipe_9.score(X_train, y_train_9))
print('검증 정확도: ', pipe_9.score(X_test_9, y_test_9))

# 10주 뒤 예측
pipe_10 = RandomForestClassifier(n_estimators = p10['n_estimators'], 
                                max_depth = p10['max_depth'],
                                min_samples_leaf = p10['min_samples_leaf'],
                                min_samples_split = p10['min_samples_split'],
                                random_state = 0,
                                n_jobs = -1)
pipe_10.fit(X_train, y_train_10)
pred = pipe_10.predict(X_test_10)
print('[10주 예측]')
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test_10,pred)))
print('훈련 정확도: ', pipe_10.score(X_train, y_train_10))
print('검증 정확도: ', pipe_10.score(X_test_10, y_test_10))


"""  Save  """
joblib.dump(pipe_1, os.path.join(base,'model','pipe_1'))
joblib.dump(pipe_2, os.path.join(base,'model','pipe_2'))
joblib.dump(pipe_3, os.path.join(base,'model','pipe_3'))
joblib.dump(pipe_4, os.path.join(base,'model','pipe_4'))
joblib.dump(pipe_5, os.path.join(base,'model','pipe_5'))
joblib.dump(pipe_6, os.path.join(base,'model','pipe_6'))
joblib.dump(pipe_7, os.path.join(base,'model','pipe_7'))
joblib.dump(pipe_8, os.path.join(base,'model','pipe_8'))
joblib.dump(pipe_9, os.path.join(base,'model','pipe_9'))
joblib.dump(pipe_10, os.path.join(base,'model','pipe_10'))
