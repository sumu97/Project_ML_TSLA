# 의문 : 데이터베이스 드갔다 나오면 정제된 데이터가 깨지는가? (datetime 자료형 등)
# 테스트 결과 : 기존 데이터와 비교해보니 datetime 자료형에 00:00:00 붙어있음. 그 외 나머지 자료형은 동일
# 어차피 크롤링 매번 해도 큰 상관 없고 : 크롤링 시간 10초 미만
# 짧은 단위로 추가해 줄 수 있는 데이터도 아님 : Search 데이터의 경우 기간에서의 비율이라 3개월만 뽑아서 데이터 추출시 못써먹는 데이터가 추출됨
# 결론 : 꺼낼때마다 짜증나게 데이터 형식 바꿔주느니 그냥 피클해서 쓰는게 났겠다는 생각.

""" Collect Stored Data """
target = 'TSLA'
# SQLite 데이터 가져오기
import sqlite3
import os
base = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base,'data',(target+'.db')) # db 경로 (p1 > data > tsla.db)
con = sqlite3.connect(db_path)
cur = con.cursor()
# 쿼리문 실행
query = cur.execute(("SELECT * FROM " + target))
cols = [column[0] for column in query.description]
import pandas as pd
df = pd.DataFrame.from_records(data=query.fetchall(), columns = cols)
# 생성된 index 열 제거
df = df.drop(['index'],axis=1)
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



import joblib
df = joblib.load(os.path.join(base,'model','df'))

print(df.columns)
print(df.dtypes)

# 폐기된 코드
""" Link Database 
# db 경로 설정
import sqlite3
import os
base = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base,'data',(target+'.db')) # db 경로 (p1 > data > tsla.db)

# db 연결 및 쿼리 적용
con = sqlite3.connect(db_path)
try: # 테이블 드랍, 테이블 생성
    cur = con.cursor()
    query = ("drop table " + target)
    cur.execute(query)
    df.to_sql(target, con)
except: # 테이블 드랍 안되면(테이블 없으면) 테이블 생성
    df.to_sql(target, con)
"""