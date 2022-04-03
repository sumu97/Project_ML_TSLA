# Project_ML_TSLA
테슬라 검색량 데이터와 주가 데이터를 통한 투자지표 생성 프로젝트
- Youtube : https://www.youtube.com/watch?v=UiNQL1OyaYo&t=4s

## WHY
- 군에 있을 때, 테슬라에 투자하며 4000만원 가량의 수익을 올렸었다.
- 이때 활용했던 투자방식은 CNN의 Fear & Greed Index 참고와 TSLA의 구글 검색량 데이터를 활용하는 방식이었다.
- TSLA에 대한 사람들의 관심이 크다면 검색량이 오르고, 가격도 '올랐다'고 판단했다
- 따라서 TSLA 검색량이 낮을 때 투자하는 방식으로 많은 수익을 올릴 수 있었다.
- 파이썬을 활용한 머신러닝으로 위 투자방식을 검증해보려 한다.

## DATA
- 테슬라 검색량 데이터 (Google Trend)
- 테슬라 주가 데이터 (Investing.com)

## HOW
- 데이터 수집 방법 : Google Trend 크롤러인 Pytrend API, Investing.com 크롤러인 FinanceDataReader API 활용
- 데이터 EDA 방법 : pandas
- 데이터, 모델, 파라미터 저장 : joblib 라이브러리 활용한 Pickle화
- 특성공학 : n주 치 이동평균선, n주 뒤 수익여부(Target)
- 머신러닝 : Random Forest Classifier 로 Target 이진분류 (각 10개의 Target, model, parameter)
- 테스트 케이스 : 예측값들로 예상투자수익률 도출

## RESULT
- 특성중요도를 시각화했더니, 초기의 가설대로 대부분의 모델에서 검색량 데이터가 상위권에 위치해 있었다. (높은 연관관계 의미)
- 실제 데이터와 근접하게 예측에 성공했다.
- 수익률은 모델 세부 설정에 따라 변동되지만 일반적인 투자방식보다 10%~20% 높은 수익률을 보여줬다.
- 다만 튜닝을 잘못하면 결과가 변별력이 없어지는 등의 문제가 있어 데이터가 들어올때마다 조정이 까다롭다.

## OPINION
- 현재 1사이클을 4주로 잡고 투자를 진행하고 있는데, 너무 진행속도가 느리다는 의견이 있어 2주나 3주단위로 테스트 할 예정이다.
- 이전 데이터를 토대로, 큰 수의 법칙에 따라 수익이 날 것으로 예측했지만, 주식은 역시 아무도 모른다는 말이 맞는것 같다.
- 계속해서 개선해 나가 추후에는 이전에 했던 프로젝트처럼 원자재가격의 변동량을 반영하는 등 있는대로 데이터를 끌어모아 예측해보고 싶다.

## 투자방식  
- Notion : https://pinto-truck-71f.notion.site/MLT-081a6ec08e4d423d8be1926b723e04b1  
  
## 코드설명

- Notion : https://pinto-truck-71f.notion.site/MLT-1a0c5b71e65f462f9dfcd453da507e81  

## 실행 가이드
1. DB_crawling 실행
2. ML_tuning 실행
3. ML_model 실행
4. DT_predict 실행

## 분석 가이드
1. PRED 의 sum_p 가 예측값들의 합으로, 10일때 주식을 산다.
2. 4주를 기다린다
3. 판다
4. 평균 수익률 22.531% 가 나오는지 확인한다.
5. 신나면 유태준한테 밥을 한번 산다.

## 특성 가이드 
Date : 날짜  
Search : 검색량, 초기 입력값인 start_date 와 end_date 사이의 값들을 대상으로 0~100의 비율을 갖는 값  
Price : 주가  
sum_r : 실제 데이터의 상승가능성 합산  
sum_p : 예측 데이터의 상승가능성 합산  
Income : 4주 뒤 팔때 얻는 수익금  
Incrat : 4주 뒤 팔때 얻는 수익률  
  
## 생성된 특성
이동평균선 : (1, 3, 5주)  
수익 여부 ~주 뒤 : (1,2,3,4,5,6,7,8,9,10)

## 내부 파일 설명  
ML_modeling.py = 머신러닝 모델 생성 : ./model에 결과 저장됨  
DB_crawling.py = 데이터베이스 크롤링 : ./data에 결과 저장됨  
ML_tuning.py = 머신러닝 하이퍼파라미터 튜닝 : ./param에 결과 저장됨  
ML_model.py = 머신러닝 모델 생성 : ./model에 결과 저장됨  
DT_predict.py = 데이터에 적용, 예측값 확인 및 테스트 : 결과물 출력됨  
test_pickle.py = 저장된 모델 성능 테스트   
test_datatype.py = 크롤링 데이터 정제를 위한 테스트  
test_db2df.py = 데이터베이스와 데이터프레임 간 연결을 위한 테스트  
test_pytrends.py = pytrend API 활용 테스트  
test_selenium.py = selenium 동적 웹 크롤링 테스트  

## 내부 폴더 설명 
data = 모델링 기초 데이터, 크롤링한 데이터(pickled)  
model = 머신러닝 모델 데이터  
param = 계산된 최적 하이퍼파라미터 데이터  

## 구동환경  
가상환경 = conda / P1  

## 활용된 라이브러리  
pandas  
numpy   
scikit-learn   
joblib   
finance-datareader  
beautifulsoup4 - not use  
matplotlib - not use  
selenium - not use  
pytrend  
 
## 개선점
Finance Data Reader 는 사용하기 쉽지만 데이터가 일일단위. 난 주단위 데이터가 필요하다고 생각했으나,  
요일별 특성을 학습시켜 별도의 모델을 생성한다면 일일단위 데이터로도 더 정밀한 결과물을 뽑아낼 수 있을것으로 보임.  
  
TSLA 말고 AAPL 도 가능할지? 물론 새로 학습해야 함.  
-> 가능함. 단, 검색 특성에 따른 영향이 적어 유의미한 결과도출은 어려움.  
  
매수신호를 녹색으로 그래프 덮어서 보여주면 시각화에 되게 좋을것같은데.  
 
## 초기개발기록
[To_Do_List : 22.2.20]  
1. 프로젝트 목표 설정 : 아 한국투자증권 가고싶다!  
2. 목표에 맞는 주제 설정 : 내 투자경험을 살린 머신러닝 모델링  
3. 필요 데이터 수집 : 구글 트렌드 검색량 데이터, 주가 데이터 (NASDAQ:TSLA)  
4. 코랩 환경 구성 (csv 데이터 가공)  

[To_Do_List : 22.2.21]  
1. EDA : 데이터를 코랩 상에 띄워 가공(자료형, 순서, 합치기 등)  
2. Wrangling : Target(Label) 특성 생성 (0 / 1), 이동평균선 특성 생성, 테스트 케이스 분리  
3. ML_modeling : 랜덤포레스트 분류, GridSearchCV로 하이퍼파라미터 튜닝, 예측  
4. Test_case : 테스트 시행, 예측모델의 평균 수익률 회당 21.8% 도출 (4주뒤 매도 기준)  
5. Product Building : 상업적(투자적) 목적으로 활용하기 위해 구성되야 할 기능 도출  

[To_Do_List : 22.2.22]  
1. 로컬 환경에서 구동할 수 있도록 맥북에어 환경설정  
2. VScode, Conda 등 필요 어플리케이션 설치  
3. 터미널 환경설정(예쁘게), VScode 확장 프로그램 설치 및 기본설정  
4. Colab에서 작성한 코드를 로컬에서 실행, 모델 저장  
5. 저장(Pickle)된 모델 테스트, 결과는 정상적  
6. 프로젝트 구성요소 및 구조 구상  

[To_Do_List : 22.2.23]  
1. 주식 데이터, 검색량 데이터 다운로드  
2. 기존 EDA 함수 / 코드에 데이터 적용  
3. 모델 수익성 재검증 (데이터 양식 변경됨)  

[To_Do_List : 22.2.24]  
1. 최신화한 데이터를 SQLite에 넣어 계층형 데이터베이스(단순)에 저장 / 완료  
2. 기능 반 자동화 (DB_crawling.py 실행 시 자동으로 모든 프로세스 진행, DB 업데이트 되도록 함) / 완료  
3. 저장된 데이터로 예측하는 과정에서 데이터베이스 시스템의 문제 발견, Pickle dump로 데이터 저장하는걸로 바꿈.  
4. 기존 데이터에서는 전부 1 (투자 신호) 인 케이스가 전체의 1/3 정도였는데, 1/2 수준으로 떨어졌다.  
 - 2주에 한번 추천하는 꼴, 투자에 활용하기 어렵다. (수익률도 21.2% -> 14.5%로 감소, 물론 Base model인 5%보다는 높긴 함.)  
 - 변별력을 추가하기 위해서 1부터 10까지 모든 케이스에 대해 학습하고, 전부 더한 총점 개념으로 투자지표를 삼는다면 변별력을 개선할 수 있을 것으로 생각됨
5. 학습모듈(GridSearchCV) 로컬라이징 및 신규 데이터로 학습 진행 (7:3)  
6. 모델 10개 학습 한 뒤 추출비율 28%까지 낮춰졌고, 전체 수익률도 22.531% 로 증가함.  
 - 모델 개선은 충분한 것 같고, Flask 적용해 웹 어플리케이션으로 적용  
7. Flask 적용, 웹 기반 구동 가능하게끔 작성  

# 현재는 다른 레포지토리에서 개발 및 활용 
