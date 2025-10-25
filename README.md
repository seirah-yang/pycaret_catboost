# pycaret_catboost

고객 이용 행동과 계약 정보를 분석하여 지원이 필요한 정도를 나타내는 등급 분류 AI 알고리즘을 개발하고자 한다. 
이를 위해 데이터(명목형 변수)의 연관성과 스피어만 상관계수의 특성을 확인 후, 이를 적용하여 상관관계를 예측한다.   

## 1. 프로젝트 개요(Overview) 

[목표]
  
고객 이용 행동과 계약 정보를 기반으로 지원 필요 수준(support_needs) 을 예측하는 AI 등급 분류 알고리즘 개발
  
[세부 목표]
  - 고객 이용 데이터의 명목형 변수(Label Encoding) 변환
  
	- Spearman 상관계수 분석을 통한 피처 간 연관성 파악
	
  - K-Means Clustering 파생 변수 추가로 행동 패턴 군집화
	
  - PyCaret AutoML을 통해 상위 3개 모델 자동 탐색
	
  - CatBoost를 블렌딩(Blending) 하여 최종 스태킹 모델 구성
  
## 2. 데이터 설명 (Data Description)
[데이터 출처]

  - Dacon Basic 해커톤 (2025.08.04 ~ 2025.09.30)
   
[데이터 유형]
  
  - 다중 분류(Multi-class Classification)
  
[샘플 수]
  
  - N = 578
  
[분석 목적]
      
 고객 행동 패턴 기반으로 지원 리소스 예측 및 최적화

[분석 목표]

    (1) 고객 이탈 방지(Customer Retention)

    (2) 지원 리소스 최적화(Support Resource Allocation) 하는 것을 목적으로 함 

## 3. 분석 프로세스 (Methodology)

[데이터 수집] → [결측치 처리] → [명목형 변수 인코딩] → [정규성·등분산성 검정] → [ANOVA/사후분석] → [AutoML (PyCaret)] → [CatBoost 튜닝 & 블렌딩] → [성능 평가]
    
  (1) 데이터 수집 및 전처리 
    
    - Label Encoding 
    
    - 이상치 처리 및 정규화

    - 사용기술 : pandas, numpy
  
  (2) 결측치 확인 및 처리 
  
    - 단일/다중 대체법 확인

    - 사용기술 : pandas
  
  (3) 명목형 변수 인코딩

    - gender, subscription_type → 수치형 변환

    - 사용기술 : LabelEncoder
      
  (4) 통계검정

    - Shapiro–Wilk, Levene Test

    - 사용기술 : scipy.stats
      
  (5) 분산분석 및 사후분석

    - ANOVA / Welch / Kruskal–Wallis

    - Tukey HSD, Games–Howell, Dunn

    - 사용기술 : pingouin, scikit-posthocs
      
  (6) 모델링

    - PyCaret AutoML

    - 사용기술 : pycaret.classification
      
  (3) K-Means Cluster

    - 파생변수의 군집화 
    
    - 사용기술 : sklearn.cluster

  
## 4. Modeling & Techniques
 1) Label Encoding
```python  
    train_df['gender'] = train_df['gender'].map({'M': 0, 'F': 1})
    train_df['subscription_type'] = train_df['subscription_type'].map({'member': 0, 'plus': 1, 'vip': 2})
    
    test_df['gender'] = test_df['gender'].map({'M': 0, 'F': 1})
    test_df['subscription_type'] = test_df['subscription_type'].map({'member': 0, 'plus': 1, 'vip': 2}) 
```

 2) K-Means Clustering
```python
   from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=42)
    train_df['cluster'] = kmeans.fit_predict(train_df[['age','tenure','frequent','payment_interval','contract_length','after_interaction']])
    test_df['cluster'] = kmeans.predict(test_df[['age','tenure','frequent','payment_interval','contract_length','after_interaction']])
```

 3)  CatBoost 블렌딩 모델
    (1)	상위 3개 모델 기반으로 CatBoost를 중심으로 블렌딩

	  (2)	5-Fold Cross Validation 수행

	  (3)	Accuracy, AUC, Recall, Precision, F1, Kappa, MCC 평가
    

    
## 5. Results & Evaluation
 (1) Top3 compare model(n_select=3, sort='AUC')
  
  ![table 1. top-3 compare model](https://github.com/seirah-yang/pycaret_catboost/blob/main/top3_model.png)

 (2) 통계분석 결과 
 
 (3) Catboost blend model Result
  
  ![table 2. catboost model(blend)](https://github.com/seirah-yang/pycaret_catboost/blob/main/catboost_final(blend).png)
  
 (4) Catboost_model tunning Result
    
   - Accuracy, AUC, Recall, Precision, F1, Kappa, MCC의 fold별 scores
 
  ![graph 1. catboost tuning result](https://github.com/seirah-yang/pycaret_catboost/blob/main/catboot_tuningresult.png)
  
  ![table 3. catboost tuning result](https://github.com/seirah-yang/pycaret_catboost/blob/main/catboost_tuned.png)
 
## 6. Discution / Reflection
 [문제사항]
    
  - 모든 예측 결과가 '0'으로 출력됨
    → 모델이 문자열형 변수를 학습하지 못함 
 
  ![table 4. Label Encoding 전](https://github.com/seirah-yang/pycaret_catboost/blob/main/beforeLE.png)

 [원인파악]
  
  - gender / subscription_type이 문자열 상태로 남아 PyCaret 내부에서 자동 제외됨
    → feature loss 발생
    
  - 수치형 입력 요구시, 해당변수를 숫자로 변환하지 않고 문자열 유지, drop 처리
    → 모델이 수치형 입력만 요구할 경우 해당 변수를 제외하고 학습

 [해결방안 탐색]
    
  - gender(M:1,F:2), subscription_type(member:0,plus:1,vip:2)을 Label Encoding 수행하여 분석에 포함
    
 [해결방안 적용]  
    
  - 데이터 분포가 '고객 활동'만 보고 예측하는 과정에서 gender와 subscription_type을 포함

```bash
  train_df['gender'] = train_df['gender'].map({'M': 0, 'F': 1})
  train_df['subscription_type'] = train_df['subscription_type'].map({'member': 0, 'plus': 1, 'vip': 2})
  test_df['gender'] = test_df['gender'].map({'M': 0, 'F': 1})
  test_df['subscription_type'] = test_df['subscription_type'].map({'member': 0, 'plus': 1, 'vip': 2})
```  
 [결과]

  - 학습 시 gender, subscription_type이 포함되어 모델이 고객 행동 + 속성 모두 반영
    
  - Fold별 Accuracy, F1, Mean, SD score를 통해 학습이 안정적으로 이루어 짐을 확인
    
  - 예측값이 다양하게 생성되며, fold별 Accuracy와 F1 안정화
    
 ![table 5. Label Encoding 후](https://github.com/seirah-yang/pycaret_catboost/blob/main/after_LEpng)
     
## 7. Author
양 소 라 (SORA YANG, Seirah) | RN, BSN, MSN  
AI Developer Bootcamp @ Alpaco | Clinical Data Analyst Trainee
    
Oncology on Severance(Cancer center), CRC(NCC) mainly Data Management Intership(6m) 
    
Education experience : alpaco campus End-to-End AI developer master course (6m)
   
  💬 SNS: GitHub Profile 링크  |  [GitHub](https://github.com/SeIRah)

  💬 E-Mail: nftsgsrz3@gmail.com | Mobile: 010-7258-5942
   
-------------------------------------------------------------------------------------
## References
	1.	wikidocs. (2025). PyCaret을 활용한 분류모델 개발. https://wikidocs.net/207087
	2.	DACON. (2025, August 4 – September 30). Basic Customer Support Level Classification. https://dacon.io/competitions/official/236214
	3.	Abdi, H., & Williams, L. J. (2010). Tukey’s HSD Test. In Encyclopedia of Research Design.
	4.	Montgomery, D. C. (2019). Design and Analysis of Experiments. Wiley.
