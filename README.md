# pycaret_catboost

고객 이용 행동과 계약 정보를 분석하여 지원이 필요한 정도를 나타내는 등급 분류 AI 알고리즘을 개발하고자 한다. 
이를 위해 데이터(명목형 변수)의 연관성과 스피어만 상관계수의 특성을 확인 후, 이를 적용하여 상관관계를 예측한다.   

## 1. Overview 

[목표]
  
 고객의 지원 필요 수준을 예측하는 AI 알고리즘 개발
  
[세부 목표]
  
 - k-Means cluster 파생변수 추가
    
 - Pycaret AutoML 상위 3개 모델 선정
    
 - Catboost를 blend model로 선정하여 전방모델은 2번 상위모델 3개로 배치하여 stacking

## 2. Data Description

[데이터 출처]

  - Dacon Basic 해커톤 (2025.08.04 ~ 2025.09.30)
   
[데이터 유형]
  
  - 다중 분류(Multi-class Classification)
  
[샘플 수]
  
  - N = 578
  
[데이터 분석 목적]
      
  - 고객의 사용 행동 및 패턴을 기반으로 지원 필요도(support_needs) 예측을 통해
     
    (1) 고객 이탈 방지(Customer Retention)

    (2) 지원 리소스 최적화(Support Resource Allocation) 하는 것을 목적으로 함 

## 3. Methodology
  본 프로젝트는 [데이터 수집/전처리] → [결측치 처리] → [변수 인코딩] → [모델 학습] → [앙상블] → [성능평가] 순으로 수행 하였다. 
    
  (1) 데이터 수집 및 전처리 
    
    - 범주형 변수 인코딩 
    
    - 이상치 처리 
    
    - 정규화  
  
  (2) 결측치 확인 및 처리 
  
  (3) 범주형 변수 인코딩(Label Encoding)
  
  (4) 정규화 및 데이터 불균형 조정 
  
  (5) 모델 튜닝, 학습 및 앙상블 
  
  (6) 성능 평가 
  
## 4. Modeling & Techniques
 (1) [데이터 전처리]
  
   - gender(M, F)와 subscription_type(member, plus, vip)을 명목변수로 변환하여 LabelEncoding으로 수치화  
    
 (2) [불균형 조율]
 
   - fix_imbalance=True 옵션 설정하여 SMOTE 적용

 (3) [Pycaret]
    
   - 상위 3개 모델 선정: compare_models()
    
   - Blend_models()로 선정한 3개 모델 앙상블 
    
   - tune_model()로 최적의 조합으로 catboost 모델 자동 탐색하여 하이퍼파라미터 조정 
    
 (4) [K-means Clustering]
    
   - 행동 특성이 비슷한 고객 그룹 정보 추가 하여 해석 및 예측
    
  ```python
   from sklearn.cluster import KMeans
  
   kmeans = KMeans(n_clusters=3, random_state=42)
   train_df['cluster'] = kmeans.fit_predict(train_df[['age','tenure','frequent','payment_interval','contract_length','after_interaction']])
         test_df['cluster'] = kmeans.predict(test_df[['age','tenure','frequent','payment_interval','contract_length','after_interaction']])
  ```
    
## 5. Results & Evaluation
 (1) Top3 compare model(n_select=3, sort='AUC')
  
  ![table 1. top-3 compare model](https://github.com/seirah-yang/pycaret_catboost/blob/main/top3_model.png)

 (2) Catboost blend model Result
  
  ![table 2. catboost model(blend)](https://github.com/seirah-yang/pycaret_catboost/blob/main/catboost_final(blend).png)
  
 (3) Catboost_model tunning Result
    
   - Accuracy, AUC, Recall, Precision, F1, Kappa, MCC의 fold별 scores
 
  ![graph 1. catboost tuning result](https://github.com/seirah-yang/pycaret_catboost/blob/main/catboot_tuningresult.png)
  
  ![table 3. catboost tuning result](https://github.com/seirah-yang/pycaret_catboost/blob/main/catboost_tuned.png)
 
## 6. Discution / Reflection
 [문제사항]
    
  - 모든 예측 결과가 '0'으로 출력 
 
  ![table 4. Label Encoding 전](https://github.com/seirah-yang/pycaret_catboost/blob/main/beforeLE.png)

 [원인파악]
  
  - gender(M,F)와 subscription_type(member,plus,vip)를 명목형 변수로인식하지만 Label Encoding 수행하지 않음 
    
  - 수치형 입력 요구시, 해당변수를 숫자로 변환하지 않고 문자열 유지, drop 처리
  
  - 즉, 모델이 수치형 입력만 요구할 경우 해당 변수를 제외하고 학습

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

  - 학습 시 gender, subscription_type이 포함되어 고객 유형과 행동 유형 모두 고려하여 분석 함
    
  - Fold별 Accuracy, F1, Mean, SD score를 통해 학습이 안정적으로 이루어 짐을 확인
    
  - 예측 결과가 '1'로 출력 되는 것을 확인
    
 ![table 5. Label Encoding 후](https://github.com/seirah-yang/pycaret_catboost/blob/main/after_LEpng)
     
## 7. Contributors / License
  양 소 라 (SORA YANG, Seirah) | RN, BSN, MSN  
    
  - JD : Oncology on Severance(Cancer center), CRC(NCC) mainly IIT & sub SIT, Data Management Intership(6m) 
    
  - Education experience : alpaco campus End-to-End AI developer master course (6m)
   
  💬 SNS: GitHub Profile 링크  |  [GitHub](https://github.com/SeIRah)

  💬 E-Mail: nftsgsrz3@gmail.com | Mobile: 010-7258-5942
   
-------------------------------------------------------------------------------------------------
## References
 1. wikidocs. (2025). PyCaret을 활용한 분류모델 개발. WikiDocs. https://wikidocs.net/207087
 2. DACON. (2023, July 17 – July 31). Wind Speed Prediction AI hackathon [Online competition]. DACON.   https://dacon.io/en/competitions/official/236126
 3. Dacon. (2025, August 4 – September 30). *Basic Customer Support Level Classification: Find Customers Who Need Help!* [Online competition]. Dacon. https://dacon.io/competitions/official/236214
