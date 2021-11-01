# nia-diabetes-data-hackathon
NIA AI data 구축 과제 중 당뇨병 추적 관찰 데이터 구축 사업 내 해커톤을 위한 tutorial code repository

## 1. 배경

당뇨병은 혈중 당 농도를 조절하는 신체 기능의 이상으로 고혈당 상태가 지속되면서 이로 인한 여러 증상 및 합병증이 발생하게 되는 만성질환입니다. 당뇨병은 장기간에 걸쳐 환자의 삶의 질을 크게 떨어뜨릴 뿐만 아니라 심뇌혈관질환, 망막질환, 신장질환 등의 심각한 합병증 및 사망으로까지 이어질 수 있어, 예방 및 조기 진단을 통한 적절한 개입이 매우 중요합니다. 대한당뇨병학회에 따르면 2018년 기준 우리나라의 30세 이상 성인 7명 중 1명이 당뇨병을 가지고 있고, 4명 중 1명이 당뇨병의 전 단계인 공복혈당장애를 가지고 있는 것으로 추정된다고 합니다. 그러나 당뇨병을 가진 성인 중 6-7명만이 당뇨병을 가진 것을 인지하고 있는 등, 당뇨병의 위험에 대한 인식은 아직 부족합니다.

**만약 개인의 당뇨병 발병 여부를 미리 예측할 수 있다면, 이를 통해 조기에 당뇨병의 위험을 알리고 생활습관 개선을 유도하는 등 당뇨병 예방 및 치료에 기여할 수 있을 것입니다.**

## 2. 목표

본 해커톤은 NIA 당뇨병 추적 관찰 데이터 구축 사업을 통해 확보된 건강검진 데이터를 활용해 **향후 당뇨병 발병 여부를 예측하는 모델**을 구축하는 것을 목표로 합니다.

## 3. 데이터 설명

* 제공되는 건강검진 데이터는 22개의 임상 변수와, baseline 및 endpoint 시점의 건강검진 일자 변수 2개로 구성되어 있습니다.

|번호|변수명|설명|
|:---:|:---:|:---:|
|01|`gender`|성별|
|02|`age`|나이|
|03|**`date`**|**Baseline 건강검진 일자**|
|04|`Ht`|신장|
|05|`Wt`|체중|
|06|`BMI`|체질량지수(BMI)|
|07|`SBP`|수축기혈압|
|08|`DBP`|이완기혈압|
|09|`PR`|맥박|
|10|**`HbA1c`**|**당화혈색소**|
|11|**`FBG`**|**공복혈당**|
|12|`TC`|총콜레스테롤|
|13|`TG`|중성지방|
|14|`LDL`|LDL 콜레스테롤|
|15|`HDL`|HDL 콜레스테롤|
|16|`Alb`|알부민|
|17|`BUN`|혈중요소질소|
|18|`Cr`|크레아티닌|
|19|`CrCl`|크레아티닌 청소율|
|20|`AST`|아스파테이트아미노전이효소|
|21|`ALT`|알라닌아미노전이효소|
|22|`GGT`|감마글루타밀전이효소|
|23|`ALP`|알칼리인산분해효소|
|24|**`date_E`**|**Endpoint 건강검진 일자**|

* 당뇨병 진단 기준: (1) 공복혈당(FBG) 126 mg/dL 이상, 또는 (2) 당화혈색소(HbA1c) 6.5% 이상

## 4. 평가 지표
* AUC_ROC