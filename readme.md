# Readme

## 소개

AIIDE 2021 story control

## 설치 방법

### 요구 사항

```
# data (51.2 MB)

```

## 파일 구성


### 저장소 구조

```bash
./assets/                # readme 에 필요한 이미지 저장

```

## 데이터 소개

```python
./data/         # 전체 데이터
    ./train_dataset/           

```

data에 대한 argument 는 arguments.py 의 DataTrainingArguments 에서 확인 가능합니다. 

# 훈련, 평가, 추론

### train

train.py 

```
# 학습 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --do_train
```

### eval

```
python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### inference


* 학습한 모델의  
```
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ./data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict
```

### How to submit

inference.py 파일을 위 예시처럼 --do_predict 으로 실행하면 --output_dir 위치에 predictions.json 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

# Citation