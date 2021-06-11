# Readme

## 소개

ICIDS 2021 story control

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


```

# 훈련, 평가, 추론

### train

train.py 

```
# 학습 예시 (train_dataset 사용)
python train_gen.py --train_data_file data/train_genre3_delex.tsv --output_dir outputs/test_scl --overwrite_output_dir --do_train --num_train_epochs 10 --per_device_train_batch_size 8 --fp16
```

### eval 까지 추가하기 

```
python train_gen.py --train_data_file data/dummy_genre3_delex.tsv --eval_data_file data/dummy_genre3_delex.tsv --output_dir outputs/test_traineval --overwrite_output_dir --do_train --do_eval --num_train_epochs 1 --per_device_train_batch_size 8 --fp16 --evaluation_strategy epoch --evaluation_metric ppl
```

### inference


* 아직 안되어잇음  
```
```

### How to submit


# Citation