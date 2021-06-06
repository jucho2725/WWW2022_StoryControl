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


```

# 훈련, 평가, 추론

### train

train.py 

```
# 학습 예시 (train_dataset 사용)
python train_scl.py --train_data_file data/eng_movieplot_3_dr_augderu.tsv --output_dir outputs/test_scl --overwrite_output_dir --do_train --num_train_epochs 10 --per_device_train_batch_size 8 --fp16
```

### eval

```
python train_gen.py --model_name_or_path ./outputs/test_scl/ --eval_data_file data/dev.tsv --output_dir outputs/test_scl --do_eval --per_device_eval_batch_size 8 --fp16
```

### inference


* 아직 안되어잇음  
```
```

### How to submit


# Citation