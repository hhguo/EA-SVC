# EA-SVC
An implement of "Phonetic Posteriorgrams based Many-to-Many Singing Voice Conversion via Adversarial Training"

## Data prepare
1. PPG features (10ms frameshift)
2. F0 features (10ms frameshift)
3. Speaker embedding (One embedding per wav file)
4. Audio files (wave format, 24000 sample rate, mono)

## Write Configuration
Set path / directory or other configurations in .json files in directory "configs"
Rewrite your data load function in utils/dataset.py

## Model Training

Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/stage1.json
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/stage2.json
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/stage3.json
```
