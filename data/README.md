# Data Directory

Use symlinks to external datasets:

```bash
cd data/
ln -s ~/datasets/behavioral_data/experiment_1 exp1
ln -s ~/datasets/behavioral_data/experiment_2 exp2
```

Then reference in scripts:

```bash
python scripts/preprocess_poses.py --input data/exp1/raw_h5/ --output data/exp1/poses/
python scripts/train_kpms.py --pose-dir data/exp1/poses/ --video-dir data/exp1/videos/ ...
```
