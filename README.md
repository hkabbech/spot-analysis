# Analysis of spot dynamics

## Data structure

```
data
├── condition_A
│   ├── coords1.xml
│   ├── coords2.xml
│   ├── coords3.xml
│   ├── ....
│   └── parms.csv
└── condition_B
    ├── coords1.xml
    ├── coords2.xml
    ├── coords3.xml
    ├── ....
    └── parms.csv
```

## Requirements

```
pip install -r requirements.txt
```

## Run a unique condition:

```
python run.py data/condition_A
```

## Analysis
- Track length + gap length over the track
- MSD on a single spot (MSD curve (+ log scaled), estimations of alpha and D)
- Average of length, alpha and D
- time-averaged MSD curve on all spots
- VAC
- Gaussianity
