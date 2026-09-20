# Model card — lesion_unet

| field | value |
|---|---|
| tag | lesion_unet |
| architecture | U-Net, 4 levels, 32 base filters, 512x512, 4 sigmoid channels |
| loss | BCE (positive weight 10) + Dice, per channel |
| epochs | 60 |
| batch | 4 |
| lr | 0.0003 |
| train_n | 383 |
| valid_n | 149 |
| test_n | 225 |
| thresholds_chosen_on_valid_max_f1 | {"MA": 0.53, "HE": 0.89, "EX": 0.92, "SE": 0.47} |
| weights | /home/asvin_laptop/venus-cache/models/lesion_unet.weights.h5 |
| finished_at | 2026-09-20T14:17:15.686805+00:00 |
| total_minutes | 13.2 |
| aupr_note | full-pixel AUPR (all negatives); in-training monitoring used subsampled negatives |
| rescored_at | 2026-09-20T15:45:44.187003+00:00 |

## Test split (scored once)

```json
{
 "MA": {
  "aupr": 0.0791,
  "pixels": 58982400,
  "negatives_subsampled": false,
  "threshold": 0.53,
  "dice_at_threshold": 0.1657,
  "precision": 0.1172,
  "recall": 0.2827
 },
 "HE": {
  "aupr": 0.4492,
  "pixels": 58982400,
  "negatives_subsampled": false,
  "threshold": 0.89,
  "dice_at_threshold": 0.4649,
  "precision": 0.6059,
  "recall": 0.3771
 },
 "EX": {
  "aupr": 0.477,
  "pixels": 58982400,
  "negatives_subsampled": false,
  "threshold": 0.92,
  "dice_at_threshold": 0.4858,
  "precision": 0.6517,
  "recall": 0.3872
 },
 "SE": {
  "aupr": 0.2615,
  "pixels": 58982400,
  "negatives_subsampled": false,
  "threshold": 0.47,
  "dice_at_threshold": 0.3084,
  "precision": 0.2286,
  "recall": 0.4739
 }
}
```

## Training history

```json
[
 {
  "epoch": 40,
  "loss": 0.8797,
  "val": {
   "MA": {
    "aupr": 0.5327,
    "threshold": 0.1,
    "f1_at_threshold": 0.5208
   },
   "HE": {
    "aupr": 0.8214,
    "threshold": 0.25,
    "f1_at_threshold": 0.7791
   },
   "EX": {
    "aupr": 0.8098,
    "threshold": 0.1,
    "f1_at_threshold": 0.7667
   },
   "SE": {
    "aupr": 0.7756,
    "threshold": 0.1,
    "f1_at_threshold": 0.6785
   }
  },
  "mean_aupr": 0.7349
 },
 {
  "epoch": 45,
  "loss": 0.848,
  "val": {
   "MA": {
    "aupr": 0.5216,
    "threshold": 0.1,
    "f1_at_threshold": 0.4732
   },
   "HE": {
    "aupr": 0.8461,
    "threshold": 0.15,
    "f1_at_threshold": 0.7907
   },
   "EX": {
    "aupr": 0.8452,
    "threshold": 0.1,
    "f1_at_threshold": 0.7891
   },
   "SE": {
    "aupr": 0.8503,
    "threshold": 0.1,
    "f1_at_threshold": 0.7775
   }
  },
  "mean_aupr": 0.7658
 },
 {
  "epoch": 50,
  "loss": 0.8266,
  "val": {
   "MA": {
    "aupr": 0.5197,
    "threshold": 0.1,
    "f1_at_threshold": 0.4551
   },
   "HE": {
    "aupr": 0.8427,
    "threshold": 0.1,
    "f1_at_threshold": 0.7724
   },
   "EX": {
    "aupr": 0.8238,
    "threshold": 0.1,
    "f1_at_threshold": 0.7831
   },
   "SE": {
    "aupr": 0.831,
    "threshold": 0.1,
    "f1_at_threshold": 0.7602
   }
  },
  "mean_aupr": 0.7543
 },
 {
  "epoch": 55,
  "loss": 0.8198,
  "val": {
   "MA": {
    "aupr": 0.5486,
    "threshold": 0.1,
    "f1_at_threshold": 0.5017
   },
   "HE": {
    "aupr": 0.8633,
    "threshold": 0.1,
    "f1_at_threshold": 0.8034
   },
   "EX": {
    "aupr": 0.8242,
    "threshold": 0.1,
    "f1_at_threshold": 0.7798
   },
   "SE": {
    "aupr": 0.8368,
    "threshold": 0.1,
    "f1_at_threshold": 0.771
   }
  },
  "mean_aupr": 0.7682
 },
 {
  "epoch": 60,
  "loss": 0.8157,
  "val": {
   "MA": {
    "aupr": 0.5503,
    "threshold": 0.1,
    "f1_at_threshold": 0.501
   },
   "HE": {
    "aupr": 0.869,
    "threshold": 0.1,
    "f1_at_threshold": 0.812
   },
   "EX": {
    "aupr": 0.8266,
    "threshold": 0.1,
    "f1_at_threshold": 0.7839
   },
   "SE": {
    "aupr": 0.8414,
    "threshold": 0.1,
    "f1_at_threshold": 0.7809
   }
  },
  "mean_aupr": 0.7718
 }
]
```
(last 5 of 12 entries)
