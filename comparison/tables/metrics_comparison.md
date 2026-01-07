
# Baseline vs Improved Models Comparison

| Model     | Input Size | Dropout | Batch Size | Epochs | Loss Type      | Validation Dice | Validation Mean IoU |
|-----------|------------|---------|------------|--------|----------------|----------------|-------------------|
| Baseline  | 128x128    | 0.15    | 8          | 50     | BCE            | 0.63         | 0.75          |
| Improved  | 192x192    | 0.10    | 8          | 50     | BCE + Dice     | 0.92         | 0.86          |
