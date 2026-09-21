# `probe_*` suites: accuracy per system (uncalibrated decisions, bootstrap 95% intervals)

## probe_count

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.913 [0.867, 0.953] | 150 |

## probe_count_color

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.887 [0.833, 0.933] | 150 |

## probe_largest

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.520 [0.440, 0.600] | 150 |

## probe_spatial

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.940 [0.900, 0.973] | 150 |

## probe_stripes

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.653 [0.573, 0.727] | 150 |

## probe_text

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 1.000 [1.000, 1.000] | 150 |

## probe_count, by count (accuracy, items)

| System | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| open 4B model, read | 1.00 (19) | 1.00 (19) | 1.00 (19) | 0.95 (19) | 1.00 (19) | 0.89 (19) | 0.89 (18) | 0.56 (18) |

## probe_count_color, by label (accuracy, items)

| System | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open 4B model, read | 1.00 (22) | 1.00 (22) | 0.95 (22) | 0.90 (21) | 0.76 (21) | 0.86 (21) | 0.71 (21) |

## probe_largest, by ratio (accuracy, items)

| System | 1.15 | 1.3 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- |
| open 4B model, read | 0.32 (38) | 0.45 (38) | 0.59 (37) | 0.73 (37) |

## probe_stripes, by label (accuracy, items)

| System | diagonal_falling | diagonal_rising | horizontal | vertical |
| --- | --- | --- | --- | --- |
| open 4B model, read | 0.49 (37) | 0.11 (37) | 1.00 (38) | 1.00 (38) |

## probe_spatial, by asked_word (accuracy, items)

| System | above | below | left | right |
| --- | --- | --- | --- | --- |
| open 4B model, read | 1.00 (49) | 0.69 (26) | 0.97 (39) | 1.00 (36) |

