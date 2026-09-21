# `ui_*` suites: accuracy per system (uncalibrated decisions, bootstrap 95% intervals)

## ui_click

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.993 [0.983, 1.000] | 300 |

## ui_done

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.825 [0.770, 0.875] | 200 |

## ui_page

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.823 [0.780, 0.867] | 300 |

## ui_reason

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.910 [0.850, 0.960] | 100 |

## ui_state

| System | same items as the hosted models | n | all items | n |
| --- | --- | --- | --- | --- |
| open 4B model, read | - | - | 0.922 [0.898, 0.944] | 500 |

## ui_state, by @id_suffix (accuracy, items)

| System | checkbox_checked | cookie_banner | dialog_open | error_shown | primary_disabled | signed_in |
| --- | --- | --- | --- | --- | --- | --- |
| open 4B model, read | 0.97 (64) | 1.00 (80) | 0.99 (76) | 1.00 (91) | 0.69 (90) | 0.92 (99) |

## ui_done, by @label (accuracy, items)

| System | False | True |
| --- | --- | --- |
| open 4B model, read | 1.00 (100) | 0.65 (100) |

## ui_reason, by @id_kind (accuracy, items)

| System | pricing | slots | stock |
| --- | --- | --- | --- |
| open 4B model, read | 0.94 (34) | 0.79 (33) | 1.00 (33) |

## ui_page, by @label (accuracy, items)

| System | article | cart | search_results | settings | sign_in |
| --- | --- | --- | --- | --- | --- |
| open 4B model, read | 1.00 (60) | 0.75 (60) | 0.85 (60) | 0.80 (60) | 0.72 (60) |

