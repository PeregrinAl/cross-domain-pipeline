# Presentation-ready tables

## synthetic_source_only

variant,target_roc_auc,target_pr_auc,target_f1
raw_only,0.8976,0.6687,0.1818
tfr_only,0.8333,0.4771,0.1818
fused,0.9184,0.6933,0.1818

## synthetic_fused_sfda

variant,target_roc_auc,target_pr_auc,target_f1
fused_after_sfda,0.9792,0.8655,0.6667

## synthetic_event_level

variant,event_f1_iou_0p05,false_alarms_per_record_iou_0p05
sfda_after_fused,0.6667,0.0000
sfda_before_fused,0.6667,0.0000
source_only_fused,0.0000,1.0000

## paderborn_source_only

variant,target_roc_auc,target_pr_auc,target_f1
fused,1.0000,1.0000,1.0000
raw_only,1.0000,1.0000,1.0000
tfr_only,1.0000,1.0000,0.9995

## paderborn_fused_sfda

variant,before_roc_auc,before_pr_auc,before_f1,after_roc_auc,after_pr_auc,after_f1,before_target_cal_f1,after_target_cal_f1

## mimii_source_only

variant,target_roc_auc,target_pr_auc,target_f1
fused,0.6779,0.7437,0.6587
raw_only,0.6838,0.7464,0.6417
tfr_only,0.6697,0.7101,0.6660

## mimii_fused_sfda

variant,before_roc_auc,before_pr_auc,before_f1,after_roc_auc,after_pr_auc,after_f1,before_target_cal_f1,after_target_cal_f1
fused,0.6779,0.7437,0.6587,0.6598,0.7234,0.6608,0.4287,0.3618
