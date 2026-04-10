WITH cohort_subjects AS (
  SELECT
    ed_stay_id,
    subject_id,
    hadm_id,
    ed_intime,
    CASE
      WHEN hadm_id IS NOT NULL AND first_icu_intime IS NOT NULL
        THEN first_icu_intime
      WHEN hadm_id IS NOT NULL
        THEN dischtime
      ELSE ed_outtime
    END AS window_end
  FROM `{PROJECT_NAME}.rl_project.cohort_base`
)
SELECT
  cs.ed_stay_id,
  cs.subject_id,
  cs.hadm_id,
  mm.ecg_time,
  mm.report_0,
  mm.report_1,
  mm.report_2,
  mm.report_3,
  mm.report_4,
  mm.report_5,
  mm.report_6,
  mm.report_7,
  mm.report_8,
  mm.report_9,
  mm.report_10,
  mm.report_11,
  mm.report_12,
  mm.report_13,
  mm.report_14,
  mm.report_15,
  mm.report_16,
  mm.report_17
FROM `physionet-data.mimiciv_ecg.record_list` rl
INNER JOIN `physionet-data.mimiciv_ecg.machine_measurements` mm
  ON rl.study_id = mm.study_id
INNER JOIN cohort_subjects cs
  ON rl.subject_id = cs.subject_id
  AND CAST(mm.ecg_time AS DATETIME) >= DATETIME_SUB(cs.ed_intime, INTERVAL 1 HOUR)
  AND CAST(mm.ecg_time AS DATETIME) <= DATETIME_ADD(cs.window_end, INTERVAL 1 HOUR)
ORDER BY cs.ed_stay_id, mm.ecg_time