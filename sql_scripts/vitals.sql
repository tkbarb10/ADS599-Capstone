WITH cohort_subjects AS (
  SELECT ed_stay_id, subject_id
  FROM `{PROJECT_NAME}.rl_project.cohort_base`
)
SELECT
  v.stay_id   AS ed_stay_id,
  v.subject_id,
  v.charttime,
  v.temperature,
  v.heartrate,
  v.resprate,
  v.o2sat,
  v.sbp,
  v.dbp,
  v.rhythm,
  v.pain
FROM `physionet-data.mimiciv_ed.vitalsign` v
INNER JOIN cohort_subjects cs ON v.stay_id = cs.ed_stay_id
ORDER BY v.stay_id, v.charttime
