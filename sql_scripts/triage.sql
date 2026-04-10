WITH cohort_subjects AS (
  SELECT ed_stay_id, subject_id
  FROM `{PROJECT_NAME}.rl_project.cohort_base`
)
SELECT
  t.stay_id   AS ed_stay_id,
  t.subject_id,
  t.temperature,
  t.heartrate,
  t.resprate,
  t.o2sat,
  t.sbp,
  t.dbp,
  t.pain,
  t.acuity,
  t.chiefcomplaint
FROM `physionet-data.mimiciv_ed.triage` t
INNER JOIN cohort_subjects cs ON t.stay_id = cs.ed_stay_id
ORDER BY t.stay_id
