WITH cohort_subjects AS (
  SELECT ed_stay_id, subject_id
  FROM `{PROJECT_NAME}.rl_project.cohort_base`
)
SELECT
  m.stay_id       AS ed_stay_id,
  m.subject_id,
  m.charttime,
  m.name,
  m.gsn,
  m.ndc,
  m.etccode,
  m.etcdescription
FROM `physionet-data.mimiciv_ed.medrecon` m
INNER JOIN cohort_subjects cs ON m.stay_id = cs.ed_stay_id
ORDER BY m.stay_id, m.name
