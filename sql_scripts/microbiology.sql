SELECT
  c.subject_id,
  c.hadm_id,
  c.ed_stay_id,
  m.microevent_id,
  m.micro_specimen_id,
  m.charttime,
  m.spec_itemid,
  m.spec_type_desc,
  m.test_itemid,
  m.test_name,
  m.org_name,
  m.comments,
  m.storetime,
  IF(m.storetime IS NOT NULL
     AND CAST(m.storetime AS DATETIME) < c.ed_outtime, 1, 0) AS result_available_in_er,
  c.disposition,
  c.cohort_label
FROM `{PROJECT_NAME}.rl_project.cohort_base` c
INNER JOIN `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
  ON c.subject_id = m.subject_id
WHERE m.charttime IS NOT NULL
  AND m.charttime >= c.ed_intime
  AND m.charttime < CASE
    WHEN c.hadm_id IS NOT NULL AND c.first_icu_intime IS NOT NULL
      THEN c.first_icu_intime
    WHEN c.hadm_id IS NOT NULL AND c.first_icu_intime IS NULL
      THEN c.dischtime
    ELSE c.ed_outtime
  END
ORDER BY c.subject_id, c.ed_stay_id, m.charttime
