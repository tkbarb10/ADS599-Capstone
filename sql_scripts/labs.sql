WITH cohort_subjects AS (
  SELECT
    ed_stay_id,
    subject_id,
    hadm_id,
    ed_intime,
    ed_outtime,
    first_icu_intime,
    CASE
      WHEN hadm_id IS NOT NULL AND first_icu_intime IS NOT NULL THEN first_icu_intime
      WHEN hadm_id IS NOT NULL THEN dischtime
      ELSE ed_outtime
    END AS window_end
  FROM `{PROJECT_NAME}.rl_project.cohort_base`
)
SELECT
  cs.ed_stay_id,
  le.subject_id,
  cs.hadm_id,
  le.labevent_id,
  le.itemid,
  dl.label,
  dl.fluid,
  dl.category,
  le.charttime AS order_time,
  le.storetime AS result_time,
  le.flag,
  CASE
    WHEN le.charttime <= cs.ed_outtime THEN 'ed'
    ELSE 'ward'
  END AS ordered_location,
  CASE
    WHEN cs.first_icu_intime IS NOT NULL AND le.storetime >= cs.first_icu_intime THEN TRUE
    ELSE FALSE
  END AS result_after_icu_transfer
FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
INNER JOIN cohort_subjects cs
  ON le.subject_id = cs.subject_id
  AND le.charttime >= cs.ed_intime
  AND le.charttime < cs.window_end
INNER JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` dl
  ON le.itemid = dl.itemid
ORDER BY cs.ed_stay_id, le.charttime
