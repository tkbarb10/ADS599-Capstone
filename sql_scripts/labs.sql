-- Lab results from hosp.labevents for cohort patients.
-- labevents has no ed_stay_id — joined via subject_id + time window.
-- Window covers:
--   - ED-only patients: ed_intime to ed_outtime
--   - Admitted patients: ed_intime to first_icu_intime (if ICU transfer occurred) or dischtime
-- Labs ordered after ICU transfer are excluded from the window entirely.
-- order_time = charttime (when the order was placed / specimen collected)
-- result_time = storetime (when the result was filed)
-- ordered_location: 'ed' if order placed during ED window, 'ward' after ED discharge
-- result_after_icu_transfer: True if result_time >= first_icu_intime — lab was ordered before
--   ICU transfer but result came back after (retrospective label, not a state feature)
-- hadm_id is NULL for ED-only patients, populated for admitted patients.
-- NOTE: This query may take several minutes due to labevents table size.

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
