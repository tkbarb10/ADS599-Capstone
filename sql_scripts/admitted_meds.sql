WITH icu_times AS (
  SELECT
    hadm_id,
    CAST(MIN(intime) AS TIMESTAMP) AS icu_intime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  GROUP BY hadm_id
),

-- Pyxis dispenses for admitted patients while physically in the ED
admitted_pyxis AS (
  SELECT
    e.stay_id       AS ed_stay_id,
    e.subject_id,
    e.hadm_id,
    CAST(p.charttime AS TIMESTAMP) AS charttime,
    p.name          AS medication,
    CAST(NULL AS STRING) AS event_txt,
    TRUE            AS in_er
  FROM `physionet-data.mimiciv_ed.edstays` e
  JOIN `physionet-data.mimiciv_ed.pyxis` p
    ON e.stay_id = p.stay_id
  WHERE (e.disposition = 'ADMITTED' OR e.hadm_id IS NOT NULL)
    AND p.charttime BETWEEN e.intime AND e.outtime
),

-- eMAR ward records for admitted patients from ED departure up to ICU transfer
ward_emar AS (
  SELECT
    e.stay_id       AS ed_stay_id,
    e.subject_id,
    em.hadm_id,
    CAST(em.charttime AS TIMESTAMP) AS charttime,
    em.medication,
    em.event_txt,
    FALSE           AS in_er
  FROM `physionet-data.mimiciv_3_1_hosp.emar` em
  JOIN `physionet-data.mimiciv_ed.edstays` e
    ON em.hadm_id = e.hadm_id
  LEFT JOIN icu_times icu
    ON em.hadm_id = icu.hadm_id
  WHERE (e.disposition = 'ADMITTED' OR e.hadm_id IS NOT NULL)
    AND em.event_txt IN (
      'Administered',
      'Partial Administered',
      'Delayed Administered',
      'Administered Bolus from IV Drip',
      'Administered in Other Location',
      'Started',
      'Restarted',
      'Stopped',
      'Stopped As Directed',
      'Stopped - Unscheduled',
      'Stopped in Other Location',
      'Rate Change'
    )
    AND CAST(em.charttime AS TIMESTAMP) < COALESCE(icu.icu_intime, TIMESTAMP('9999-01-01'))
),

-- Pyxis dispenses for ED-only patients (never admitted)
ed_only_pyxis AS (
  SELECT
    e.stay_id       AS ed_stay_id,
    e.subject_id,
    e.hadm_id,
    CAST(p.charttime AS TIMESTAMP) AS charttime,
    p.name          AS medication,
    CAST(NULL AS STRING) AS event_txt,
    TRUE            AS in_er
  FROM `physionet-data.mimiciv_ed.edstays` e
  JOIN `physionet-data.mimiciv_ed.pyxis` p
    ON e.stay_id = p.stay_id
  WHERE e.disposition = 'HOME'
    AND e.hadm_id IS NULL
    AND p.charttime BETWEEN e.intime AND e.outtime
)

SELECT * FROM admitted_pyxis
UNION ALL
SELECT * FROM ward_emar
UNION ALL
SELECT * FROM ed_only_pyxis
ORDER BY subject_id, ed_stay_id, charttime
