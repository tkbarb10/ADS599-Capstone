SELECT
  e.subject_id,
  e.stay_id        AS ed_stay_id,
  e.hadm_id,
  e.disposition,
  p.charttime,
  p.med_rn,
  p.name           AS medication,
  TRUE             AS in_er
FROM `physionet-data.mimiciv_ed.edstays` e
JOIN `physionet-data.mimiciv_ed.pyxis` p
  ON e.stay_id = p.stay_id
WHERE e.disposition = 'HOME'
  AND e.hadm_id IS NULL
  AND p.charttime BETWEEN e.intime AND e.outtime
ORDER BY e.subject_id, e.stay_id, p.charttime
