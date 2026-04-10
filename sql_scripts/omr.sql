SELECT DISTINCT
  o.subject_id,
  o.chartdate,
  o.result_name,
  o.result_value
FROM `physionet-data.mimiciv_3_1_hosp.omr` AS o
WHERE o.subject_id IN (
  SELECT DISTINCT subject_id
  FROM `physionet-data.mimiciv_ed.edstays`
)
AND o.result_name NOT LIKE 'BMI%'
AND o.result_name NOT LIKE 'Blood Pressure%'
AND o.result_name NOT LIKE 'eGFR'
