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
),
radiology_with_detail AS (
  SELECT
    r.note_id,
    r.subject_id,
    r.hadm_id,
    r.charttime,
    r.text AS report_text,
    r.note_type,
    MAX(CASE WHEN rd.field_name = 'exam_name' THEN rd.field_value END) AS exam_name,
    MAX(CASE WHEN rd.field_name = 'cpt_code'  THEN rd.field_value END) AS cpt_code
  FROM `physionet-data.mimiciv_note.radiology` r
  LEFT JOIN `physionet-data.mimiciv_note.radiology_detail` rd
    ON r.note_id = rd.note_id
    AND rd.field_name IN ('exam_name', 'cpt_code')
  WHERE r.note_type = 'RR'
  GROUP BY r.note_id, r.subject_id, r.hadm_id, r.charttime, r.text, r.note_type
)
SELECT
  cs.ed_stay_id,
  rwd.subject_id,
  rwd.hadm_id,
  rwd.note_id,
  rwd.charttime,
  rwd.exam_name,
  rwd.cpt_code,
  rwd.report_text
FROM radiology_with_detail rwd
INNER JOIN cohort_subjects cs
  ON rwd.subject_id = cs.subject_id
  AND CAST(rwd.charttime AS DATETIME) >= cs.ed_intime
  AND CAST(rwd.charttime AS DATETIME) <= cs.window_end
ORDER BY cs.ed_stay_id, rwd.charttime
