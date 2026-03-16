# ============================================================
# COHORT IDENTIFICATION: ED VISITS FOR RL PROJECT
# ADS599 Capstone - MIMIC-IV
# ============================================================
# This script builds the base patient cohort and saves it to BigQuery
# as ads-599-final.rl_project.cohort_base so all feature queries can
# reference it without repeating the full CTE chain.
#
# Run this script first before any other pipeline notebooks.
#
# Pathway Labels:
#   ED_DIRECT_ICU          : ED → ICU (no intermediate ward)
#   ED_WARD_ICU            : ED → Ward → ICU (delayed escalation)
#   ED_WARD_DISCHARGE      : ED → Ward → Discharged (no ICU)
#   ED_DISCHARGE_STABLE    : ED → Home, no ICU return or death within 72h
#   ED_DISCHARGE_RETURN_ICU: ED → Home → ED → ICU within 72h
#   ED_DISCHARGE_DIED_72H  : ED → Home → Died within 72h
#
# Inclusion: disposition IN ('HOME', 'ADMITTED') only.
# Excluded: TRANSFER (no follow-up data), EXPIRED (died in ED, out of scope),
#           LEFT WITHOUT BEING SEEN / LEFT AGAINST MEDICAL ADVICE / ELOPED
#           (patient-driven departure, not a provider disposition decision).
# ============================================================

from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import warnings
warnings.filterwarnings(action='ignore', message="Your application has authenticated using end user")

client = bigquery.Client(project="ads-599-final")

query = """
-- ============================================================
-- COHORT IDENTIFICATION: ED VISITS FOR RL PROJECT
-- ADS599 Capstone - MIMIC-IV
-- ============================================================
-- Pathway Labels:
--   ED_DIRECT_ICU          : ED → ICU (no intermediate ward)
--   ED_WARD_ICU            : ED → Ward → ICU (delayed escalation)
--   ED_WARD_DISCHARGE      : ED → Ward → Discharged (no ICU)
--   ED_DISCHARGE_STABLE    : ED → Home, no ICU return or death within 72h
--   ED_DISCHARGE_RETURN_ICU: ED → Home → ED → ICU within 72h
--   ED_DISCHARGE_DIED_72H  : ED → Home → Died within 72h
--
-- Inclusion: disposition IN ('HOME', 'ADMITTED') only.
-- Excluded: TRANSFER (no follow-up data), EXPIRED (died in ED, out of scope),
--           LEFT WITHOUT BEING SEEN / LEFT AGAINST MEDICAL ADVICE / ELOPED
--           (patient-driven departure, not a provider disposition decision).
-- ============================================================

WITH

-- ── 1. All ED visits (scoped to HOME or ADMITTED dispositions) ────────────────
ed_base AS (
  SELECT
    e.subject_id,
    e.stay_id         AS ed_stay_id,
    e.hadm_id,
    e.intime          AS ed_intime,
    e.outtime         AS ed_outtime,
    e.disposition,
    e.race,
    e.arrival_transport
  FROM `physionet-data.mimiciv_ed.edstays` e
  WHERE e.disposition IN ('HOME', 'ADMITTED')
),

-- ── 2. ICU care unit names (MIMIC-IV transfers table values) ─────────────────
icu_units AS (
  SELECT careunit FROM UNNEST([
    'Medical Intensive Care Unit (MICU)',
    'Surgical Intensive Care Unit (SICU)',
    'Cardiac Vascular Intensive Care Unit (CVICU)',
    'Medical/Surgical Intensive Care Unit (MICU/SICU)',
    'Coronary Care Unit (CCU)',
    'Trauma SICU (TSICU)',
    'Neuro Surgical Intensive Care Unit (Neuro SICU)',
    'Neuro Intermediate'
  ]) AS careunit
),

-- ── 3. Transfer events after the ED (excluding ED unit itself, excluding discharge events) ──
transfers_flagged AS (
  SELECT
    t.subject_id,
    t.hadm_id,
    t.careunit,
    t.intime,
    CASE WHEN u.careunit IS NOT NULL THEN 1 ELSE 0 END AS is_icu
  FROM `physionet-data.mimiciv_3_1_hosp.transfers` t
  LEFT JOIN icu_units u USING (careunit)
  WHERE t.eventtype != 'discharge'
    AND (t.careunit NOT LIKE '%Emergency%' OR t.careunit IS NULL)
),

-- ── 4. Per admission: ever reach ICU, and when? ───────────────────────────────
hadm_icu_summary AS (
  SELECT
    hadm_id,
    MAX(is_icu)                               AS ever_icu,
    MIN(CASE WHEN is_icu = 1 THEN intime END) AS first_icu_intime
  FROM transfers_flagged
  GROUP BY hadm_id
),

-- ── 5. Per admission: what was the FIRST non-ED care unit? ───────────────────
hadm_first_unit AS (
  SELECT
    tf.hadm_id,
    tf.careunit AS first_careunit,
    tf.is_icu   AS first_is_icu
  FROM transfers_flagged tf
  INNER JOIN (
    SELECT hadm_id, MIN(intime) AS min_intime
    FROM transfers_flagged
    GROUP BY hadm_id
  ) fm ON tf.hadm_id = fm.hadm_id AND tf.intime = fm.min_intime
),

-- ── 6. Primary pathway per ED visit ──────────────────────────────────────────
ed_pathway AS (
  SELECT
    eb.subject_id,
    eb.ed_stay_id,
    eb.hadm_id,
    eb.ed_intime,
    eb.ed_outtime,
    eb.disposition,
    eb.race,
    eb.arrival_transport,
    his.ever_icu,
    his.first_icu_intime,
    hfu.first_careunit,
    CASE
      WHEN eb.disposition = 'ADMITTED' AND his.ever_icu = 1 AND hfu.first_is_icu = 1
        THEN 'ED_DIRECT_ICU'
      WHEN eb.disposition = 'ADMITTED' AND his.ever_icu = 1 AND COALESCE(hfu.first_is_icu, 0) = 0
        THEN 'ED_WARD_ICU'
      WHEN eb.disposition = 'ADMITTED' AND COALESCE(his.ever_icu, 0) = 0
        THEN 'ED_WARD_DISCHARGE'
      WHEN eb.disposition = 'HOME'
        THEN 'ED_DISCHARGE'
      ELSE 'UNKNOWN'
    END AS base_pathway
  FROM ed_base eb
  LEFT JOIN hadm_icu_summary his ON eb.hadm_id = his.hadm_id
  LEFT JOIN hadm_first_unit  hfu ON eb.hadm_id = hfu.hadm_id
),

-- ── 7. Bounced back: discharge → return ED → ICU within 72h ─────────────────
bounced_icu AS (
  SELECT DISTINCT ep1.ed_stay_id AS original_ed_stay_id
  FROM ed_pathway ep1
  INNER JOIN ed_pathway ep2
    ON  ep1.subject_id    = ep2.subject_id
    AND ep2.ed_stay_id   != ep1.ed_stay_id
    AND ep2.ed_intime    >= ep1.ed_outtime
    AND TIMESTAMP_DIFF(ep2.ed_intime, ep1.ed_outtime, HOUR) <= 72
  WHERE ep1.base_pathway = 'ED_DISCHARGE'
    AND ep2.base_pathway IN ('ED_DIRECT_ICU', 'ED_WARD_ICU')
),

-- ── 8. Died within 72h of ED discharge ───────────────────────────────────────
died_72h AS (
  SELECT ep.ed_stay_id
  FROM ed_pathway ep
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON ep.subject_id = p.subject_id
  WHERE ep.base_pathway = 'ED_DISCHARGE'
    AND p.dod IS NOT NULL
    AND DATE_DIFF(p.dod, DATE(ep.ed_outtime), DAY) BETWEEN 0 AND 3
)

-- ── 9. Final labeled cohort ───────────────────────────────────────────────────
SELECT
  ep.subject_id,
  ep.ed_stay_id,
  ep.hadm_id,
  ep.ed_intime,
  ep.ed_outtime,
  ep.disposition,
  ep.race,
  ep.arrival_transport,
  ep.first_careunit,
  ep.first_icu_intime,
  ep.base_pathway,
  CASE
    WHEN ep.base_pathway IN ('ED_DIRECT_ICU', 'ED_WARD_ICU', 'ED_WARD_DISCHARGE')
      THEN ep.base_pathway
    WHEN ep.base_pathway = 'ED_DISCHARGE' AND bi.original_ed_stay_id IS NOT NULL
      THEN 'ED_DISCHARGE_RETURN_ICU'
    WHEN ep.base_pathway = 'ED_DISCHARGE' AND d72.ed_stay_id IS NOT NULL
      THEN 'ED_DISCHARGE_DIED_72H'
    WHEN ep.base_pathway = 'ED_DISCHARGE'
      THEN 'ED_DISCHARGE_STABLE'
    ELSE 'UNKNOWN'
  END AS cohort_label,
  p.gender,
  p.anchor_age,
  p.anchor_year,
  p.anchor_age + (EXTRACT(YEAR FROM ep.ed_intime) - p.anchor_year) AS age_at_visit,
  p.dod,
  a.admittime,
  a.dischtime,
  a.admission_type,
  a.discharge_location,
  a.insurance,
  a.language,
  a.marital_status
FROM ed_pathway ep
INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
  ON ep.subject_id = p.subject_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON ep.hadm_id = a.hadm_id
LEFT JOIN bounced_icu bi  ON ep.ed_stay_id = bi.original_ed_stay_id
LEFT JOIN died_72h    d72 ON ep.ed_stay_id = d72.ed_stay_id
ORDER BY ep.subject_id, ep.ed_intime
"""

# Create dataset if needed
dataset_ref = client.dataset("rl_project", project="ads-599-final")
try:
    client.get_dataset(dataset_ref)
    print("Dataset rl_project already exists")
except NotFound:
    from google.cloud.bigquery import Dataset
    ds = Dataset(dataset_ref)
    ds.location = "US"
    client.create_dataset(ds)
    print("Created dataset: ads-599-final.rl_project")

# Save cohort to BigQuery
job_config = bigquery.QueryJobConfig(
    destination="ads-599-final.rl_project.cohort_base",
    write_disposition="WRITE_TRUNCATE"
)
client.query(query, job_config=job_config).result()
print("Cohort saved to BigQuery: ads-599-final.rl_project.cohort_base")

# Validation
df_cohort = client.query("SELECT * FROM `ads-599-final.rl_project.cohort_base`").to_dataframe()
print(f"\nShape: {df_cohort.shape}")
print(f"Null race count: {df_cohort['race'].isna().sum()}  (should be 0)")
print(f"Total ED visits: {len(df_cohort):,}")
print("\n--- Cohort Label Distribution ---")
print(df_cohort['cohort_label'].value_counts())
unknown = (df_cohort['cohort_label'] == 'UNKNOWN').sum()
print(f"\nUNKNOWN labels: {unknown}  (should be 0)")
print("\nCohort base created and saved successfully.")
