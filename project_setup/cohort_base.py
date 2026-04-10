# ============================================================
# COHORT IDENTIFICATION: ED VISITS FOR RL PROJECT
# ADS599 Capstone - MIMIC-IV
# ============================================================
# This script builds the base patient cohort and saves it to BigQuery
# as in this format: project_id.dataset_id.table_id. All feature queries can
# reference it without repeating the full CTE chain.
#
# Run this script first before any other pipeline notebooks.
#
# Pathway Labels:
#   ED_DIRECT_ICU: ED → ICU (no intermediate ward)
#   ED_WARD_ICU: ED → Ward → ICU (delayed escalation)
#   ED_WARD_DISCHARGE: ED → Ward → Discharged (no ICU)
#   ED_DISCHARGE_STABLE: ED → Home, no ICU return or death within 72h
#   ED_DISCHARGE_RETURN_ICU: ED → Home → ED → ICU within 72h
#   ED_DISCHARGE_DIED_72H: ED → Home → Died within 72h
#
# Inclusion: disposition IN ('HOME', 'ADMITTED') only.
# Excluded: TRANSFER (no follow-up data), EXPIRED (died in ED, out of scope),
#           LEFT WITHOUT BEING SEEN / LEFT AGAINST MEDICAL ADVICE / ELOPED
#           (patient-driven departure, not a provider disposition decision).
# ============================================================

from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import warnings
from dotenv import load_dotenv
import os
warnings.filterwarnings(action='ignore', message="Your application has authenticated using end user")

load_dotenv()

PROJECT_NAME = os.environ.get('PROJECT_NAME')
if not PROJECT_NAME:
    raise ValueError("Need to set PROJECT_NAME env variable to initiate big query client")

client = bigquery.Client(project=PROJECT_NAME)

query = """
-- ============================================================
-- COHORT IDENTIFICATION: ED VISITS FOR RL PROJECT
-- ADS599 Capstone - MIMIC-IV
-- ============================================================
-- Pathway Labels:
--   ED_DIRECT_ICU: ED → ICU (no intermediate ward)
--   ED_WARD_ICU: ED → Ward → ICU (delayed escalation)
--   ED_WARD_DISCHARGE: ED → Ward → Discharged (no ICU)
--   ED_DISCHARGE_STABLE: ED → Home, no ICU return or death within 72h
--   ED_DISCHARGE_RETURN_ICU: ED → Home → ED → ICU within 72h
--   ED_DISCHARGE_DIED_72H: ED → Home → Died within 72h
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

-- ── 2. Transfer events after the ED (for first_careunit tracking) ────────────
transfers_post_ed AS (
  SELECT
    t.subject_id,
    t.hadm_id,
    t.careunit,
    t.intime
  FROM `physionet-data.mimiciv_3_1_hosp.transfers` t
  WHERE t.eventtype != 'discharge'
    AND (t.careunit NOT LIKE '%Emergency%' OR t.careunit IS NULL)
),

-- ── 3. ICU summary from icustays (authoritative clinical ICU management) ──────
-- Uses icu.icustays instead of care unit name matching — a patient placed in an
-- ICU bed for capacity reasons (not clinical ICU management) is excluded here.
hadm_icu_summary AS (
  SELECT
    hadm_id,
    1           AS ever_icu,
    MIN(intime) AS first_icu_intime
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  GROUP BY hadm_id
),

-- ── 4. Per admission: what was the FIRST non-ED care unit? ───────────────────
hadm_first_unit AS (
  SELECT
    tf.hadm_id,
    tf.careunit AS first_careunit,
    CASE WHEN tf.intime = his.first_icu_intime THEN 1 ELSE 0 END AS first_is_icu
  FROM transfers_post_ed tf
  INNER JOIN (
    SELECT hadm_id, MIN(intime) AS min_intime
    FROM transfers_post_ed
    GROUP BY hadm_id
  ) fm ON tf.hadm_id = fm.hadm_id AND tf.intime = fm.min_intime
  LEFT JOIN hadm_icu_summary his ON tf.hadm_id = his.hadm_id
),

-- ── 5. Primary pathway per ED visit ──────────────────────────────────────────
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

-- ── 6. Bounced back: discharge → return ED → ICU within 72h ─────────────────
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

-- ── 7. Died within 72h of ED discharge ───────────────────────────────────────
died_72h AS (
  SELECT ep.ed_stay_id
  FROM ed_pathway ep
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON ep.subject_id = p.subject_id
  WHERE ep.base_pathway = 'ED_DISCHARGE'
    AND p.dod IS NOT NULL
    AND DATE_DIFF(p.dod, DATE(ep.ed_outtime), DAY) BETWEEN 0 AND 3
)

-- ── 8. Final labeled cohort ───────────────────────────────────────────────────
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

def run(dataset_name: str, table_name: str) -> None:
    """Build the base cohort and save it to BigQuery.

    Args:
        dataset_name: BQ dataset to write into (e.g. 'rl_project')
        table_name:   BQ table name (e.g. 'cohort_base')
    """
    destination = f"{PROJECT_NAME}.{dataset_name}.{table_name}"

    # Create dataset if needed
    dataset_ref = client.dataset(dataset_name, project=PROJECT_NAME)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_name} already exists")
    except NotFound:
        from google.cloud.bigquery import Dataset
        ds = Dataset(dataset_ref)
        ds.location = "US"
        client.create_dataset(ds)
        print(f"Created dataset: {PROJECT_NAME}.{dataset_name}")

    # Write destination to .env so downstream scripts can reference it
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    with open(env_path, 'a') as f:
        f.write(f"\nCOHORT_DESTINATION={destination}\n")
    print(f"Written to .env: COHORT_DESTINATION={destination}")

    # Save cohort to BigQuery
    job_config = bigquery.QueryJobConfig(
        destination=destination,
        write_disposition="WRITE_TRUNCATE"
    )
    client.query(query, job_config=job_config).result()
    print(f"Cohort saved to BigQuery: {destination}")

    # Validation
    df_cohort = client.query(f"SELECT * FROM `{destination}`").to_dataframe()
    print(f"\nShape: {df_cohort.shape}")
    print(f"Null race count: {df_cohort['race'].isna().sum()}  (should be 0)")
    print(f"Total ED visits: {len(df_cohort):,}")
    print("\n--- Cohort Label Distribution ---")
    print(df_cohort['cohort_label'].value_counts())
    unknown = (df_cohort['cohort_label'] == 'UNKNOWN').sum()
    print(f"\nUNKNOWN labels: {unknown}  (should be 0)")
    print("\nCohort base created and saved successfully.")


if __name__ == "__main__":
    run(dataset_name="rl_project", table_name="cohort_base")
