# admitted_meds_pipeline

Dispensed medication records for all cohort patients across two phases and two populations. ED phase (in_er=True): Pyxis dispenses for all patients (admitted and ED-only) while physically in the ED, derived from ed.pyxis. Ward phase (in_er=False): eMAR administration records for admitted patients from ED departure up to ICU transfer (ICU records excluded to prevent leakage), derived from hosp.emar. event_txt filtered to administration and rate-change events only. Medication names mapped to a 22-class drug vocabulary via regex. Each record includes minutes_into_stay (time since ED arrival) and time_step bucket (floor(minutes / time_block)).

# cohort_pipeline

Processed ED patient cohort derived from MIMIC-IV. One row per ED visit (ed_stay_id). Built from cohort_base (BigQuery) with the following post-processing steps: (1) AGAINST ADVICE discharge_location records removed — patient-driven departures are out of scope. (2) Consecutive ED visits sharing a hadm_id collapsed into single rows; second ed_stay_id stored in ed_stay_id_2. (3) stay_window_start/stay_window_end columns added covering ED arrival to final discharge. (4) ed_boarding_time_hrs added: hours in ED after admission decision (null for ED-only patients). Intended as the primary cohort table for feature engineering and RL state construction.

# ecg_data

ECG acuity labels derived from mimiciv_ecg.machine_measurements for cohort patients. Each of the 18 report columns is classified independently via regex with priority order: abnormal (2) > neutral (1) > normal (0) > empty (-1). Row-level max across all 18 columns gives ecg_acuity: 0=normal, 1=neutral/unknown, 2=abnormal. Patients with all 18 columns empty are assigned 1 (missing ECG ≠ normal ECG). Multiple ECGs per ED stay resolved to one row: highest acuity kept, ties broken by earliest ecg_time. Window covers ED arrival through hospital ward stay (capped at ICU transfer if applicable), with ±1 hour buffer for ECG machine clock drift.

# ed_meds_pipeline

Superseded by admitted_meds_pipeline. ED-only patient Pyxis records are now included in the combined dispensed_meds dataset (in_er=True rows where hadm_id IS NULL).

# labs_pipeline

Laboratory results from hosp.labevents for cohort patients during their stay window. Grouped by category x fluid (19 unique combinations) rather than individual test label to reduce action space. Each result includes the abnormal flag for worst-case aggregation. Intended as the primary lab state feature source.

# medrecon_pipeline

Medication reconciliation records from ed.medrecon — medications the patient was taking prior to ED arrival. One row per medication per ED visit. Represents pre-arrival medication state, not actions taken during the visit.

# microbiology_pipeline

Microbiology culture events from hosp.microbiologyevents for cohort patients during their stay window. Includes culture order time, specimen type, organism name, and antibiotic sensitivity results. culture_ordered is the real-time state signal; culture_positive is a retrospective label only (~2% of results available before ED discharge).

# omr_pipeline

Height and weight measurement records from hosp.omr for cohort patients — separated to take single measurements of height as an average of all available height measurements for a patient and retaining all separate weight measurements. Duplicates were dropped. Used to supplement ED and inpatient state features with recent baseline measurements.

# radiology_data

Radiology report text from mimiciv_note.radiology for cohort patients. Covers all imaging modalities (CXR, CT, MRI, ultrasound, etc.). Primary reports only (note_type=RR). Window covers ED arrival through hospital ward stay (capped at ICU transfer if applicable). hadm_id is NULL for ED-only patients and populated for admitted patients. exam_name and cpt_code included from radiology_detail to identify imaging modality.

# triage_pipeline

ED triage records from ed.triage. One row per ED visit. Includes ESI acuity level, chief complaint, and initial vital signs at triage. Includes back-to-back ED visits that are handled when merging with the cohort.

## cohort_with_triage

Primary cohort table with ED triage data merged in. One row per ED visit. Built by merging the processed cohort (AGAINST ADVICE removed, consecutive ED visits collapsed, stay_window and ed_boarding_time_hrs added) with ed.triage on ed_stay_id + subject_id. Triage rows for the second visit in a consecutive pair are dropped by the inner join. Intended as the main feature engineering input for the RL model.

# vitals_pipeline

ED vital signs time series from ed.vitalsign, joined to the cohort by ed_stay_id. One row per vitals measurement per ED visit. stay_id remapping applied so that vitals from a patient's second consecutive ED visit are attributed to the first (merged) stay_id.
