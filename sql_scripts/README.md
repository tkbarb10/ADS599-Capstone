# sql_scripts/

BigQuery SQL queries that extract raw MIMIC-IV data for the cohort. These are not run directly -- each query is loaded and executed by its corresponding pipeline script in `data_pipelines/pipeline_scripts/`.

---

## Scripts

| File | Description |
|---|---|
| `cohort_base.sql` | Defines the base patient cohort: adult ED visits with a provider disposition of home or admitted. Assigns pathway labels (e.g. `ED_DIRECT_ICU`, `ED_DISCHARGE_STABLE`). Used by `project_setup/cohort_base.py`. |
| `triage.sql` | Triage snapshot: chief complaint, acuity (ESI), pain score, arrival transport, and initial vital signs. |
| `vitals.sql` | Repeated vital sign measurements recorded during the ED visit. |
| `labs.sql` | Lab results from `hosp.labevents` joined to ED stays via subject_id + time window. Covers ED stay through ICU transfer (if applicable). |
| `microbiology.sql` | Microbiology culture results from `hosp.microbiologyevents`. |
| `ed_meds.sql` | Medications dispensed during the ED visit from `ed.pyxis`. |
| `admitted_meds.sql` | Medications administered after hospital admission (post-ED), used to construct the dispensed meds feature. |
| `medrecon.sql` | Pre-arrival medication reconciliation from `ed.medrecon` -- medications the patient was already taking before the ED visit. |
| `radiology_data.sql` | Radiology report records from `note.radiology` for cohort patients during their ED stay. |
| `ecg_data.sql` | ECG machine measurement records from `ecg.machine_measurements` for cohort patients. |
| `omr.sql` | Height and weight measurements from `hosp.omr` (outpatient/clinic records), used as static patient features. |

---

## Notes

- All queries reference `{PROJECT_NAME}.rl_project.cohort_base` as their base filter. The cohort table must exist in BigQuery before any query runs -- build it with `python -m project_setup.cohort_base`.
- `{PROJECT_NAME}` is a placeholder substituted at runtime from the `PROJECT_NAME` environment variable.
- Query results are not saved to disk; they are streamed directly into pandas DataFrames by the pipeline scripts and then pushed to HuggingFace.
