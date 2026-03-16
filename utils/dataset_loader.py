from datasets import load_dataset

# Registry: logical name -> (hf_config_name, hf_split_name)
DATASETS = {
    "cohort_base": ("cohort_base", "base"),
    "cohort_with_triage": ("cohort_with_triage", "with_triage"),
    "vitals": ("vitals", "vitals"),
    "ed_only_meds": ("ed_only_meds","ed_only"),
    "meds_admitted": ("meds_admitted","meds_admit"),
    "ecg": ("ecg", "ecg"),
    "microbiology_events": ("microbiology_events", "microbiology_events"),
    "labs_base": ("labs_base", "train"),
    "medrecon": ("medrecon", "medrecon"),
    "omr": ("omr", "omr"),
    "radiology_details": ("radiology_details", "radiology_details"),
}

def mimic_loader(path: str, name: str):
    """Load a MIMIC-IV derived dataset from the HuggingFace Hub as a pandas DataFrame.

    Parameters
    ----------
    path : str
        HuggingFace repo path, e.g. "ADS599-Capstone/raw_data".
    name : str
        Logical dataset name. Must be a key in DATASETS. Valid options:
            "cohort_base" - base ED cohort, one row per ED visit
            "cohort_with_triage" - cohort with triage merged in, AGAINST ADVICE removed from discharge_location
            "vitals" - ED vital signs time series
            "ed_only_meds" - Pyxis dispense records for ED-only patients
            "meds_admitted" - eMAR medication records for admitted patients (includes pyxis records from prior to hospital admission)
            "ecg" - ECG machine report text (report_0–report_17 concatenated into one column)
            "microbiology_events" - culture orders and results for the patient cohort
            "labs_base" - lab results grouped by category × fluid
            "medrecon" - pre-arrival medication reconciliation
            "omr" - outpatient measurements (weight, height, eGFR).  BP and BMI removed
            "radiology_details" - radiology report text with exam_name and cpt_code

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> from utils.dataset_loader import mimic_loader
    >>> cohort = mimic_loader("ADS599-Capstone/raw_data", "cohort_with_triage")
    >>> vitals = mimic_loader("ADS599-Capstone/raw_data", "vitals")
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {list(DATASETS)}")
    config, split = DATASETS[name]
    return load_dataset(path, config, split=split).to_pandas()