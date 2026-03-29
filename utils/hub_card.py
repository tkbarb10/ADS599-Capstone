from huggingface_hub import DatasetCard

PHYSIONET_LICENSE = "PhysioNet Credentialed Health Data License 1.5.0 — https://physionet.org/content/mimiciv/view-license/3.1/"

PHYSIONET_YAML = {
    "license": "other",
    "license_name": "physionet-credentialed-health-data-1.5.0",
    "license_link": "https://physionet.org/content/mimiciv/view-license/3.1/",
}


def push_dataset_card(repo_id: str, config_name: str, split: str, data_dir: str, description: str):
    """Load the existing dataset card for repo_id, add/update the config entry and
    description section for this config, then push back — without overwriting other configs.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo, e.g. "ADS599-Capstone/raw_data".
    config_name : str
        The config_name used in push_to_hub, e.g. "vitals".
    split : str
        The split name used in Dataset.from_pandas, e.g. "vitals".
    data_dir : str
        The data_dir used in push_to_hub, e.g. "vitals". Used to build the parquet glob.
    description : str
        Human-readable description of this config.
    """
    try:
        card = DatasetCard.load(repo_id)
    except Exception:
        card = DatasetCard(f"---\n---\n\n# {repo_id.split('/')[-1]}\n")

    # Update top-level YAML fields
    for key, value in PHYSIONET_YAML.items():
        card.data[key] = value

    if not card.data.get("pretty_name"):
        card.data["pretty_name"] = repo_id.split("/")[-1]

    # Add/update this config's entry in the configs list
    new_config = {
        "config_name": config_name,
        "data_files": [{"split": split, "path": f"{data_dir}/*.parquet"}],
    }
    configs = list(card.data.get("configs") or [])
    configs = [c for c in configs if c.get("config_name") != config_name]
    configs.append(new_config)
    card.data["configs"] = configs

    # Append a description section if one doesn't already exist for this config
    section_header = f"## {config_name}"
    if section_header not in card.content:
        card.content = card.content.rstrip() + f"\n\n{section_header}\n\n{description}\n\n**License:** {PHYSIONET_LICENSE}\n"

    card.push_to_hub(repo_id)


def get_dataset_description(repo_id: str, config_name: str) -> str:
    """Return the description for a specific config from the dataset card README.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo, e.g. "ADS599-Capstone/raw_data".
    config_name : str
        The config whose description section to retrieve, e.g. "vitals".

    Returns
    -------
    str
        The text content of the ## config_name section, or an empty string if not found.
    """
    card = DatasetCard.load(repo_id)
    lines = card.content.split("\n")
    in_section = False
    section_lines = []
    for line in lines:
        if line.strip() == f"## {config_name}":
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            section_lines.append(line)
    return "\n".join(section_lines).strip()