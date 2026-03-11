# Who our patient cohort is

# How we define and set up the state object

## Microbiology / Culture State Features

Cultures are included in the state object as a single binary feature:

- **`culture_ordered`** (binary): was any culture ordered during this ER visit? This is the primary real-time state signal — the agent knows a culture has been sent.

A time-decay feature on the order time is deferred to the feature engineering phase.

`culture_positive` (whether the culture grew an organism) is retained in the dataset as a **retrospective training label** and for post-analysis, but is NOT a real-time state feature — only ~2% of culture results were available before ED discharge, so it carries no meaningful signal during the visit.

### `org_name` and `comments`

`org_name` alone is not a fully reliable indicator of a positive result — spot checks found cases where `org_name` is NULL but `comments` contains text describing positive growth. Both columns are retained in the dataset so that a robust `culture_positive` label can be derived in post-processing using `org_name IS NOT NULL` OR regex matches on `comments` (e.g., patterns like "positive", organism names, colony count descriptions). This gives us the best coverage without requiring manual review of every free-text entry.

## What the time step is

# Actions

## Microbiology / Culture Orders

Microbiology cultures are represented as a **single binary action: order culture (yes/no)**. We do not enumerate specific culture types (blood, urine, sputum, etc.) as separate actions. The clinically meaningful decision is whether to send cultures at all — the specific type is a downstream detail driven by clinical context that we do not need the agent to learn separately.

## Rationale
Cultures ordered in the ER are a strong signal of physician suspicion of infection at triage, making them a meaningful action to include. We found ~160k unique ED stays across the full MIMIC-IV ED cohort that had at least one culture drawn during the ER visit, split between patients who were subsequently admitted (~150k rows) and patients discharged home (~50k rows).

# Rewards

## Estimating values of rewards