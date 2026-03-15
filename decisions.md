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

## Training Data Structure

Each row is a `(visit_id, time_step)` tuple representing the full patient state and the action taken during that time step. The number of rows per visit varies based on visit length; padding happens in the model dataloader at training time, not in the stored dataset.

| visit_id | time_step | temperature | culture_ordered | lab_hematology_blood | lab_chemistry_blood | administer_medication | start_medication | stop_medication | rate_change | order_culture | order_lab | observe | reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v001 | 1 | 38.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| v001 | 2 | 38.5 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| v001 | 3 | 38.8 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| v001 | 4 | 39.1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| v001 | 5 | 38.6 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | -1 |
| v002 | 1 | 37.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| v002 | 2 | 37.3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| v002 | 48 | 37.0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |

Action columns are mutually exclusive — exactly one is `1` per time step. State columns reflect the patient's known state at the start of that time step; values carry forward from prior steps until updated.

# Actions

## Microbiology / Culture Orders

Microbiology cultures are represented as a **single binary action: order culture (yes/no)**. We do not enumerate specific culture types (blood, urine, sputum, etc.) as separate actions. The clinically meaningful decision is whether to send cultures at all — the specific type is a downstream detail driven by clinical context that we do not need the agent to learn separately.

### Rationale
Cultures ordered in the ER are a strong signal of physician suspicion of infection at triage, making them a meaningful action to include. We found ~160k unique ED stays across the full MIMIC-IV ED cohort that had at least one culture drawn during the ER visit, split between patients who were subsequently admitted (~150k rows) and patients discharged home (~50k rows).

## Medication Actions

Medication administration is represented as **4 discrete actions**. For admitted patients, these are derived from the `event_txt` column of the `hosp.emar` table. For ED-only patients (and ED-phase events for admitted patients), Pyxis dispenses from `ed.pyxis` map to `administer_medication` — the only action available during an ED stay, since drips, rate changes, and stops are inpatient concepts.

| Action | emar `event_txt` values mapped | Clinical meaning |
|---|---|---|
| `administer_medication` | Administered, Partial Administered, Delayed Administered, Administered Bolus from IV Drip, Administered in Other Location | One-time dose (pill, injection, bolus, IV push) |
| `start_medication` | Started, Restarted | Begin continuous infusion or drip. Restart is collapsed here — the agent's decision is identical to starting; it is a documentation artifact of a prior stop. |
| `stop_medication` | Stopped, Stopped As Directed, Stopped - Unscheduled, Stopped in Other Location | Discontinue an active continuous infusion or drip |
| `rate_change` | Rate Change | Titrate dose of an active drip without stopping it |
| *(excluded)* | Flushed, Not Flushed, Flushed in Other Location | IV line maintenance — not a medication decision |
| *(excluded)* | Applied, Removed, Removed Existing / Applied New, Not Removed | Topical/patch administration — different care context, excluded for simplicity |
| *(excluded)* | Assessed, Not Assessed, Confirmed, Not Confirmed | Documentation and verification events — not clinical actions |
| *(excluded)* | Not Given, Not Applied, Not Given per Sliding Scale | Non-events — medication was not administered; documentation artifacts |
| *(excluded)* | Hold Dose | Intentional skip of a scheduled dose — not the same as stopping; excluded to avoid ambiguity with `stop_medication` |
| *(excluded)* | Infusion Reconciliation, in Other Location | Administrative reconciliation events |
| *(excluded)* | Delayed, Not Started, Not Stopped | Incomplete or cancelled intentions |

### Rationale

The `administer_medication` / `start_medication` distinction captures the clinically important difference between a one-time dose and the initiation of a continuous drip. Drip management (start, stop, titrate) is particularly relevant for ICU escalation decisions — vasopressor initiation and titration are strong signals of patient acuity.

`rate_change` is retained as a separate action rather than folded into `start_medication` because titration frequency is itself a meaningful signal: patients requiring repeated upward titrations are deteriorating, while downward titration may precede successful discharge. Merging these would obscure that pattern.

### Medication State Representation

The active medication state (what drugs are currently running) is represented separately in the state vector, not through the action space. This handles the case where multiple medications are active simultaneously — each drug or drug class has its own state flag rather than being encoded as a compound action. The specific design of the medication state features is deferred to the feature engineering phase (Stage 2.1) and will be documented here once finalized.


## Lab Orders

Labs are represented as **19 discrete actions**, one per unique combination of `category` and `fluid` present in the ED cohort. This replaces the prior design of 35 individual actions per unique `label`.

### Rationale for Category × Fluid Action Space

The MIMIC-IV lab data includes a `category` field (e.g., "Hematology", "Chemistry") and a `fluid` field (e.g., "Blood", "Urine") that together describe the type of lab ordered at a higher level of abstraction than the individual `label`. Grouping by these two fields yields 19 unique combinations, reducing the action space from 35 to 19 while still capturing the clinically meaningful distinctions between different lab types (e.g., blood chemistry vs. urine chemistry are genuinely different ordering decisions).

**Limitation:** Collapsing to category × fluid means the agent cannot learn to target a single specific test within a group — it acts on the combination as a whole. In practice, individual tests within the same category × fluid group tend to be ordered together, so this is an acceptable tradeoff for a more tractable action space.

### Lab State Representation

Each category × fluid combination maps to a **single 3-state ordinal feature** in the state vector:

- `0` = not ordered
- `1` = ordered, result normal
- `2` = ordered, result abnormal

**Grouping within a time step:** When multiple individual lab labels belonging to the same category × fluid combination are resulted at the same time, they are collapsed into a single observation. If all results are normal, the feature is `1`. If any result within the group is abnormal, the feature takes the abnormal value (`2`) — the worst-case result governs. This ensures that a group with one flagged value is not incorrectly recorded as normal.

**Rationale for worst-case aggregation:** A clinician reviewing lab results acts on the most alarming result present. Encoding the abnormal flag when any component is abnormal mirrors this clinical decision-making process and avoids information loss that could misdirect the agent's learned policy.

**Limitation:** Worst-case aggregation loses information about *which specific test within the category × fluid group* was abnormal. In cases where a group has both critically abnormal and incidentally abnormal values, the agent cannot distinguish these scenarios from the state alone. This is a known limitation of the grouped representation.

**Time decay:** A decay variable is included in the state for each category × fluid action to represent how stale the most recent result is. Results observed long ago carry less clinical relevance than recent ones, and the decay term allows the agent to learn that old normal results should not preclude re-ordering if enough time has passed.

**Most recent result update:** At each time step, the action and result features reflect only the most recent observation for that category × fluid combination — prior results are not accumulated. This keeps the state vector fixed-size and ensures the agent is always acting on the current best information.

### Outlier Handling

Stays in the extreme tail of the length-of-stay distribution will be removed from the training cohort. Exceptionally long hospital stays introduce non-representative trajectories with unusual lab ordering patterns driven by chronic care rather than acute decision-making, which can distort the learned policy. A length-of-stay cutoff (e.g., 95th or 99th percentile) will be applied during cohort construction; the exact threshold will be determined during EDA.

# Rewards

## Estimating values of rewards