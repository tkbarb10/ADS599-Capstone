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

### Rationale
Cultures ordered in the ER are a strong signal of physician suspicion of infection at triage, making them a meaningful action to include. We found ~160k unique ED stays across the full MIMIC-IV ED cohort that had at least one culture drawn during the ER visit, split between patients who were subsequently admitted (~150k rows) and patients discharged home (~50k rows).

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