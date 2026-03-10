# ADS599 Capstone Project Plan

## Project Summary
Build an offline reinforcement learning agent that optimizes ED triage-to-disposition decisions using MIMIC-IV data. The agent learns a policy that maximizes correct ICU escalation, avoids unsafe discharges, and makes faster decisions without sacrificing quality.

---

## Stage 0: RL Design Lock-in *(must complete before data prep)*

> **Critical gate.** Everything downstream — which variables to pull, how to format time series, what to compute — depends on locking these in. The items below are rough drafts from prior work that must be finalized and documented in `decisions.md` before moving on.

### 0.1 Finalize Patient Cohort
- [ ] Confirm the 6 cohort labels and their inclusion/exclusion rationale
  - Note: The 399,573 visit count reflects cohort filtering (dispositions IN 'HOME', 'ADMITTED'; excludes TRANSFER, EXPIRED in ED, LWBS, AMA, ELOPED) from the full 400k+ ED visits — document this filter logic explicitly
- [ ] Decision on small adverse cohorts (224 `ED_DISCHARGE_RETURN_ICU` + 57 `ED_DISCHARGE_DIED_72H`): **keep as terminal penalty cases, no oversampling; N is statistically weak — require targeted evaluation**
- [ ] Define train/validation/test split strategy
  - Split by `subject_id` to prevent data leakage
  - Stratify by cohort label to preserve ICU case representation
  - Suggested: 70/15/15 — confirm proportions
  - Test set is **held out entirely** until final evaluation

**Rough draft cohort (from `data_queries.ipynb`):**
| Label | N |
|---|---|
| ED_DIRECT_ICU | 23,088 |
| ED_WARD_ICU | 8,533 |
| ED_WARD_DISCHARGE | 162,478 |
| ED_DISCHARGE_STABLE | 205,193 |
| ED_DISCHARGE_RETURN_ICU | 224 |
| ED_DISCHARGE_DIED_72H | 57 |

### 0.2 Define State Space
Lock in what the agent "sees" at each timestep. Break into:

**Static** (set at episode start):
- Demographics: age, gender, race, insurance, language, marital status
  - *Note: Protected attributes (race, gender, insurance) will be included and audited for bias during evaluation — see Stage 4*
- Arrival info: admission type, transport mode
- Triage: acuity/ESI score, chief complaint
- Pre-arrival medications (medrecon) grouped by therapeutic class

**Dynamic** (updated each timestep):
- Vitals: HR, RR, O2sat, SBP, DBP, temp, pain, rhythm
- Lab results (unique labs present in patient cohort — finalize list during Stage 1)
- ECG: PR interval, QRS, QTc, axes
- Per dynamic feature: current value, time-since-last, delta/velocity, running avg

**Action result flags** (per ordered test/medication):
- Status: 0 = Not Ordered, 1 = Ordered/Pending, 2 = Resulted
- `time_since_result`: continuous (0 if not yet resulted)
- `time_ordered_to_resulted`: continuous — captures turnaround time; potential source of decision-speed insight (e.g., lab processing delays outside provider control)

**Medications in state:**
- Pre-arrival medications: grouped by therapeutic class (from medrecon)
- Medications ordered by provider during stay: grouped by therapeutic class (from Pyxis, after cleaning)
- High cardinality (~1000 unique Pyxis values) requires grouping — see Stage 2.1

### 0.3 Define Action Space
- **Granularity decision deferred to design stage** — requires analysis of Pyxis/radiology cardinality
- Candidate coarse action set:
  - `check_vitals` — always available (no action masking)
  - `observe/wait` — always available
  - `order_labs`
  - `order_imaging`
  - `give_medication`
  - `admit_ward`
  - `admit_ICU`
  - `discharge_home`
- **Action masking**: mask already-completed actions — exceptions are `observe/wait` and `check_vitals`
- **Termination states**: `admit_ICU`, `discharge_home`, and potentially `deceased` — flag as open decision; depends on how MIMIC records in-ED deaths vs post-discharge deaths (see Open Decisions)
- Episode ends at ED disposition

### 0.4 Define Reward Function
- Terminal rewards:
  - **Highest positive**: admit to ICU when patient needed it (ED_DIRECT_ICU, ED_WARD_ICU)
  - Moderate positive: appropriate ward admission, appropriate discharge (STABLE)
  - **Highest penalty**: discharge → bounce back or death within 72h
  - **Small penalty**: delayed ICU escalation (ED_WARD_ICU vs ED_DIRECT_ICU — path was correct, timing was slow)
- Time-based reward component: faster correct final disposition earns higher reward than slower correct disposition
- Document all reward values explicitly in `decisions.md`

### 0.5 Select RL Algorithm
**Open decision — resolve after data exploration.** Regardless of algorithm chosen, the implementation must produce a **distribution over Q-values** (not a point estimate) to properly quantify uncertainty in policy recommendations.

| Algorithm | Pros | Cons |
|---|---|---|
| **DQN / Dueling DQN** | Well-studied; prior clinical RL literature; interpretable | Offline distributional shift; may overestimate Q on unseen actions |
| **DQN + Supervisor Network** | Constrains actions to clinically valid sequences; matches Liu et al. 2024 | More complex; supervisor network requires action sequence labels per patient (sequences of actions taken at each timestep, not outcomes — same data, different label) |
| **CQL (Conservative Q-Learning)** | Principled offline RL; explicitly handles distributional shift | Less clinical precedent; hyperparameter sensitive |
| **BCQ (Batch-Constrained Q-Learning)** | Strong offline RL guarantees; constrains policy near behavioral policy | May limit improvement over clinician baseline |

*Starting recommendation: DQN + Supervisor Network (aligned with cited literature). Consider adding CQL comparison if time permits.*

---

## Stage 1: Data Understanding

- [ ] Load all CSVs from `/data/` into working environment
- [ ] Audit each feature for missingness (overall + per cohort label)
- [ ] Identify outliers in continuous variables (vitals, labs) using clinical plausibility bounds
- [ ] Document temporal coverage per data source (ECG/radiology have known gaps)
- [ ] **Distribution of ED stay duration** per unique visit — this defines the time intervals to use for time series construction
- [ ] **Distribution of actions taken** per visit (how many labs ordered, how many imaging, how many meds) — informs action space granularity decision
- [ ] Check for duplicate records (Pyxis has known deduplication issues)
- [ ] Correlation analysis: key features vs outcome labels
- [ ] Visualize distributions of priority variables: vitals, labs, acuity score, time-to-ICU
- [ ] Finalize lab variable list (unique labs present in cohort — not all possible labs)

---

## Stage 2: Data Preparation

### 2.1 Feature Engineering
- [ ] Clean and normalize continuous variables (vitals, labs) using clinical plausibility bounds
- [ ] Encode categorical variables (race, insurance, language, chief complaint, medrecon classes)
- [ ] **Clean Pyxis medication data** (high cardinality ~1000 unique values due to charting quirks):
  - Use regex patterns to normalize medication name variants (e.g., `Acetaminophen 2mg4mg` → `Acetaminophen`)
  - Check rare instances against known drug name patterns before grouping
  - Map to therapeutic class groupings
- [ ] Handle missingness — strategy per variable:
  - Carry-forward for vitals between readings
  - Flag as `Not Ordered` (0) for labs/imaging not done during the visit
  - Explicit `missing` indicator features where clinically relevant

### 2.2 Time Series Construction
> Most complex preprocessing step.

- [ ] Set time step interval — informed by Stage 1 stay duration and action frequency distributions
- [ ] For each patient stay, build a matrix: `[timestep × features]`
- [ ] Encode action result flags per ordered test: status (0/1/2), `time_since_result`, `time_ordered_to_resulted`
- [ ] Align events to time bins: vitals, labs, medication dispenses, imaging orders
- [ ] Truncate/pad episodes to consistent length (or variable-length sequences)
- [ ] Encode actions taken at each timestep from Pyxis and radiology data
- [ ] Output: one record per patient per timestep, keyed by `subject_id` + `stay_id`

### 2.3 Train/Val/Test Split
- [ ] Split by `subject_id` to prevent leakage
- [ ] Stratify by cohort label
- [ ] Save splits to separate files — **test set never touched until final evaluation**

### 2.4 Final RL Formatting
- [ ] Format as RL transition tuples: `(state_t, action_t, reward_t, state_t+1, done)`
- [ ] Validate: no `subject_id` leakage across splits
- [ ] Validate: class distribution preserved across splits

---

## Stage 3: Modeling

- [ ] Implement environment wrapper (maps data → RL episodes)
- [ ] Implement action masking logic (except `observe` and `check_vitals`)
- [ ] Implement selected algorithm (per Stage 0.5 decision) with distributional output
- [ ] Train on training set
- [ ] Monitor training: loss curves, Q-value distributions, policy entropy
- [ ] Tune hyperparameters on validation set

---

## Stage 4: Evaluation

### Primary Metrics
- **ICU escalation accuracy**: Did the agent recommend ICU for patients who needed it? (Sensitivity on ED_DIRECT_ICU + ED_WARD_ICU)
- **Safe discharge accuracy**: Did the agent avoid recommending discharge for bounce-back/death cases? (ED_DISCHARGE_RETURN_ICU + ED_DISCHARGE_DIED_72H)
- **Decision speed**: Average timestep of final disposition recommendation vs clinician — faster with same quality is the goal

### Secondary Metrics
- Clinician agreement rate: % of episodes where agent final disposition matches actual, stratified by outcome
- Off-policy value estimate (OPE): expected reward under learned policy vs behavioral (clinician) policy

### Fairness Audit
- Evaluate policy differences across demographic groups (race, gender, insurance)
- Report whether protected attributes influence recommendations in ways that aren't clinically justified

### Adverse Outcome Evaluation
- Report results on ED_DISCHARGE_RETURN_ICU (224) and ED_DISCHARGE_DIED_72H (57) with confidence intervals
- Statistically weak signal — interpret with appropriate caution

---

## Stage 5: Insights & Reporting

- [ ] Summarize learned policy patterns vs clinician behavior
- [ ] Identify high-risk feature combinations the agent responds to
- [ ] Quantify decision time improvement and where time is lost (e.g., lab turnaround)
- [ ] Report uncertainty quantification: how confident is the agent in its recommendations?
- [ ] Discuss limitations: offline RL constraints, confounding, data coverage gaps
- [ ] Document in final report

---

## Open Decisions Log
*(to be resolved and logged in `decisions.md` during Stage 0)*

| Decision | Status | Notes |
|---|---|---|
| Action space granularity | Open | Requires Stage 1 Pyxis/radiology cardinality analysis |
| RL algorithm selection | Open | See Stage 0.5 options; must include distributional output |
| Time step interval | Open | Set after Stage 1 stay-duration distribution analysis |
| Reward function values | Open | Define exact numerical values |
| Episode max length | Open | Set after Stage 1 stay-duration analysis |
| Termination states | Open | Are admit_ICU / discharge_home / deceased terminal? Depends on MIMIC in-ED death tracking |
| Patient state definition | Open | Finalize all features and encodings after Stage 1 |
| time_ordered_to_resulted feature | Open | Add as continuous feature? Could reveal non-provider bottlenecks (e.g., lab turnaround) |
| Oversample adverse cohorts? | **Decided: No** | Natural class imbalance; targeted evaluation instead |
| Bias handling | **Decided** | Include protected attributes; audit for fairness in Stage 4 |
