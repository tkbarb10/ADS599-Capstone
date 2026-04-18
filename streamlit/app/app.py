"""
Patient Portal — Emergency Department Clinical Decision Support
Iteration 2: Transfer Decision (Tab 2) + What Matters placeholder (Tab 3)
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Portal | ED Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent  # streamlit/ root

# ── Data loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ps = pd.read_parquet(DATA_DIR / 'patient_stats.parquet')
    pp = pd.read_parquet(DATA_DIR / 'patient_probability.parquet')
    sf = pd.read_parquet(DATA_DIR / 'step_features.parquet')
    return ps, pp, sf

patient_stats, patient_probability, step_features = load_data()

# ── Session state ──────────────────────────────────────────────────────────
if 'subject_id'  not in st.session_state:
    st.session_state.subject_id  = None
if 'ed_stay_id'  not in st.session_state:
    st.session_state.ed_stay_id  = None

# ── Stay-level summary — one row per ED stay ──────────────────────────────
stay_outcomes = (
    patient_probability[patient_probability['terminal_code'] == 1]
    .drop_duplicates('ed_stay_id')[['ed_stay_id', 'terminal_event']]
    .assign(Outcome=lambda d: d['terminal_event'].map({'discharge': 'Discharge', 'transfer_icu': 'ICU Transfer'}))
)
patient_stats_with_outcome = patient_stats.merge(stay_outcomes[['ed_stay_id', 'Outcome']], on='ed_stay_id', how='left')

subject_summary = (
    patient_stats_with_outcome
    .sort_values('ed_intime', ascending=False)
    [['ed_stay_id', 'subject_id', 'anchor_age', 'chiefcomplaint', 'arrival_transport', 'Outcome']]
    .rename(columns={
        'ed_stay_id':        'Stay ID',
        'subject_id':        'Patient ID',
        'anchor_age':        'Age',
        'chiefcomplaint':    'Chief Complaint',
        'arrival_transport': 'Arrival',
    })
    .reset_index(drop=True)
)

# ── Helper functions ───────────────────────────────────────────────────────
VITAL_COLORS = {
    'low':    ('#d6eaf8', '#1a5276'),
    'normal': ('#d5f5e3', '#1e8449'),
    'high':   ('#fadbd8', '#922b21'),
    'null':   ('#f5f5f5', '#aaa'),
}

def _vital_color(value, low, high):
    if value is None or pd.isna(value):
        return VITAL_COLORS['null']
    if value < low:
        return VITAL_COLORS['low']
    if value > high:
        return VITAL_COLORS['high']
    return VITAL_COLORS['normal']

def vital_card(col, label, value, unit, low, high, fmt='.0f'):
    bg, fg = _vital_color(value, low, high)
    display = f'{value:{fmt}}' if (value is not None and pd.notna(value)) else '—'
    with col:
        st.markdown(f"""
        <div style="background:{bg};border-radius:10px;padding:16px 10px;
                    text-align:center;margin-bottom:4px;">
            <div style="font-size:.72rem;color:#555;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;
                        margin-bottom:6px;">{label}</div>
            <div style="font-size:1.75rem;font-weight:700;color:{fg};
                        line-height:1.1;">{display}</div>
            <div style="font-size:.7rem;color:#888;margin-top:4px;">{unit}</div>
        </div>""", unsafe_allow_html=True)

def bp_card(col, sbp, dbp):
    both_ok = pd.notna(sbp) and pd.notna(dbp)
    if not both_ok:
        bg, fg = VITAL_COLORS['null']
        display = '— / —'
    elif sbp < 90 or dbp < 60:
        bg, fg = VITAL_COLORS['low']
        display = f'{sbp:.0f} / {dbp:.0f}'
    elif sbp > 130 or dbp > 80:
        bg, fg = VITAL_COLORS['high']
        display = f'{sbp:.0f} / {dbp:.0f}'
    else:
        bg, fg = VITAL_COLORS['normal']
        display = f'{sbp:.0f} / {dbp:.0f}'
    with col:
        st.markdown(f"""
        <div style="background:{bg};border-radius:10px;padding:16px 10px;
                    text-align:center;margin-bottom:4px;">
            <div style="font-size:.72rem;color:#555;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;
                        margin-bottom:6px;">Blood Pressure</div>
            <div style="font-size:1.75rem;font-weight:700;color:{fg};
                        line-height:1.1;">{display}</div>
            <div style="font-size:.7rem;color:#888;margin-top:4px;">mmHg (sys / dia)</div>
        </div>""", unsafe_allow_html=True)

ACUITY_STYLE = {
    1: ('#c0392b', '#fff', 'Immediate'),
    2: ('#e74c3c', '#fff', 'Emergent'),
    3: ('#e67e22', '#fff', 'Urgent'),
    4: ('#f39c12', '#333', 'Less Urgent'),
    5: ('#2ecc71', '#fff', 'Non-Urgent'),
}

def pain_bg(val):
    if pd.isna(val): return '#f5f5f5', '#aaa'
    if val <= 3:     return '#fff9c4', '#7d6608'
    if val <= 6:     return '#fde8d8', '#935116'
    return '#fadbd8', '#922b21'

def safe_str(val, fallback='—'):
    return str(val).title() if pd.notna(val) else fallback

def safe_num(val, fmt='.0f', fallback='—'):
    return f'{val:{fmt}}' if pd.notna(val) else fallback


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Patient Portal")
    st.caption("Emergency Department · Clinical Decision Support")
    st.divider()
    st.markdown("**Select a Patient**")
    st.caption(f"{len(subject_summary):,} patients — filter then click a row to select.")

    # Filter widgets
    age_min, age_max = int(subject_summary['Age'].min()), int(subject_summary['Age'].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max), key='age_filter')
    cc_filter = st.text_input("Chief complaint contains", key='cc_filter', placeholder="e.g. chest pain")
    outcome_filter = st.radio("Outcome", ["All", "Discharge", "ICU Transfer"], horizontal=True, key='outcome_filter')

    filtered = subject_summary[
        subject_summary['Age'].between(age_range[0], age_range[1])
    ]
    if cc_filter:
        filtered = filtered[
            filtered['Chief Complaint'].str.contains(cc_filter, case=False, na=False)
        ]
    if outcome_filter != "All":
        filtered = filtered[filtered['Outcome'] == outcome_filter]

    sel = st.dataframe(
        filtered.reset_index(drop=True),
        width='stretch',
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=300,
        key='patient_table',
    )

    sel_rows = (sel or {}).get("selection", {}).get("rows", [])
    if sel_rows and sel_rows[0] < len(filtered):
        chosen_row = filtered.iloc[sel_rows[0]]
        chosen_stay = int(chosen_row['Stay ID'])
        chosen_subject = int(chosen_row['Patient ID'])
        if chosen_stay != st.session_state.ed_stay_id:
            st.session_state.ed_stay_id = chosen_stay
            st.session_state.subject_id = chosen_subject


# ── Main content ───────────────────────────────────────────────────────────
if st.session_state.ed_stay_id is None:
    st.markdown("## Patient Portal")
    st.markdown("##### Emergency Department · Clinical Decision Support")
    st.divider()
    st.info("Select a patient from the sidebar to begin.", icon="👈")
    st.stop()

# Pull patient row
row = patient_stats[patient_stats['ed_stay_id'] == st.session_state.ed_stay_id].iloc[0]
ed_intime_fmt = pd.to_datetime(row['ed_intime']).strftime('%b %d,  %I:%M %p')

# Patient header
col_hdr, col_badge = st.columns([3, 1])
with col_hdr:
    st.markdown(f"## Patient {int(row['subject_id'])}")
    st.caption(f"ED Stay {int(row['ed_stay_id'])}  ·  Arrived {ed_intime_fmt}")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🧑‍⚕️  Patient Info",
    "📈  Transfer Decision",
    "🔍  What Matters",
])


# ── Tab 1: Patient Info ────────────────────────────────────────────────────
with tab1:
    col_demo, col_intake = st.columns(2, gap="large")

    # Demographics
    with col_demo:
        with st.container(border=True):
            st.markdown("#### Demographics")

            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Age", safe_num(row['anchor_age'], '.0f'))
            r1c2.metric("Gender", safe_str(row['gender']))

            r2c1, r2c2 = st.columns(2)
            height_str = f"{row['height']:.1f} in" if pd.notna(row['height']) else '—'
            weight_str = f"{row['weight']:.1f} lbs" if pd.notna(row['weight']) else '—'
            r2c1.metric("Height", height_str)
            r2c2.metric("Weight", weight_str)

            st.metric("Language", safe_str(row['language']))

    # Intake Information
    with col_intake:
        with st.container(border=True):
            st.markdown("#### Intake Information")

            cc = safe_str(row['chiefcomplaint'])
            st.metric("Chief Complaint", cc)

            i1, i2 = st.columns(2)
            i1.metric("Arrival Transport", safe_str(row['arrival_transport']))

            # Acuity — color-coded badge
            acuity_val = row['acuity']
            if pd.notna(acuity_val):
                av = int(acuity_val)
                abg, afg, alabel = ACUITY_STYLE.get(av, ('#aaa', '#fff', str(av)))
                i2.markdown(f"""
                <div style="padding:4px 0 8px;">
                    <div style="font-size:.85rem;color:#555;">ESI Acuity</div>
                    <div style="display:inline-block;background:{abg};color:{afg};
                                font-weight:700;font-size:1rem;border-radius:6px;
                                padding:4px 12px;margin-top:4px;">
                        {av} — {alabel}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                i2.metric("ESI Acuity", "—")

            # Pain — color-coded bar
            pain_val = row['current_pain']
            if pd.notna(pain_val):
                pbg, pfg = pain_bg(pain_val)
                st.markdown(f"""
                <div style="background:{pbg};border-radius:8px;
                            padding:10px 14px;margin-top:8px;">
                    <div style="font-size:.72rem;color:#555;font-weight:600;
                                text-transform:uppercase;letter-spacing:.06em;">Pain Level</div>
                    <div style="font-size:1.5rem;font-weight:700;color:{pfg};">
                        {int(pain_val)}
                        <span style="font-size:.9rem;font-weight:400;color:#888;"> / 10</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.metric("Pain Level", "—")

    st.markdown("#### Triage Vital Signs")

    v1, v2, v3, v4, v5, v6 = st.columns(6, gap="small")

    vital_card(v1, "Temperature",  row['current_temperature'], "°F",   97,  99,  '.1f')
    vital_card(v2, "Heart Rate",   row['current_heartrate'],   "bpm",  60,  100, '.0f')
    vital_card(v3, "Resp Rate",    row['current_resprate'],    "/min", 12,  20,  '.0f')
    vital_card(v4, "O2 Saturation",row['current_o2sat'],       "%",    95,  100, '.0f')
    bp_card(v5, row['current_sbp'], row['current_dbp'])

    # Mean Arterial Pressure (computed from sbp / dbp)
    sbp, dbp = row['current_sbp'], row['current_dbp']
    map_val = (dbp + (sbp - dbp) / 3) if (pd.notna(sbp) and pd.notna(dbp)) else None
    vital_card(v6, "MAP", map_val, "mmHg", 70, 100, '.0f')

    st.caption("🟢 Normal range   🔵 Below normal   🔴 Above normal")


# ── Tab 2: Transfer Decision ───────────────────────────────────────────────
with tab2:
    stay_prob = patient_probability[
        patient_probability['ed_stay_id'] == st.session_state.ed_stay_id
    ].sort_values('step_idx').reset_index(drop=True)

    stay_sf = step_features[
        step_features['ed_stay_id'] == st.session_state.ed_stay_id
    ].sort_values('step_idx').reset_index(drop=True)

    if stay_prob.empty:
        st.info("No trajectory data available for this patient.", icon="📈")
    else:
        ed_intime = pd.to_datetime(row['ed_intime'])
        terminal_event = stay_prob['terminal_event'].iloc[0]
        terminal_label = "ICU Transfer" if terminal_event == 'transfer_icu' else "Discharge"
        terminal_code = 1 if terminal_event == 'transfer_icu' else 0
        n_steps = len(stay_prob)

        step_times = [
            pd.to_datetime(t).strftime('%b %d, %I:%M %p')
            for t in stay_sf['time']
        ]

        # ── View toggle ───────────────────────────────────────────────────
        view = st.radio(
            "View",
            ["Full Stay", "Single Step"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if view == "Full Stay":
            st.markdown("#### ICU Transfer Probability — Full Stay")

            fig = go.Figure()

            # 50% threshold reference line
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="#aaa",
                annotation_text="50% threshold",
                annotation_position="bottom right",
            )

            # Trajectory line
            fig.add_trace(go.Scatter(
                x=step_times,
                y=stay_prob['p_icu'],
                mode='lines+markers',
                line=dict(color='#2c7bb6', width=2.5),
                marker=dict(size=5, color='#2c7bb6'),
                hovertemplate='%{x}<br>P(ICU): %{y:.1%}<extra></extra>',
                name='P(ICU)',
            ))

            # Terminal event marker
            terminal_color = '#d73027' if terminal_code == 1 else '#1a9850'
            fig.add_trace(go.Scatter(
                x=[step_times[-1]],
                y=[stay_prob['p_icu'].iloc[-1]],
                mode='markers+text',
                marker=dict(size=14, color=terminal_color, symbol='diamond'),
                text=[terminal_label],
                textposition='top center',
                hovertemplate=f'%{{x}}<br>P(ICU): %{{y:.1%}}<br>{terminal_label}<extra></extra>',
                name=terminal_label,
            ))

            fig.update_layout(
                xaxis_title=None,
                yaxis_title="P(ICU Transfer)",
                yaxis=dict(tickformat='.0%', range=[0, 1.05]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=20, r=20, t=10, b=40),
                height=340,
                hovermode='x unified',
            )
            fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))

            st.plotly_chart(fig, width='stretch')

            # Outcome summary
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Total Steps", n_steps)
            oc2.metric("Outcome", terminal_label)
            oc3.metric("Final P(ICU)", f"{stay_prob['p_icu'].iloc[-1]:.2%}")

        else:
            # ── Single step view ──────────────────────────────────────────
            step_labels = (
                ["Arrival (Step 0)"] +
                [f"Step {i}  ·  {step_times[i]}" for i in range(1, n_steps - 1)] +
                [f"{terminal_label}  ·  {step_times[-1]}"]
            ) if n_steps > 1 else [f"Arrival  ·  {step_times[0]}"]

            step_sel = st.select_slider(
                "Select a time step",
                options=list(range(n_steps)),
                format_func=lambda i: step_labels[i],
                key='step_slider',
            )

            step_row = stay_prob.iloc[step_sel]
            sf_row   = stay_sf.iloc[[step_sel]]
            sf_prev  = stay_sf.iloc[[step_sel - 1]] if step_sel > 0 else None

            p_icu_val = float(step_row['p_icu'])
            in_ed    = bool(step_row['in_ed'])
            in_ward  = bool(step_row['in_ward'])

            # Location badge
            if in_ed and in_ward:
                loc_bg, loc_fg, loc_text = '#f39c12', '#fff', '⚠️ Boarding in ED — Ward Transfer Pending'
            elif in_ward:
                loc_bg, loc_fg, loc_text = '#2980b9', '#fff', '🏥 In Ward'
            else:
                loc_bg, loc_fg, loc_text = '#27ae60', '#fff', '🚑 In Emergency Department'

            st.markdown(f"""
            <div style="background:{loc_bg};color:{loc_fg};border-radius:8px;
                        padding:10px 18px;font-weight:600;font-size:1rem;
                        display:inline-block;margin-bottom:16px;">
                {loc_text}
            </div>""", unsafe_allow_html=True)

            gauge_col, metric_col = st.columns([2, 1], gap="large")

            with gauge_col:
                gauge_color = '#d73027' if p_icu_val >= 0.5 else '#1a9850'
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_icu_val * 100,
                    number=dict(suffix="%", valueformat=".2f",
                                font=dict(size=36, color=gauge_color)),
                    gauge=dict(
                        axis=dict(range=[0, 100], ticksuffix='%'),
                        bar=dict(color=gauge_color, thickness=0.3),
                        bgcolor='#f0f0f0',
                        steps=[
                            dict(range=[0, 50],  color='#d5f5e3'),
                            dict(range=[50, 100], color='#fadbd8'),
                        ],
                        threshold=dict(
                            line=dict(color='#555', width=3),
                            thickness=0.85,
                            value=50,
                        ),
                    ),
                    title=dict(text="P(ICU Transfer)", font=dict(size=15)),
                ))
                fig_gauge.update_layout(margin=dict(l=20, r=20, t=30, b=10), height=240)
                st.plotly_chart(fig_gauge, width='stretch')

            with metric_col:
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Time", step_times[step_sel])
                st.metric("Step", f"{step_sel} of {n_steps - 1}")
                if step_sel > 0:
                    prev_p = float(stay_prob.iloc[step_sel - 1]['p_icu'])
                    delta = p_icu_val - prev_p
                    delta_str = f"{delta:+.1%}"
                    st.metric("Change from prev", delta_str)

            # ── What changed this step ────────────────────────────────────
            if step_sel > 0 and not sf_row.empty and sf_prev is not None and not sf_prev.empty:
                st.markdown("---")
                st.markdown("**What changed at this step**")

                sf_cur  = sf_row.iloc[0]
                sf_prv  = sf_prev.iloc[0]
                changes = []

                # Vitals that shifted meaningfully
                VITAL_THRESHOLDS = {
                    'current_heartrate': ('Heart Rate', 5, 'bpm'),
                    'current_temperature': ('Temperature', 0.5, '°F'),
                    'current_o2sat': ('O2 Saturation', 2, '%'),
                    'current_sbp': ('Systolic BP', 10, 'mmHg'),
                    'current_dbp': ('Diastolic BP', 10, 'mmHg'),
                    'current_resprate': ('Resp Rate', 3, '/min'),
                    'current_pain': ('Pain', 1, '/10'),
                }
                for col, (label, thresh, unit) in VITAL_THRESHOLDS.items():
                    if col in sf_cur.index:
                        cur_v, prv_v = sf_cur[col], sf_prv[col]
                        if pd.notna(cur_v) and pd.notna(prv_v) and abs(cur_v - prv_v) >= thresh:
                            arrow = '↑' if cur_v > prv_v else '↓'
                            changes.append(f"{arrow} **{label}**: {prv_v:.1f} → {cur_v:.1f} {unit}")

                # Labs: result came in (Pending→Normal or Pending→Abnormal)
                LAB_GROUPS = {
                    'Chemistry-Blood':   'Chemistry (Blood)',
                    'Hematology-Blood':  'Hematology (Blood)',
                    'Blood Gas-Blood':   'Blood Gas',
                    'BLOOD CULTURE':     'Blood Culture',
                }
                for prefix, label in LAB_GROUPS.items():
                    prev_pend = sf_prv.get(f'{prefix}_Pending', 0)
                    cur_pend  = sf_cur.get(f'{prefix}_Pending', 0)
                    if prev_pend == 0 and cur_pend == 1:
                        changes.append(f"🔬 **{label}** ordered — result pending")
                    elif prev_pend == 1 and cur_pend == 0:
                        # result came in — find what it is
                        for status in ['Normal', 'Abnormal', 'Negative', 'Positive']:
                            k = f'{prefix}_{status}'
                            if k in sf_cur.index and sf_cur[k] == 1:
                                emoji = '✅' if status in ('Normal', 'Negative') else '⚠️'
                                changes.append(f"{emoji} **{label}** result: {status}")
                                break

                # ECG / Radiology status changes
                for grp, label in [('ecg_status', 'ECG'), ('rad_status', 'Radiology')]:
                    for level in ['Normal', 'Moderate', 'Acute']:
                        col = f'{grp}_{level}'
                        if col in sf_cur.index and sf_cur[col] == 1 and sf_prv.get(col, 0) == 0:
                            emoji = {'Normal': '✅', 'Moderate': '⚠️', 'Acute': '🚨'}.get(level, '')
                            changes.append(f"{emoji} **{label}** result received: {level}")

                # Medications newly administered
                MED_LABELS = {
                    'Antibiotic': 'Antibiotic',
                    'IV Fluid': 'IV Fluid',
                    'Analgesic - Opioid/NSAID': 'Opioid / NSAID',
                    'Analgesic - Acetaminophen': 'Acetaminophen',
                    'Antiemetic': 'Antiemetic',
                    'Anticoagulant': 'Anticoagulant',
                    'Corticosteroid': 'Corticosteroid',
                    'Benzodiazepine - Sedative/Anxiolytic': 'Benzodiazepine',
                    'Beta Blocker': 'Beta Blocker',
                    'Diuretic': 'Diuretic',
                    'Bronchodilator': 'Bronchodilator',
                }
                for col, label in MED_LABELS.items():
                    if col in sf_cur.index:
                        if sf_prv.get(col, 0) == 0 and sf_cur[col] > 0:
                            changes.append(f"💊 **{label}** administered")

                if changes:
                    for c in changes:
                        st.markdown(f"- {c}")
                else:
                    st.caption("No significant changes detected at this step.")


# ── Tab 3: What Matters ────────────────────────────────────────────────────
with tab3:
    stay_sf3 = step_features[
        step_features['ed_stay_id'] == st.session_state.ed_stay_id
    ].sort_values('step_idx').reset_index(drop=True)

    st.markdown("#### Feature Attribution")
    st.caption("🚧 Placeholder — Policy gradient attribution coming soon. "
               "Values below are illustrative only.")

    if stay_sf3.empty:
        st.info("No feature data available for this patient.", icon="🔍")
    else:
        # Step selector for waterfall
        n3 = len(stay_sf3)
        step_times3 = [
            pd.to_datetime(t).strftime('%b %d, %I:%M %p')
            for t in stay_sf3['time']
        ]
        step_sel3 = st.select_slider(
            "Select a step",
            options=list(range(n3)),
            format_func=lambda i: f"Step {i}  ·  {step_times3[i]}",
            key='step_slider_tab3',
        )

        sf3 = stay_sf3.iloc[step_sel3]

        # Placeholder feature contributions — feature value × fixed weight,
        # anchored so the running total lands near the actual p_icu.
        WATERFALL_FEATURES = [
            ('current_heartrate',         'Heart Rate',       0.004),
            ('current_o2sat',             'O2 Saturation',    -0.01),
            ('current_sbp',               'Systolic BP',      0.003),
            ('current_pain',              'Pain Level',       0.025),
            ('acuity_val',                'ESI Acuity',       -0.08),
            ('Chemistry-Blood_Abnormal',  'Chem Labs — Abnormal', 0.12),
            ('Hematology-Blood_Abnormal', 'Heme Labs — Abnormal', 0.10),
            ('ecg_status_Acute',          'ECG — Acute',      0.18),
            ('rad_status_Acute',          'Radiology — Acute',0.15),
            ('BLOOD CULTURE_Pending',     'Blood Culture Pending', 0.09),
            ('Antibiotic',                'Antibiotic Given', 0.07),
            ('IV Fluid',                  'IV Fluid Given',   0.04),
        ]

        # Pull acuity from patient row, not step features
        acuity_for_wf = float(row['acuity']) if pd.notna(row['acuity']) else 3.0

        labels, measures, values, texts = [], [], [], []
        running = 0.0
        baseline = max(0.05, float(sf3['p_icu']) * 0.15)

        for col, label, weight in WATERFALL_FEATURES:
            if col == 'acuity_val':
                raw = acuity_for_wf
            else:
                raw = float(sf3[col]) if col in sf3.index else 0.0
            contrib = raw * weight
            running += contrib
            labels.append(label)
            measures.append('relative')
            values.append(round(contrib, 4))
            sign = '+' if contrib >= 0 else ''
            texts.append(f"{sign}{contrib:.3f}")

        # Clamp total to actual p_icu
        total = float(sf3['p_icu'])
        labels  = ['Baseline'] + labels  + ['P(ICU)']
        measures = ['absolute'] + measures + ['total']
        values   = [round(baseline, 4)] + values + [round(total, 4)]
        texts    = [f"{baseline:.3f}"] + texts + [f"{total:.2%}"]

        colors = []
        for i, (m, v) in enumerate(zip(measures, values)):
            if m in ('absolute', 'total'):
                colors.append('#2c7bb6')
            elif v >= 0:
                colors.append('#d73027')
            else:
                colors.append('#1a9850')

        fig_wf = go.Figure(go.Waterfall(
            orientation='v',
            measure=measures,
            x=labels,
            y=values,
            text=texts,
            textposition='outside',
            connector=dict(line=dict(color='#ccc', width=1)),
            increasing=dict(marker=dict(color='#d73027')),
            decreasing=dict(marker=dict(color='#1a9850')),
            totals=dict(marker=dict(color='#2c7bb6')),
        ))
        fig_wf.update_layout(
            title=dict(
                text="Illustrative Feature Contributions to P(ICU Transfer)",
                font=dict(size=14),
            ),
            yaxis_title="Contribution to P(ICU)",
            yaxis=dict(tickformat='.3f'),
            xaxis_tickangle=-35,
            margin=dict(l=20, r=20, t=50, b=80),
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_wf, width='stretch')
        st.caption("Bar color: 🔴 increases risk of ICU transfer · 🟢 decreases risk · 🔵 baseline / total")