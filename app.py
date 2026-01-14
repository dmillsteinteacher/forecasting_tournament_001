import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
ADMIN_PASSWORD = "professor_ghost"
STATES = ["State 1", "State 2", "State 3"]
OBSERVATIONS = ["Low", "Medium", "High"]

# --- UTILITY FUNCTIONS ---
def brier_score(prob_vector, true_state_idx):
    actual = np.zeros(len(prob_vector))
    actual[true_state_idx] = 1.0
    return np.sum((prob_vector - actual)**2)

def get_stationary_dist(A):
    try:
        n = A.shape[0]
        # Solving the linear system for stationary distribution
        M = A.T - np.eye(n)
        M[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        stationary = np.linalg.solve(M, b)
        return np.clip(stationary, 0, 1)
    except:
        return np.array([0.333, 0.333, 0.334])

def update_forecast(prior, A, B, obs_idx):
    projection = np.dot(prior, A)
    new_forecast = projection * B[:, obs_idx]
    if np.sum(new_forecast) > 0:
        return new_forecast / np.sum(new_forecast)
    return np.array([0.333, 0.333, 0.334])

def normalize_matrix(df):
    """Normalizes rows to sum to 1.0 and returns the cleaned dataframe."""
    arr = df.to_numpy()
    # Avoid division by zero for empty rows
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0 
    arr = arr / row_sums
    return pd.DataFrame(arr, columns=df.columns, index=df.index)

# --- SESSION STATE ---
if 'game_step' not in st.session_state:
    st.session_state.game_step = -1
    st.session_state.true_path = []  
    st.session_state.history = []   
    st.session_state.current_belief = None 
    st.session_state.true_A = None
    st.session_state.true_B = None
    st.session_state.start_state_idx = 0
    # Persistence for Student Theories
    st.session_state.student_A = pd.DataFrame(np.eye(3), columns=STATES, index=STATES)
    st.session_state.student_B = pd.DataFrame(np.full((3,3), 0.33), columns=OBSERVATIONS, index=STATES)

# --- SIDEBAR: DIRECTOR ---
with st.sidebar:
    st.title("Tournament Director")
    mode = st.radio("View", ["Forecaster", "Director"])
    
    if mode == "Director":
        pwd = st.text_input("Director Password", type="password")
        if pwd == ADMIN_PASSWORD:
            if st.button("RESET TOURNAMENT"):
                for key in ["game_step", "true_path", "history", "current_belief"]:
                    if key in st.session_state: del st.session_state[key]
                st.session_state.game_step = -1
                st.rerun()
            
            st.subheader("1. Setup Hidden Reality")
            t_a_df = st.data_editor(pd.DataFrame(np.eye(3), columns=STATES, index=STATES), key="set_true_a")
            t_b_df = st.data_editor(pd.DataFrame(np.full((3,3), 0.33), columns=OBSERVATIONS, index=STATES), key="set_true_b")
            
            start_state_name = st.selectbox("Starting State", STATES)
            st.session_state.start_state_idx = STATES.index(start_state_name)
            steps = st.number_input("Steps", 5, 20, 10)
            
            if st.button("Initialize & Start"):
                st.session_state.true_A = t_a_df.to_numpy()
                st.session_state.true_B = t_b_df.to_numpy()
                path = []
                curr = st.session_state.start_state_idx 
                for _ in range(steps + 1):
                    o = np.random.choice([0,1,2], p=st.session_state.true_B[curr])
                    path.append((curr, o))
                    curr = np.random.choice([0,1,2], p=st.session_state.true_A[curr])
                st.session_state.true_path = path
                st.session_state.game_step = 0
                st.session_state.history = []
                st.rerun()

            if st.session_state.game_step >= 0:
                st.divider()
                if st.button("ADVANCE TO NEXT STEP"):
                    st.session_state.game_step += 1
                    st.rerun()

# --- MAIN: FORECASTER ---
st.title("🔮 Forecasting Tournament")

if st.session_state.game_step == -1:
    st.info("Waiting for Director to initialize the session...")
    st.stop()

step = st.session_state.game_step
true_state, true_obs = st.session_state.true_path[step]

st.header(f"Time Step {step}")
if step == 0:
    st.warning(f"🚨 INITIAL INTEL: System confirmed in **{STATES[st.session_state.start_state_idx]}**.")
else:
    st.info(f"🛰️ SENSOR DATA: **{OBSERVATIONS[true_obs]}**")

# Theory Tuning (Accordions)
with st.expander("🛠️ Update World Model (A & B Matrices)", expanded=(step == 0)):
    st.write("Refine your theories. These will persist to the next round.")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Transition (A)**")
        st.session_state.student_A = st.data_editor(st.session_state.student_A, key=f"a_edit_{step}")
    with c2:
        st.write("**Emissions (B)**")
        st.session_state.student_B = st.data_editor(st.session_state.student_B, key=f"b_edit_{step}")
    
    if st.button("Validate & Normalize Matrices"):
        st.session_state.student_A = normalize_matrix(st.session_state.student_A)
        st.session_state.student_B = normalize_matrix(st.session_state.student_B)
        st.toast("Matrices normalized to sum to 1.0!")
        st.rerun()

# Forecast Logic & Visuals
st.divider()
if step == 0:
    st.write("### Initial Belief Vector")
    s_vec_df = st.data_editor(pd.DataFrame([[0.33, 0.33, 0.34]], columns=STATES), key="init_v")
    s_vec = s_vec_df.to_numpy().flatten()
else:
    s_vec = update_forecast(st.session_state.current_belief, st.session_state.student_A.to_numpy(), st.session_state.student_B.to_numpy(), true_obs)
    st.write("### Predicted State Probability (The Ghost)")
    # Visual Refinement: Bar Chart
    chart_data = pd.DataFrame({"Probability": s_vec}, index=STATES)
    st.bar_chart(chart_data, color="#29b5e8")

# Commit Forecast
if st.button("Commit Forecast", type="primary"):
    # Final check on normalization
    sum_a = st.session_state.student_A.to_numpy().sum(axis=1)
    sum_b = st.session_state.student_B.to_numpy().sum(axis=1)
    
    if not (np.allclose(sum_a, 1.0, atol=1e-2) and np.allclose(sum_b, 1.0, atol=1e-2)):
        st.error("🚨 Row sums do not equal 1.0. Please use 'Validate & Normalize' above.")
    else:
        base_dist = get_stationary_dist(st.session_state.true_A)
        s_brier = brier_score(s_vec, true_state)
        b_brier = brier_score(base_dist, true_state)
        
        # Record keeping
        st.session_state.history.append({
            "Step": step,
            "Observation": OBSERVATIONS[true_obs] if step > 0 else "Initial Intel",
            "Outcome": STATES[true_state],
            "Forecast Distribution": [f"{x:.2f}" for x in s_vec],
            "Edge vs House": round(b_brier - s_brier, 4),
            "s_brier": s_brier, "b_brier": b_brier
        })
        st.session_state.current_belief = s_vec
        st.success(f"Forecast Committed! True State was: {STATES[true_state]}")

# History Record
if st.session_state.history:
    st.write("### Official Tournament Record")
    st.table(pd.DataFrame(st.session_state.history)[["Step", "Observation", "Outcome", "Forecast Distribution"]])

# Final Reveal
if step == len(st.session_state.true_path) - 1:
    if st.checkbox("🏆 FINAL TOURNAMENT ANALYSIS"):
        df = pd.DataFrame(st.session_state.history)
        avg_edge = df["Edge vs House"].mean()
        st.metric("Total Avg Edge Over House", f"{avg_edge:.4f}", help="Positive means you outperformed the stationary baseline.")
        st.write("### Detailed Step-by-Step Scoring")
        st.dataframe(df)
