import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
ADMIN_PASSWORD = "professor_ghost"
STATES = ["State 1", "State 2", "State 3"]
OBSERVATIONS = ["Low", "Medium", "High"]

# --- FORECASTING MATH ---
def brier_score(prob_vector, true_state_idx):
    """Measures accuracy: 0 is perfect, 2 is worst."""
    actual = np.zeros(len(prob_vector))
    actual[true_state_idx] = 1.0
    return np.sum((prob_vector - actual)**2)

def get_stationary_dist(A):
    """The 'House' baseline: the long-term average probability."""
    try:
        n = A.shape[0]
        M = A.T - np.eye(n)
        M[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        stationary = np.linalg.solve(M, b)
        return np.clip(stationary, 0, 1)
    except:
        return np.array([0.333, 0.333, 0.334])

def update_forecast(prior, A, B, obs_idx):
    """Propagates belief: Drift (A) then Evidence (B)."""
    # 1. Project forward (Transition)
    projection = np.dot(prior, A)
    # 2. Integrate evidence (Emission)
    new_forecast = projection * B[:, obs_idx]
    # Normalize
    if np.sum(new_forecast) > 0:
        return new_forecast / np.sum(new_forecast)
    return np.array([0.333, 0.333, 0.334])

# --- PERSISTENT STATE ---
if 'game_step' not in st.session_state:
    st.session_state.game_step = -1
    st.session_state.true_path = []  
    st.session_state.history = []   
    st.session_state.current_belief = None 
    st.session_state.true_A = None
    st.session_state.true_B = None

# --- SIDEBAR: TOURNAMENT DIRECTOR ---
with st.sidebar:
    st.title("Tournament Control")
    mode = st.radio("View", ["Forecaster", "Director"])
    
    if mode == "Director":
        pwd = st.text_input("Director Password", type="password")
        if pwd == ADMIN_PASSWORD:
            st.success("Director Access Active")
            
            st.subheader("Define Hidden Reality")
            t_a = st.data_editor(np.eye(3), key="setup_a", caption="True Physics (A)")
            t_b = st.data_editor(np.full((3,3), 0.33), key="setup_b", caption="True Sensor Reliability (B)")
            
            steps = st.number_input("Tournament Length", 5, 20, 10)
            
            if st.button("Initialize Tournament Path"):
                path = []
                curr = 0 # Step 0 always starts in State 1
                for _ in range(steps + 1):
                    o = np.random.choice([0,1,2], p=t_b[curr])
                    path.append((curr, o))
                    curr = np.random.choice([0,1,2], p=t_a[curr])
                
                st.session_state.true_path = path
                st.session_state.true_A = t_a
                st.session_state.true_B = t_b
                st.session_state.game_step = 0
                st.session_state.history = []
                st.rerun()

            if st.session_state.game_step >= 0:
                st.divider()
                if st.button("ADVANCE TO NEXT TIME STEP"):
                    st.session_state.game_step += 1
                    st.rerun()

# --- MAIN INTERFACE: FORECASTER DASHBOARD ---
st.title("🔮 Forecasting Tournament")

if st.session_state.game_step == -1:
    st.info("The Tournament Director has not started the session. Please wait.")
    st.stop()

step = st.session_state.game_step
true_state, true_obs = st.session_state.true_path[step]

st.subheader(f"Time Step: {step}")

# 1. Provide Observations
if step == 0:
    st.warning("🚨 INITIAL INTEL: The system is confirmed to be in **State 1**.")
else:
    st.info(f"🛰️ SENSOR DATA RECEIVED: **{OBSERVATIONS[true_obs]}**")

# 2. Model Tuning Section
st.divider()
st.write("### Refine Your World Model")
c1, c2 = st.columns(2)
with c1:
    st.write("Transition Theory (How states move)")
    s_a = st.data_editor(np.eye(3), key=f"s_a_{step}")
with c2:
    st.write("Evidence Theory (Sensor reliability)")
    s_b = st.data_editor(np.full((3,3), 0.33), key=f"s_b_{step}")

# 3. Forecast Generation
if step == 0:
    st.write("### Initial Belief Vector")
    s_vec_raw = st.data_editor(np.array([[0.33, 0.33, 0.34]]), key="s_vec_0")
    s_vec = s_vec_raw.flatten()
else:
    st.write("### Calculated Forecast (Posterior)")
    # Recursive forecasting based on previous belief
    res = update_forecast(st.session_state.current_belief, s_a, s_b, true_obs)
    s_vec = res
    st.write(pd.DataFrame([s_vec], columns=STATES))

# 4. Commit Forecast
if st.button("Submit Forecast"):
    # Score against the 'House' (Stationary Baseline)
    base_dist = get_stationary_dist(st.session_state.true_A)
    
    s_brier = brier_score(s_vec, true_state)
    b_brier = brier_score(base_dist, true_state)
    
    st.session_state.history.append({
        "Step": step,
        "Outcome": STATES[true_state],
        "Forecaster Score": s_brier,
        "Baseline Score": b_brier,
        "Relative Advantage": b_brier - s_brier # Higher is better: how much you beat the house
    })
    st.session_state.current_belief = s_vec
    st.success(f"Forecast Committed! Reality revealed: {STATES[true_state]}")

# 5. Tournament History
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.write("### Official Record")
    st.table(hist_df[["Step", "Outcome"]])

# 6. The Grand Reveal
if step == len(st.session_state.true_path) - 1:
    if st.checkbox("FINAL LEADERBOARD: Show My Performance"):
        st.header("🏆 Performance Review")
        final_df = pd.DataFrame(st.session_state.history)
        
        # We calculate the average of (Baseline - Student)
        # A positive number means the student added more value than the house.
        avg_advantage = final_df["Relative Advantage"].mean()
        
        c_a, c_b = st.columns(2)
        c_a.metric("Avg. Edge Over House", round(avg_advantage, 4))
        c_b.write("Performance Log")
        st.dataframe(final_df[["Step", "Outcome", "Forecaster Score", "Baseline Score", "Relative Advantage"]])