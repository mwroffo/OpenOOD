import streamlit as st
import pandas as pd
import os
import glob

st.set_page_config(layout="wide")
st.title("🧠 OpenOOD Method Benchmark Dashboard")
st.markdown("Compare CUB-150 fine-grained OSR methods across key OOD metrics.")

# Load all ood.csv files
@st.cache_data
def load_method_summaries(base_path=".", file_name="ood.csv"):
    summaries = []
    # Get all the results folders
    # TODO: We can modify this to pick up results from other datasets/architectures
    folders = glob.glob(f"{base_path}/cub150_seed1_resnet18*")
    for folder in folders:
        # Extract the method from the folder path
        method = os.path.basename(folder).replace("cub150_seed1_resnet18_224x224_test_ood_osr_", "").replace("_default", "")
        csv_path = os.path.join(folder,file_name)
        if os.path.exists(csv_path):
            # Read the data
            df = pd.read_csv(csv_path)
            osr_rows = df[df["dataset"] == "osr"].copy()
            if not osr_rows.empty:
                # Average all repeated runs so we only get 1 value for each metric
                mean_row = osr_rows.drop(columns=["dataset"]).mean(numeric_only=True)
                mean_row["Method"] = method.upper()
                summaries.append(pd.DataFrame([mean_row])) 
    return pd.concat(summaries, ignore_index=True)

df_summary = load_method_summaries(base_path="../results")

# Show full table
st.subheader("📊 Method Summary Table")
st.dataframe(df_summary[["Method", "FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC"]].set_index("Method"))

# Chart selectors
st.subheader("📈 Visualize a Metric")
metric = st.selectbox("Select a metric to plot:", ["AUROC", "FPR@95", "AUPR_IN", "AUPR_OUT", "ACC"])
st.bar_chart(df_summary.set_index("Method")[metric])

# Optional: highlight best method
best_idx = df_summary[metric].idxmax() if metric != "FPR@95" else df_summary[metric].idxmin()
best_method = df_summary.iloc[best_idx]["Method"]
st.success(f"🏆 Best method for **{metric}**: `{best_method}`")
