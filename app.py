import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# PREMIUM DARK DASHBOARD THEME
# ==========================================

st.markdown("""
/* =========================================
   RESTORED BLACK GRADIENT + GLOWING PARTICLES
   ========================================= */

/* Aesthetic black gradient */
.stApp {
    background:
        radial-gradient(
            ellipse at 12% 12%,
            rgba(100, 25, 40, 0.22) 0%,
            transparent 42%
        ),
        radial-gradient(
            ellipse at 88% 20%,
            rgba(25, 45, 90, 0.22) 0%,
            transparent 44%
        ),
        radial-gradient(
            ellipse at 55% 100%,
            rgba(90, 48, 20, 0.10) 0%,
            transparent 42%
        ),
        linear-gradient(
            135deg,
            #050505 0%,
            #101014 50%,
            #070709 100%
        ) !important;

    background-attachment: fixed !important;
}

/* Prevent Streamlit containers from hiding the gradient */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
section.main {
    background: transparent !important;
}

/* Animated glowing dots */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;

    background-image:
        radial-gradient(circle, rgba(255, 65, 75, 0.65) 1px, transparent 2px),
        radial-gradient(circle, rgba(65, 135, 255, 0.60) 1px, transparent 2px),
        radial-gradient(circle, rgba(255, 150, 65, 0.50) 1px, transparent 2px),
        radial-gradient(circle, rgba(255, 255, 255, 0.35) 1px, transparent 2px);

    background-size:
        130px 130px,
        175px 175px,
        215px 215px,
        255px 255px;

    background-position:
        0 0,
        45px 60px,
        100px 25px,
        30px 110px;

    opacity: 0.65;
    animation: glowingParticles 45s linear infinite;
}

/* Keep interactive app content above the particles */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stHeader"] {
    position: relative;
    z-index: 1;
}

/* Slow, subtle movement */
@keyframes glowingParticles {
    0% {
        background-position:
            0 0, 45px 60px, 100px 25px, 30px 110px;
    }

    50% {
        background-position:
            65px 90px, 125px 145px, 205px 105px, 155px 225px;
    }

    100% {
        background-position:
            130px 180px, 220px 230px, 310px 185px, 280px 340px;
    }
}

/* Disable movement for users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
    .stApp::before {
        animation: none;
    }
}
<style>

/* ===== SUBTLE BLACK GRADIENT BACKGROUND ===== */

.stApp {
    background:
        radial-gradient(
            ellipse at 10% 0%,
            rgba(80, 35, 45, 0.20),
            transparent 42%
        ),
        radial-gradient(
            ellipse at 90% 15%,
            rgba(30, 50, 85, 0.18),
            transparent 45%
        ),
        linear-gradient(
            135deg,
            #080808 0%,
            #101014 45%,
            #08090D 100%
        );
    background-attachment: fixed;
    color: #FFFFFF;
}

/* ===== CLEAN HEADER ===== */

[data-testid="stHeader"] {
    background: rgba(8, 8, 8, 0.55);
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

/* ===== HEADINGS ===== */

h1, h2, h3 {
    color: #F5F5F5 !important;
    letter-spacing: 0.3px;
}

h1 {
    text-shadow: 0 0 24px rgba(255, 60, 70, 0.10);
}

/* ===== BODY TEXT ===== */

p, .stMarkdown, label,
[data-testid="stCaptionContainer"] {
    color: #D2D2D8;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111114 0%,
        #09090B 100%
    );
    border-right: 1px solid #292930;
}

/* ===== INPUTS: CHARCOAL GREY WITH BLACK TEXT ===== */

/* Text and number input fields */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #D8D8DE !important;
    color: #111111 !important;
    border: 1px solid #A5A5AE !important;
    border-radius: 8px !important;
    caret-color: #111111 !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #45454C !important;
    opacity: 1;
}

/* Select boxes */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background-color: #D8D8DE !important;
    color: #111111 !important;
    border-color: #A5A5AE !important;
    border-radius: 8px !important;
}

/* Selected text inside dropdowns */
.stSelectbox [data-baseweb="select"] *,
.stMultiSelect [data-baseweb="select"] * {
    color: #111111 !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #E2E2E7 !important;
}

[data-baseweb="menu"] *,
[role="option"] {
    color: #111111 !important;
}

/* ===== FILE UPLOAD AREA ===== */

[data-testid="stFileUploader"] section {
    background: rgba(205, 205, 214, 0.96) !important;
    border: 1px dashed #777780 !important;
    border-radius: 12px !important;
}

/* Text inside the upload area */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section * {
    color: #17171B !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background: #303038 !important;
    color: #FFFFFF !important;
    border: 1px solid #50505A !important;
    border-radius: 7px !important;
}

/* ===== PREDICTION BUTTONS ===== */

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        100deg,
        #D92F40,
        #F05A35
    ) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(
        100deg,
        #286CDE,
        #408BFF
    ) !important;
    box-shadow: 0 0 18px rgba(55, 125, 255, 0.20);
    transform: translateY(-1px);
}

/* ===== METRIC CARDS ===== */

[data-testid="stMetric"] {
    background: rgba(24, 24, 29, 0.88);
    border: 1px solid #303039;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #B9BAC4 !important;
}

[data-testid="stMetricValue"] {
    color: #FF9345 !important;
}

/* ===== TABS ===== */

.stTabs [data-baseweb="tab"] {
    color: #C9CAD2;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom-color: #F04452 !important;
}

/* ===== TABLES ===== */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #303039;
    border-radius: 9px;
}

/* ===== DIVIDERS ===== */

hr {
    border-color: #303039 !important;
}

/* ===== ACCESSIBILITY ===== */

@media (prefers-reduced-motion: reduce) {
    .stButton > button,
    .stDownloadButton > button {
        transition: none;
    }
}

</style>
""", unsafe_allow_html=True)

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Cu(II) Adsorption Predictor", page_icon="🌿", layout="wide")

st.title("🌿 Cu(II) Adsorption Capacity — Random Forest Demo")
st.caption("PBL prototype | GO, rGO and related graphene-based adsorbents")

st.warning(
    "**Research prototype only — not a validated scientific predictor.** "
    "The current dataset is small, highly imbalanced by source paper, and some source values "
    "still need verification. Treat every output as an exploratory demo estimate, not an "
    "experimental result or an engineering recommendation."
)

DATA_PATH = Path(__file__).parent / "cu_ii_training_data.csv"
NUM_FEATURES = ["pH", "Initial_Cu (mg/l)", "Contact_Time (min)", "Dose (g/L)"]
CAT_FEATURES = ["Adsorbent_type", "Synthesis_method"]
TARGET = "qe(mg/g)"
GROUP = "Paper_ID"

@st.cache_data
def load_data():
    frame = pd.read_csv(DATA_PATH)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    for col in NUM_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=[TARGET, GROUP]).copy()
    return frame

@st.cache_resource
def fit_model():
    frame = load_data()
    X = frame[NUM_FEATURES + CAT_FEATURES]
    y = frame[TARGET].astype(float)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUM_FEATURES),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), CAT_FEATURES)
        ],
        remainder="drop"
    )
    # Shallow RF was the lowest-MAE candidate in the exploratory tuning exercise.
    model = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=300, max_depth=3, min_samples_leaf=2,
            max_features=1.0, random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model, frame

try:
    model, frame = fit_model()
except Exception as exc:
    st.error(f"Could not load/train the demo model: {exc}")
    st.stop()

with st.sidebar:
    st.header("About this model")
    st.write("**Target:** adsorption capacity, qe (mg/g)")
    st.write("**Inputs used:** pH, initial Cu(II), contact time, adsorbent dose, adsorbent type, and synthesis method.")
    st.write("Temperature, BET surface area, and O/C ratio are not model inputs in this prototype.")
    st.write(f"**Training rows:** {len(frame)}")
    st.write(f"**Source papers/labels:** {frame[GROUP].nunique()}")
    st.write("**Algorithm:** RandomForestRegressor (shallow demo configuration)")
    st.divider()
    st.caption("Model is fitted locally when this app starts. No external service is called.")

st.subheader("1. Enter experimental conditions")
adsorbents = sorted(frame["Adsorbent_type"].dropna().astype(str).unique().tolist())
methods = sorted(frame["Synthesis_method"].dropna().astype(str).unique().tolist())
if not adsorbents:
    adsorbents = ["GO", "rGO", "rGO-TiO2"]
if not methods:
    methods = ["Not specified"]

col1, col2, col3 = st.columns(3)
with col1:
    adsorbent = st.selectbox("Adsorbent type", adsorbents)
    ph = st.number_input("Solution pH", min_value=0.0, max_value=14.0, value=4.0, step=0.1)
with col2:
    initial_cu = st.number_input("Initial Cu(II) concentration (mg/L)", min_value=0.01, max_value=100000.0, value=50.0, step=1.0)
    contact_time = st.number_input("Contact time (min)", min_value=0.0, max_value=100000.0, value=60.0, step=5.0)
with col3:
    dose = st.number_input("Adsorbent dose (g/L)", min_value=0.001, max_value=10000.0, value=1.0, step=0.1)
    synthesis = st.selectbox("Synthesis method", methods)

st.caption("Choose values and material categories that resemble the source dataset. A category not represented in training may produce unreliable estimates.")

if st.button("🌱 Predict demo qe", type="primary", use_container_width=True):
    input_row = pd.DataFrame([{
        "pH": ph,
        "Initial_Cu (mg/l)": initial_cu,
        "Contact_Time (min)": contact_time,
        "Dose (g/L)": dose,
        "Adsorbent_type": adsorbent,
        "Synthesis_method": synthesis
    }])
    prediction = float(model.predict(input_row)[0])
    st.subheader("2. Demo output")
    st.metric("Predicted adsorption capacity, qe", f"{prediction:.2f} mg/g")
    st.info(
        "This is a model output from a limited literature-derived dataset. "
        "It is not a measured value, and the current cross-paper validation has not shown reliable generalisation."
    )
    # Simple range/context check against observed training inputs
    ranges = {}
    for c, val in [("pH", ph), ("Initial_Cu (mg/l)", initial_cu),
                   ("Contact_Time (min)", contact_time), ("Dose (g/L)", dose)]:
        lo, hi = frame[c].min(), frame[c].max()
        if pd.notna(lo) and pd.notna(hi):
            ranges[c] = (float(lo), float(hi), float(val))
    out_of_range = [f"{c}: entered {v:g}, training range {lo:g}–{hi:g}"
                    for c, (lo, hi, v) in ranges.items() if v < lo or v > hi]
    if out_of_range:
        st.warning("One or more inputs are outside the observed training range:\n\n" + "\n\n".join(out_of_range))
    if adsorbent not in set(frame["Adsorbent_type"].dropna().astype(str)):
        st.warning("This adsorbent type was not represented in the training data; the estimate is especially unreliable.")

with st.expander("Model validation summary (important)"):
    st.write(
        "The previous leave-one-paper-out validation on the current candidate dataset produced "
        "MAE around 61 mg/g for the shallow Random Forest and R² around −0.01. "
        "These results indicate poor predictive performance on held-out papers."
    )
    st.write(
        "Do not use the demo for treatment design, safety decisions, scale-up, or replacing laboratory measurements. "
        "The next research step is source verification and adding comparable independent studies."
    )

with st.expander("Preview training data"):
    st.dataframe(frame, use_container_width=True, hide_index=True)

st.divider()
st.caption("Built for a biotechnology PBL demonstration. Document the dataset provenance and validation limitations in your report.")

st.divider()
st.header("📂 Dataset Management")
st.write(
    "Upload a CSV file to inspect its columns, check whether it contains "
    "the model's required features, and download a copy."
)

uploaded_file = st.file_uploader(
    "Upload your research dataset (CSV)",
    type=["csv"],
    key="dataset_management_upload"
)

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded dataset")
        st.write(f"Rows: {len(uploaded_data)}")
        st.write(f"Columns: {len(uploaded_data.columns)}")

        required_columns = NUM_FEATURES + CAT_FEATURES + [TARGET]
        missing_columns = [
            col for col in required_columns
            if col not in uploaded_data.columns
        ]

        if missing_columns:
            st.error(
                "Some required model columns are missing: "
                + ", ".join(missing_columns)
            )
            st.caption(
                "Required columns: " + ", ".join(required_columns)
            )
        else:
            st.success("All required model columns are present.")

            numeric_columns = NUM_FEATURES + [TARGET]
            numeric_data = uploaded_data[numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )

            missing_values = numeric_data.isna().sum()
            if missing_values.any():
                st.warning(
                    "Some required numeric values are missing or "
                    "cannot be interpreted as numbers."
                )
                st.dataframe(
                    missing_values.rename("Missing/invalid values"),
                    use_container_width=True
                )

        st.subheader("Dataset preview")
        st.dataframe(uploaded_data.head(100), use_container_width=True)

        st.download_button(
            "⬇️ Download uploaded dataset as CSV",
            data=uploaded_data.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_cuii_dataset.csv",
            mime="text/csv"
        )
        st.divider()
        st.subheader("🌳 Train Random Forest on Uploaded Dataset")

        if st.button("Train model using uploaded dataset", key="train_uploaded_model"):
            try:
                training_data = uploaded_data.copy()

                # Convert required numeric columns to numbers.
                numeric_columns = NUM_FEATURES + [TARGET]
                for col in numeric_columns:
                    training_data[col] = pd.to_numeric(
                        training_data[col], errors="coerce"
                    )

                # Keep only rows with a target and all required inputs.
                training_data = training_data.dropna(
                    subset=NUM_FEATURES + CAT_FEATURES + [TARGET]
                )

                if len(training_data) < 10:
                    st.error(
                        "At least 10 complete rows are required for this demo. "
                        "More data is preferable."
                    )
                else:
                    X = training_data[NUM_FEATURES + CAT_FEATURES]
                    y = training_data[TARGET].astype(float)

                    preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline([
                                    ("imputer", SimpleImputer(strategy="median"))
                                ]),
                                NUM_FEATURES
                            ),
                            (
                                "categorical",
                                Pipeline([
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                                ]),
                                CAT_FEATURES
                            )
                        ],
                        remainder="drop"
                    )

                    uploaded_model = Pipeline([
                        ("preprocess", preprocessor),
                        ("rf", RandomForestRegressor(
                            n_estimators=300,
                            max_depth=3,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        ))
                    ])

                    uploaded_model.fit(X, y)

                    st.success(
                        f"Model trained successfully on {len(training_data)} complete rows."
                    )

                    st.warning(
                        "These are training-set metrics, not independent validation. "
                        "They can look better than real-world performance. "
                        "Use paper-grouped validation before drawing research conclusions."
                    )

                    training_predictions = uploaded_model.predict(X)

                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric(
                        "Training MAE",
                        f"{mean_absolute_error(y, training_predictions):.2f} mg/g"
                    )
                    metric2.metric(
                        "Training RMSE",
                        f"{np.sqrt(mean_squared_error(y, training_predictions)):.2f} mg/g"
                    )
                    metric3.metric(
                        "Training R²",
                        f"{r2_score(y, training_predictions):.3f}"
                    )

                    st.subheader("Feature importance")
                    transformed_names = (
                        uploaded_model.named_steps["preprocess"]
                        .get_feature_names_out()
                    )
                    importances = uploaded_model.named_steps["rf"].feature_importances_

                    importance_frame = pd.DataFrame({
                        "Feature": transformed_names,
                        "Importance": importances
                    }).sort_values("Importance", ascending=False)

                    st.dataframe(
                        importance_frame,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download trained model predictions",
                        data=pd.DataFrame({
                            "Observed_qe_mg_g": y.to_numpy(),
                            "Training_predicted_qe_mg_g": training_predictions
                        }).to_csv(index=False).encode("utf-8"),
                        file_name="cuii_training_predictions.csv",
                        mime="text/csv",
                        key="download_training_predictions"
                    )

            except Exception as exc:
                st.error(f"Model training failed: {exc}")

    except Exception as exc:
        st.error(f"Could not read this CSV file: {exc}")
st.divider()
st.header("🧪 Leave-One-Paper-Out Validation")

st.write(
    "This evaluation holds out one research paper at a time, trains the "
    "model on the remaining papers, and evaluates predictions on the "
    "excluded paper. This tests cross-paper generalisation more directly "
    "than training-set metrics."
)

if st.button("Run Leave-One-Paper-Out Validation"):
    try:
        validation_data = frame.copy()

        validation_data[TARGET] = pd.to_numeric(
            validation_data[TARGET], errors="coerce"
        )

        validation_data = validation_data.dropna(
    subset=[TARGET, GROUP]
)

        number_of_papers = validation_data[GROUP].nunique()

        if number_of_papers < 3:
            st.error(
                "At least three distinct research papers are needed "
                "for this exploratory validation."
            )
        else:
            logo = LeaveOneGroupOut()

            X_val = validation_data[NUM_FEATURES + CAT_FEATURES]
            y_val = validation_data[TARGET].astype(float)
            groups = validation_data[GROUP]

            observed_all = []
            predicted_all = []
            paper_results = []

            for train_idx, test_idx in logo.split(
                X_val, y_val, groups=groups
            ):
                X_train, X_test = (
                    X_val.iloc[train_idx],
                    X_val.iloc[test_idx]
                )
                y_train, y_test = (
                    y_val.iloc[train_idx],
                    y_val.iloc[test_idx]
                )

                fold_preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline([
                                ("imputer", SimpleImputer(strategy="median"))
                            ]),
                            NUM_FEATURES
                        ),
                        (
                            "categorical",
                            Pipeline([
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore"))
                            ]),
                            CAT_FEATURES
                        )
                    ],
                    remainder="drop"
                )

                fold_model = Pipeline([
                    ("preprocess", fold_preprocessor),
                    ("rf", RandomForestRegressor(
                        n_estimators=300,
                        max_depth=3,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])

                fold_model.fit(X_train, y_train)
                fold_predictions = fold_model.predict(X_test)

                paper_name = str(groups.iloc[test_idx[0]])

                observed_all.extend(y_test.tolist())
                predicted_all.extend(fold_predictions.tolist())

                paper_results.append({
                    "Held-out paper": paper_name,
                    "Test rows": len(test_idx),
                    "MAE (mg/g)": mean_absolute_error(
                        y_test, fold_predictions
                    ),
                    "RMSE (mg/g)": np.sqrt(
                        mean_squared_error(y_test, fold_predictions)
                    )
                })

           
            overall_mae = mean_absolute_error(
                observed_all, predicted_all
            )

            overall_rmse = np.sqrt(
                mean_squared_error(observed_all, predicted_all)
            )

            overall_r2 = r2_score(
                observed_all, predicted_all
            )

            macro_mae = np.mean([
                result["MAE (mg/g)"]
                for result in paper_results
            ])

            st.success(
                f"Validation completed across {number_of_papers} papers."
            )

            col1, col2 = st.columns(2)
            col1.metric("LOPO MAE", f"{overall_mae:.2f} mg/g")
            col2.metric("LOPO RMSE", f"{overall_rmse:.2f} mg/g")

            col3, col4 = st.columns(2)
            col3.metric("LOPO R²", f"{overall_r2:.3f}")
            col4.metric("Macro-average paper MAE", f"{macro_mae:.2f} mg/g")
            st.warning(
                "These results are exploratory. Performance may vary greatly "
                "between papers, especially when the dataset is small or "
                "the papers report very different adsorption capacities. "
                "Do not interpret these metrics as proof of general "
                "predictive accuracy."
            )

            st.subheader("Performance for each held-out paper")
            paper_results_df = pd.DataFrame(paper_results)
            st.dataframe(
                paper_results_df,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "Download LOPO validation results",
                data=paper_results_df.to_csv(index=False).encode("utf-8"),
                file_name="cuii_lopo_validation_results.csv",
                mime="text/csv",
                key="download_lopo_results"
            )

    except Exception as exc:
        st.error(f"LOPO validation failed: {exc}")
