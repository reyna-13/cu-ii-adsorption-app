import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cu(II) Adsorption Predictor",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 8%, rgba(110,25,43,.20), transparent 42%),
            radial-gradient(ellipse at 88% 15%, rgba(28,48,95,.20), transparent 45%),
            radial-gradient(ellipse at 55% 100%, rgba(95,48,20,.09), transparent 45%),
            linear-gradient(135deg, #060607 0%, #101014 48%, #08080B 100%) !important;
        background-attachment: fixed !important;
        color: #F5F5F7;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    section.main {
        background: transparent !important;
    }

    [data-testid="stHeader"] {
        background: rgba(7,7,9,.65) !important;
    }

    /* Animated glowing particles */

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        background-image:
            radial-gradient(circle, rgba(255,65,78,.65) 1px, transparent 2px),
            radial-gradient(circle, rgba(65,135,255,.60) 1px, transparent 2px),
            radial-gradient(circle, rgba(255,150,65,.48) 1px, transparent 2px),
            radial-gradient(circle, rgba(255,255,255,.32) 1px, transparent 2px);
        background-size:
            135px 135px,
            180px 180px,
            225px 225px,
            265px 265px;
        background-position:
            0 0,
            45px 60px,
            100px 25px,
            30px 110px;
        opacity: .38;
        animation: particleDrift 45s linear infinite;
    }

    @keyframes particleDrift {
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
                135px 180px, 225px 240px, 325px 205px, 295px 370px;
        }
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111115, #09090C) !important;
        border-right: 1px solid #292932;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #D1D1D8 !important;
    }

    /* Text and headings */

    h1, h2, h3 {
        color: #F7F7FA !important;
        letter-spacing: .2px;
    }

    h1 {
        text-shadow: 0 0 24px rgba(255,55,70,.12);
    }

    p, li, .stMarkdown {
        color: #D1D1D8;
    }

    [data-testid="stCaptionContainer"] {
        color: #A7A7B2 !important;
    }

    /* Input fields */

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background-color: #D8D8DE !important;
        color: #111116 !important;
        border: 1px solid #A6A6B0 !important;
        border-radius: 8px !important;
        caret-color: #111116 !important;
    }

    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #45454D !important;
        opacity: 1 !important;
    }

    .stNumberInput button {
        background: #C3C3CC !important;
        color: #17171B !important;
        border-color: #A6A6B0 !important;
    }

    /* Select boxes */

    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background: #D8D8DE !important;
        color: #111116 !important;
        border-color: #A6A6B0 !important;
        border-radius: 8px !important;
    }

    .stSelectbox [data-baseweb="select"] *,
    .stMultiSelect [data-baseweb="select"] * {
        color: #111116 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background: #E0E0E6 !important;
    }

    [role="option"] {
        color: #111116 !important;
    }

    /* CSV uploader */

    [data-testid="stFileUploader"] section {
        background: #D8D8DE !important;
        border: 1px dashed #777782 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {
        color: #15151A !important;
    }

    [data-testid="stFileUploader"] button {
        background: #303038 !important;
        color: #FFFFFF !important;
        border: 1px solid #50505A !important;
        border-radius: 7px !important;
    }

    /* Buttons */

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(105deg, #D92F42, #F06435) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,.12) !important;
        border-radius: 9px !important;
        font-weight: 650 !important;
        transition: all .22s ease-in-out;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(105deg, #286CDE, #438DFF) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 18px rgba(55,125,255,.22);
        transform: translateY(-1px);
    }

    /* Metric cards */

    [data-testid="stMetric"] {
        background: rgba(23,23,29,.90);
        border: 1px solid #303039;
        border-radius: 12px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #BFC0CB !important;
    }

    [data-testid="stMetricValue"] {
        color: #FF9345 !important;
    }

    /* Tabs */

    .stTabs [data-baseweb="tab"] {
        color: #D0D0D8;
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #F04452 !important;
    }

    /* Tables */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #303039;
        border-radius: 9px;
    }

    /* Expander headings */

    [data-testid="stExpander"] summary {
        background: #17171D !important;
        color: #FFFFFF !important;
        border: 1px solid #383842 !important;
        border-radius: 8px !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    hr {
        border-color: #303039 !important;
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp::before {
            animation: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATA_PATH = Path(__file__).parent / "cu_ii_training_data.csv"

NUM_FEATURES = [
    "pH",
    "Initial_Cu (mg/l)",
    "Contact_Time (min)",
    "Dose (g/L)"
]

CAT_FEATURES = [
    "Adsorbent_type",
    "Synthesis_method"
]

TARGET = "qe(mg/g)"
GROUP = "Paper_ID"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. "
            "Place the CSV file in the same folder as app.py."
        )

    frame = pd.read_csv(DATA_PATH)

    required_columns = NUM_FEATURES + CAT_FEATURES + [TARGET, GROUP]

    missing_columns = [
        column for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame[TARGET] = pd.to_numeric(
        frame[TARGET],
        errors="coerce"
    )

    for column in NUM_FEATURES:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce"
        )

    frame = frame.dropna(subset=[TARGET, GROUP]).copy()

    if frame.empty:
        raise ValueError(
            "No usable rows remain after removing records without "
            "a target value or Paper_ID."
        )

    return frame


# ============================================================
# BUILD RANDOM FOREST MODEL
# ============================================================

def build_model():
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

    return Pipeline([
        ("preprocess", preprocessor),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                max_features=1.0,
                random_state=42,
                n_jobs=-1
            )
        )
    ])


@st.cache_resource
def fit_model():
    frame = load_data()

    X = frame[NUM_FEATURES + CAT_FEATURES]
    y = frame[TARGET].astype(float)

    model = build_model()
    model.fit(X, y)

    return model, frame


# ============================================================
# INITIALISE MODEL
# ============================================================

try:
    model, frame = fit_model()
except Exception as exc:
    st.error(f"Could not load/train the demo model: {exc}")
    st.stop()


# ============================================================
# TITLE AND INTRODUCTION
# ============================================================

st.title("🌿 Cu(II) Adsorption Capacity — Random Forest Demo")

st.caption(
    "PBL prototype | GO, rGO and related graphene-based adsorbents"
)

st.warning(
    "**Research prototype only — not a validated scientific predictor.** "
    "The current dataset is small, highly imbalanced by source paper, and "
    "some source values still need verification. Treat every output as an "
    "exploratory demo estimate, not an experimental result or an engineering "
    "recommendation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About this model")

    st.write("**Target:** adsorption capacity, qe (mg/g)")

    st.write(
        "**Inputs used:** pH, initial Cu(II), contact time, adsorbent "
        "dose, adsorbent type, and synthesis method."
    )

    st.write(
        "Temperature, BET surface area, and O/C ratio are not model "
        "inputs in this prototype."
    )

    st.write(f"**Training rows:** {len(frame)}")
    st.write(f"**Source papers/labels:** {frame[GROUP].nunique()}")

    st.write(
        "**Algorithm:** RandomForestRegressor "
        "(shallow demo configuration)"
    )

    st.divider()

    st.caption(
        "Model is fitted locally when this app starts. "
        "No external service is called."
    )


# ============================================================
# PREDICTION INPUTS
# ============================================================

st.subheader("1. Enter experimental conditions")

adsorbents = sorted(
    frame["Adsorbent_type"].dropna().astype(str).unique().tolist()
)

methods = sorted(
    frame["Synthesis_method"].dropna().astype(str).unique().tolist()
)

if not adsorbents:
    adsorbents = ["GO", "rGO", "rGO-TiO2"]

if not methods:
    methods = ["Not specified"]

col1, col2, col3 = st.columns(3)

with col1:
    adsorbent = st.selectbox(
        "Adsorbent type",
        adsorbents
    )

    ph = st.number_input(
        "Solution pH",
        min_value=0.0,
        max_value=14.0,
        value=4.0,
        step=0.1
    )

with col2:
    initial_cu = st.number_input(
        "Initial Cu(II) concentration (mg/L)",
        min_value=0.01,
        max_value=100000.0,
        value=50.0,
        step=1.0
    )

    contact_time = st.number_input(
        "Contact time (min)",
        min_value=0.0,
        max_value=100000.0,
        value=60.0,
        step=5.0
    )

with col3:
    dose = st.number_input(
        "Adsorbent dose (g/L)",
        min_value=0.001,
        max_value=10000.0,
        value=1.0,
        step=0.1
    )

    synthesis = st.selectbox(
        "Synthesis method",
        methods
    )

st.caption(
    "Choose values and material categories that resemble the source dataset. "
    "A category not represented in training may produce unreliable estimates."
)


# ============================================================
# PREDICTION AND RANGE WARNINGS
# ============================================================

if st.button(
    "🌱 Predict demo qe",
    type="primary",
    use_container_width=True
):
    input_row = pd.DataFrame([{
        "pH": ph,
        "Initial_Cu (mg/l)": initial_cu,
        "Contact_Time (min)": contact_time,
        "Dose (g/L)": dose,
        "Adsorbent_type": adsorbent,
        "Synthesis_method": synthesis
    }])

    try:
        prediction = float(model.predict(input_row)[0])

        st.subheader("2. Demo output")

        st.metric(
            "Predicted adsorption capacity, qe",
            f"{prediction:.2f} mg/g"
        )

        st.info(
            "This is a model output from a limited literature-derived dataset. "
            "It is not a measured value, and the current cross-paper validation "
            "has not shown reliable generalisation."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

    ranges = {}

    for column, value in [
        ("pH", ph),
        ("Initial_Cu (mg/l)", initial_cu),
        ("Contact_Time (min)", contact_time),
        ("Dose (g/L)", dose)
    ]:
        lower = frame[column].min()
        upper = frame[column].max()

        if pd.notna(lower) and pd.notna(upper):
            ranges[column] = (
                float(lower),
                float(upper),
                float(value)
            )

    out_of_range = [
        (
            f"{column}: entered {value:g}, "
            f"training range {lower:g}–{upper:g}"
        )
        for column, (lower, upper, value) in ranges.items()
        if value < lower or value > upper
    ]

    if out_of_range:
        st.warning(
            "One or more inputs are outside the observed training range:\n\n"
            + "\n\n".join(out_of_range)
        )

    if adsorbent not in set(
        frame["Adsorbent_type"].dropna().astype(str)
    ):
        st.warning(
            "This adsorbent type was not represented in the training data; "
            "the estimate is especially unreliable."
        )


# ============================================================
# MODEL VALIDATION SUMMARY
# ============================================================

with st.expander("Model validation summary (important)"):
    st.write(
        "The previous leave-one-paper-out validation on the current "
        "candidate dataset produced MAE around 61 mg/g for the shallow "
        "Random Forest and R² around −0.01. These results indicate poor "
        "predictive performance on held-out papers."
    )

    st.write(
        "Do not use the demo for treatment design, safety decisions, scale-up, "
        "or replacing laboratory measurements. The next research step is "
        "source verification and adding comparable independent studies."
    )


# ============================================================
# PREVIEW TRAINING DATA
# ============================================================

with st.expander("Preview training data"):
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.caption(
    "Built for a biotechnology PBL demonstration. Document the dataset "
    "provenance and validation limitations in your report."
)


# ============================================================
# DATASET MANAGEMENT
# ============================================================

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
            column
            for column in required_columns
            if column not in uploaded_data.columns
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
                pd.to_numeric,
                errors="coerce"
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

        st.dataframe(
            uploaded_data.head(100),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download uploaded dataset as CSV",
            data=uploaded_data.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_cuii_dataset.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("🌳 Train Random Forest on Uploaded Dataset")

        if st.button(
            "Train model using uploaded dataset",
            key="train_uploaded_model"
        ):
            try:
                training_data = uploaded_data.copy()

                numeric_columns = NUM_FEATURES + [TARGET]

                for column in numeric_columns:
                    training_data[column] = pd.to_numeric(
                        training_data[column],
                        errors="coerce"
                    )

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

                    uploaded_model = build_model()
                    uploaded_model.fit(X, y)

                    st.success(
                        f"Model trained successfully on "
                        f"{len(training_data)} complete rows."
                    )

                    st.warning(
                        "These are training-set metrics, not independent "
                        "validation. They can look better than real-world "
                        "performance. Use paper-grouped validation before "
                        "drawing research conclusions."
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

                    importances = (
                        uploaded_model.named_steps["rf"].feature_importances_
                    )

                    importance_frame = pd.DataFrame({
                        "Feature": transformed_names,
                        "Importance": importances
                    }).sort_values(
                        "Importance",
                        ascending=False
                    )

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

# Choose which dataset to validate.
lopo_sources = ["App's original dataset"]

uploaded_data = None
if uploaded_data is not None:
    lopo_sources.append("Uploaded CSV dataset")

lopo_source = st.selectbox(
    "Dataset to use for LOPO validation",
    options=lopo_sources,
    help="Choose which dataset the validation should evaluate."
)

if st.button("Run Leave-One-Paper-Out Validation"):
    try:
        # Select the requested dataset.
        if lopo_source == "Uploaded CSV dataset":
            validation_data = uploaded_data.copy()

            required_lopo_columns = (
                NUM_FEATURES + CAT_FEATURES + [TARGET, GROUP]
            )

            missing_lopo_columns = [
                col for col in required_lopo_columns
                if col not in validation_data.columns
            ]

            if missing_lopo_columns:
                st.error(
                    "The uploaded CSV is missing required columns: "
                    + ", ".join(missing_lopo_columns)
                )
                st.stop()
        else:
            validation_data = frame.copy()

        # Convert numerical columns safely.
        for numeric_col in NUM_FEATURES + [TARGET]:
            validation_data[numeric_col] = pd.to_numeric(
                validation_data[numeric_col],
                errors="coerce"
            )

        # Remove rows without a target or paper ID.
        validation_data = validation_data.dropna(
            subset=[TARGET, GROUP]
        ).copy()

        # Remove blank paper IDs.
        validation_data[GROUP] = (
            validation_data[GROUP].astype(str).str.strip()
        )

        validation_data = validation_data[
            validation_data[GROUP] != ""
        ].copy()

        number_of_papers = validation_data[GROUP].nunique()

        if number_of_papers < 3:
            st.error(
                "At least three distinct research papers are needed "
                "for this exploratory validation."
            )

        else:
            logo = LeaveOneGroupOut()

            X_val = validation_data[
                NUM_FEATURES + CAT_FEATURES
            ]

            y_val = validation_data[TARGET].astype(float)
            groups = validation_data[GROUP]

            observed_all = []
            predicted_all = []
            paper_results = []
            observation_results = []

            # Hold out one complete paper per validation round.
            for train_idx, test_idx in logo.split(
                X_val,
                y_val,
                groups=groups
            ):
                X_train = X_val.iloc[train_idx]
                X_test = X_val.iloc[test_idx]

                y_train = y_val.iloc[train_idx]
                y_test = y_val.iloc[test_idx]

                # Fit preprocessing only on the training fold.
                fold_preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline([
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="median"
                                    )
                                )
                            ]),
                            NUM_FEATURES
                        ),
                        (
                            "categorical",
                            Pipeline([
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="most_frequent"
                                    )
                                ),
                                (
                                    "onehot",
                                    OneHotEncoder(
                                        handle_unknown="ignore"
                                    )
                                )
                            ]),
                            CAT_FEATURES
                        )
                    ],
                    remainder="drop"
                )

                fold_model = Pipeline([
                    ("preprocess", fold_preprocessor),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=3,
                            min_samples_leaf=2,
                            max_features=1.0,
                            random_state=42,
                            n_jobs=-1
                        )
                    )
                ])

                # Train and predict.
                fold_model.fit(X_train, y_train)
                fold_predictions = fold_model.predict(X_test)

                paper_name = str(groups.iloc[test_idx[0]])

                observed_all.extend(y_test.tolist())
                predicted_all.extend(
                    fold_predictions.tolist()
                )

                # Save every held-out observation.
                fold_observations = (
                    validation_data.iloc[test_idx].copy()
                )

                fold_observations["Observed_qe_mg_g"] = (
                    y_test.to_numpy()
                )

                fold_observations[
                    "LOPO_predicted_qe_mg_g"
                ] = fold_predictions

                fold_observations[
                    "Absolute_error_mg_g"
                ] = np.abs(
                    y_test.to_numpy() - fold_predictions
                )

                fold_observations["Held_out_paper"] = paper_name

                observation_results.append(fold_observations)

                # Calculate metrics for this paper.
                paper_results.append({
                    "Held-out paper": paper_name,
                    "Test rows": len(test_idx),
                    "MAE (mg/g)": mean_absolute_error(
                        y_test,
                        fold_predictions
                    ),
                    "RMSE (mg/g)": np.sqrt(
                        mean_squared_error(
                            y_test,
                            fold_predictions
                        )
                    )
                })

            # Calculate overall validation metrics.
            overall_mae = mean_absolute_error(
                observed_all,
                predicted_all
            )

            overall_rmse = np.sqrt(
                mean_squared_error(
                    observed_all,
                    predicted_all
                )
            )

            overall_r2 = r2_score(
                observed_all,
                predicted_all
            )

            # Give each paper equal weight in macro-average MAE.
            macro_mae = np.mean([
                result["MAE (mg/g)"]
                for result in paper_results
            ])

            st.success(
                f"Validation completed across {number_of_papers} papers."
            )

            st.caption(
                f"Dataset evaluated: {lopo_source} | "
                f"Usable observations: {len(validation_data)}"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "LOPO MAE",
                f"{overall_mae:.2f} mg/g"
            )

            col2.metric(
                "LOPO RMSE",
                f"{overall_rmse:.2f} mg/g"
            )

            col3, col4 = st.columns(2)

            col3.metric(
                "LOPO R²",
                f"{overall_r2:.3f}"
            )

            col4.metric(
                "Macro-average paper MAE",
                f"{macro_mae:.2f} mg/g"
            )

            st.warning(
                "These results are exploratory. High errors or negative "
                "R² indicate that the model may not generalise well to "
                "unseen research papers. Verify the source data before "
                "drawing research conclusions."
            )

            # Show paper-level metrics.
            st.subheader(
                "Performance for each held-out paper"
            )

            paper_results_df = pd.DataFrame(paper_results)

            st.dataframe(
                paper_results_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download LOPO paper-summary results",
                data=paper_results_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="cuii_lopo_paper_summary.csv",
                mime="text/csv",
                key="download_lopo_results"
            )

            # Combine individual predictions into one table.
            observation_results_df = pd.concat(
                observation_results,
                ignore_index=True
            )

            st.subheader(
                "🔎 Individual held-out predictions"
            )

            st.write(
                "Each row represents an observation tested while its "
                "entire source paper was excluded from training. Compare "
                "the actual adsorption capacity with the prediction "
                "to investigate large errors."
            )

            display_columns = [
                col for col in [
                    "Paper_ID",
                    "Observed_qe_mg_g",
                    "LOPO_predicted_qe_mg_g",
                    "Absolute_error_mg_g",
                    "pH",
                    "Initial_Cu (mg/l)",
                    "Contact_Time (min)",
                    "Dose (g/L)",
                    "Adsorbent_type",
                    "Synthesis_method",
                    "Held_out_paper"
                ]
                if col in observation_results_df.columns
            ]

            st.dataframe(
                observation_results_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

            # Download detailed predictions and errors.
            st.download_button(
                "⬇️ Download individual LOPO predictions and errors",
                data=observation_results_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="cuii_lopo_individual_predictions.csv",
                mime="text/csv",
                key="download_lopo_individual_results"
            )

            # Highlight the largest prediction errors.
            st.subheader("⚠️ Largest prediction errors")

            largest_errors = observation_results_df.sort_values(
                "Absolute_error_mg_g",
                ascending=False
            ).head(10)

            st.dataframe(
                largest_errors[display_columns],
                use_container_width=True,
                hide_index=True
            )

    except Exception as exc:
        st.error(
            f"LOPO validation failed: {exc}"
        )
