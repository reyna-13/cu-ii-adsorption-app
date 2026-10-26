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
# DESIGN: BLACK GRADIENT, GLOWING DOTS, COLOURED INPUTS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                ellipse at 12% 8%,
                rgba(110, 25, 43, 0.20) 0%,
                transparent 42%
            ),
            radial-gradient(
                ellipse at 88% 15%,
                rgba(28, 48, 95, 0.20) 0%,
                transparent 45%
            ),
            radial-gradient(
                ellipse at 55% 100%,
                rgba(95, 48, 20, 0.09) 0%,
                transparent 45%
            ),
            linear-gradient(
                135deg,
                #060607 0%,
                #101014 48%,
                #08080B 100%
            ) !important;
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
        background: rgba(7, 7, 9, 0.65) !important;
    }

    /* ANIMATED GLOWING DOTS */

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;

        background-image:
            radial-gradient(
                circle,
                rgba(255, 65, 78, 0.65) 1px,
                transparent 2px
            ),
            radial-gradient(
                circle,
                rgba(65, 135, 255, 0.60) 1px,
                transparent 2px
            ),
            radial-gradient(
                circle,
                rgba(255, 150, 65, 0.48) 1px,
                transparent 2px
            ),
            radial-gradient(
                circle,
                rgba(255, 255, 255, 0.32) 1px,
                transparent 2px
            );

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

        opacity: 0.45;
        animation: particleDrift 45s linear infinite;
    }

    @keyframes particleDrift {
        0% {
            background-position:
                0 0,
                45px 60px,
                100px 25px,
                30px 110px;
        }

        50% {
            background-position:
                65px 90px,
                125px 145px,
                205px 105px,
                155px 225px;
        }

        100% {
            background-position:
                135px 180px,
                225px 240px,
                325px 205px,
                295px 370px;
        }
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111115 0%,
            #09090C 100%
        ) !important;
        border-right: 1px solid #292932;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #D1D1D8 !important;
    }

    /* HEADINGS */

    h1, h2, h3 {
        color: #F7F7FA !important;
        letter-spacing: 0.2px;
    }

    h1 {
        text-shadow: 0 0 24px rgba(255, 55, 70, 0.12);
    }

    p, li, .stMarkdown {
        color: #D1D1D8;
    }

    [data-testid="stCaptionContainer"] {
        color: #A7A7B2 !important;
    }

    /* INPUT FIELDS: GREY BACKGROUND, BLACK TEXT */

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

    /* SELECT BOXES */

    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: #D8D8DE !important;
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
        background-color: #E0E0E6 !important;
    }

    [role="option"] {
        color: #111116 !important;
    }

    /* FILE UPLOAD AREA */

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

    /* BUTTONS */

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(
            105deg,
            #D92F42 0%,
            #F06435 100%
        ) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        font-weight: 650 !important;
        transition: all 0.22s ease-in-out;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(
            105deg,
            #286CDE 0%,
            #438DFF 100%
        ) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 18px rgba(55, 125, 255, 0.22);
        transform: translateY(-1px);
    }

    /* METRIC CARDS */

    [data-testid="stMetric"] {
        background: rgba(23, 23, 29, 0.90);
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

    /* TABS */

    .stTabs [data-baseweb="tab"] {
        color: #D0D0D8;
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #F04452 !important;
    }

    /* TABLES */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #303039;
        border-radius: 9px;
    }

    /* DIVIDERS */

    hr {
        border-color: #303039 !important;
    }

    /* ACCESSIBILITY */

    @media (prefers-reduced-motion: reduce) {
        .stApp::before {
            animation: none !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CONFIGURATION
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

    required_columns = (
        NUM_FEATURES + CAT_FEATURES + [TARGET, GROUP]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "The CSV is missing these required columns: "
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

    frame = frame.dropna(
        subset=[TARGET, GROUP]
    ).copy()

    if frame.empty:
        raise ValueError(
            "No usable rows remain after removing records "
            "without a target value or Paper_ID."
        )

    return frame


# ============================================================
# BUILD THE MODEL
# ============================================================

def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    )
                ]),
                NUM_FEATURES
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]),
                CAT_FEATURES
            )
        ],
        remainder="drop"
    )

    model = Pipeline([
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

    return model


@st.cache_resource
def fit_model():
    frame = load_data()

    X = frame[NUM_FEATURES + CAT_FEATURES]
    y = frame[TARGET].astype(float)

    model = build_model()
    model.fit(X, y)

    return model, frame


# ============================================================
# LOAD AND TRAIN SAFELY
# ============================================================

try:
    model, frame = fit_model()

except Exception as exc:
    st.error(
        f"Could not load the dataset or train the model: {exc}"
    )
    st.stop()


# ============================================================
# TITLE AND INTRODUCTION
# ============================================================

st.title(
    "🌿 Cu(II) Adsorption Capacity — Random Forest Demo"
)

st.caption(
    "PBL prototype | GO, rGO and related graphene-based adsorbents"
)

st.warning(
    "**Research prototype only — not a validated scientific predictor.** "
    "The current dataset is small, highly imbalanced by source paper, "
    "and some source values still need verification. Treat every output "
    "as an exploratory demo estimate, not an experimental result or "
    "an engineering recommendation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About this model")

    st.write(
        "**Target:** adsorption capacity, qe (mg/g)"
    )

    st.write(
        "**Inputs used:** pH, initial Cu(II), contact time, "
        "adsorbent dose, adsorbent type, and synthesis method."
    )

    st.write(
        "Temperature, BET surface area, and O/C ratio are not "
        "model inputs in this prototype."
    )

    st.write(
        f"**Training rows:** {len(frame)}"
    )

    st.write(
        f"**Source papers/labels:** {frame[GROUP].nunique()}"
    )

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
    frame["Adsorbent_type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

methods = sorted(
    frame["Synthesis_method"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
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
    "Choose values and material categories that resemble the source "
    "dataset. A category not represented in training may produce "
    "unreliable estimates."
)


# ============================================================
# PREDICTION
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
            "This is a model output from a limited literature-derived "
            "dataset. It is not a measured value, and the current "
            "cross-paper validation has not shown reliable generalisation."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

    # Compare input values with the observed training ranges.
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
            "One or more inputs are outside the observed "
            "training range:\n\n"
            + "\n\n".join(out_of_range)
        )

    if adsorbent not in set(
        frame["Adsorbent_type"].dropna().astype(str)
    ):
        st.warning(
            "This adsorbent type was not represented in the "
            "training data; the estimate is especially unreliable."
        )


# ============================================================
# MODEL VALIDATION: MAE, RMSE, R² AND LOPO
# ============================================================

with st.expander(
    "Model validation: MAE, RMSE, R² and per-paper checking",
    expanded=True
):
    st.write(
        "Leave-One-Paper-Out (LOPO) validation holds out each source "
        "paper in turn, trains on the remaining papers, and evaluates "
        "predictions on the held-out paper. This tests generalisation "
        "to an unseen source."
    )

    st.info(
        "Previously recorded overall results: MAE = 61.24 mg/g, "
        "RMSE = 64.07 mg/g, R² = -0.016. These are historical results, "
        "not recalculated values. Run the validation below to calculate "
        "metrics from the current CSV and code."
    )

    if st.button(
        "Run validation checks",
        key="run_lopo_validation",
        use_container_width=True
    ):
        validation_data = frame.dropna(
            subset=[TARGET, GROUP]
        ).copy()

        X_val = validation_data[
            NUM_FEATURES + CAT_FEATURES
        ]

        y_val = validation_data[TARGET].astype(float)
        groups_val = validation_data[GROUP]

        if groups_val.nunique() < 2:
            st.error(
                "LOPO validation requires at least two distinct Paper_ID groups."
            )

        else:
            logo = LeaveOneGroupOut()

            actual_all = []
            predicted_all = []
            paper_results = []

            for train_idx, test_idx in logo.split(
                X_val,
                y_val,
                groups_val
            ):
                X_train = X_val.iloc[train_idx]
                X_test = X_val.iloc[test_idx]

                y_train = y_val.iloc[train_idx]
                y_test = y_val.iloc[test_idx]

                # Fit preprocessing separately in each fold to prevent leakage.
                fold_model = build_model()
                fold_model.fit(X_train, y_train)

                predictions = fold_model.predict(X_test)

                actual_all.extend(y_test.tolist())
                predicted_all.extend(predictions.tolist())

                paper_id = str(groups_val.iloc[test_idx[0]])

                paper_results.append({
                    "Paper_ID": paper_id,
                    "Test observations": len(test_idx),
                    "MAE (mg/g)": mean_absolute_error(
                        y_test,
                        predictions
                    ),
                    "RMSE (mg/g)": float(
                        np.sqrt(
                            mean_squared_error(
                                y_test,
                                predictions
                            )
                        )
                    )
                })

            overall_mae = mean_absolute_error(
                actual_all,
                predicted_all
            )

            overall_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        actual_all,
                        predicted_all
                    )
                )
            )

            overall_r2 = r2_score(
                actual_all,
                predicted_all
            )

            st.subheader("Overall LOPO validation metrics")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "MAE",
                f"{overall_mae:.2f} mg/g"
            )

            metric2.metric(
                "RMSE",
                f"{overall_rmse:.2f} mg/g"
            )

            metric3.metric(
                "R²",
                f"{overall_r2:.3f}"
            )

            st.subheader("Per-paper validation results")

            results_df = pd.DataFrame(paper_results)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download validation results (CSV)",
                data=results_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="cu_ii_lopo_validation_results.csv",
                mime="text/csv"
            )

            if overall_r2 < 0:
                st.warning(
                    "R² is negative, meaning this model performed worse "
                    "than a mean-prediction baseline on these held-out "
                    "papers. Treat the model as exploratory, not a "
                    "reliable predictor."
                )

            st.caption(
                "MAE is the average absolute prediction error. "
                "RMSE penalises larger errors more strongly. "
                "R² compares predictions with a mean-based baseline. "
                "Results depend on the current dataset."
            )


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander("Preview training data"):
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built for a biotechnology PBL demonstration. "
    "Document the dataset provenance and validation limitations "
    "in your report."
)
