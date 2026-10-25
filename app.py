import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cu(II) Adsorption | AI Research Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DESIGN: BLACK GRADIENT + ANIMATED PARTICLES
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL BACKGROUND ---------- */

    html, body {
        background: #070709 !important;
    }

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

    /* ---------- ANIMATED GLOWING PARTICLES ---------- */

    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;

        background-image:
            radial-gradient(
                circle,
                rgba(255, 65, 78, 0.72) 1px,
                transparent 2px
            ),
            radial-gradient(
                circle,
                rgba(65, 135, 255, 0.68) 1px,
                transparent 2px
            ),
            radial-gradient(
                circle,
                rgba(255, 150, 65, 0.52) 1px,
                transparent 2px
            ),
            radial-gradient(
                circle,
                rgba(255, 255, 255, 0.38) 1px,
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

        opacity: 0.48;

        animation:
            particleDrift 45s linear infinite,
            particleGlow 7s ease-in-out infinite alternate;
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

    @keyframes particleGlow {
        from {
            opacity: 0.30;
        }

        to {
            opacity: 0.62;
        }
    }

    /* Keep the application interface above the particles. */

    [data-testid="stAppViewContainer"] {
        background: transparent !important;
        position: relative;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 1;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 7, 9, 0.72) !important;
    }

    /* ---------- TYPOGRAPHY ---------- */

    h1, h2, h3 {
        color: #F7F7FA !important;
        letter-spacing: 0.2px;
    }

    h1 {
        font-weight: 750 !important;
        text-shadow: 0 0 24px rgba(255, 55, 70, 0.12);
    }

    p, li, .stMarkdown {
        color: #D1D1D8;
    }

    [data-testid="stCaptionContainer"] {
        color: #A7A7B2 !important;
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #111115 0%,
                #09090C 100%
            ) !important;

        border-right: 1px solid #292932;
    }

    [data-testid="stSidebar"] * {
        color: #E2E2E8;
    }

    /* ---------- GREY INPUT FIELDS ---------- */

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

    /* Number-input controls */

    .stNumberInput button {
        background: #C3C3CC !important;
        color: #17171B !important;
        border-color: #A6A6B0 !important;
    }

    /* ---------- SELECT BOXES ---------- */

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

    /* ---------- FILE UPLOAD AREA ---------- */

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

    /* ---------- BUTTONS ---------- */

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

    /* ---------- METRIC CARDS ---------- */

    [data-testid="stMetric"] {
        background: rgba(23, 23, 29, 0.90);
        border: 1px solid #303039;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #BFC0CB !important;
    }

    [data-testid="stMetricValue"] {
        color: #FF9345 !important;
    }

    /* ---------- TABS ---------- */

    .stTabs [data-baseweb="tab"] {
        color: #D0D0D8;
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #F04452 !important;
    }

    /* ---------- DATA TABLES ---------- */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #303039;
        border-radius: 9px;
    }

    /* ---------- ALERTS ---------- */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ---------- DIVIDERS ---------- */

    hr {
        border-color: #303039 !important;
    }

    /* ---------- ACCESSIBILITY ---------- */

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
# DATASET AND MODEL CONFIGURATION
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
# LOAD THE LITERATURE-DERIVED DATASET
# ============================================================

@st.cache_data
def load_data():
    frame = pd.read_csv(DATA_PATH)

    required_columns = (
        NUM_FEATURES
        + CAT_FEATURES
        + [TARGET, GROUP]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "The training CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    for column in NUM_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce"
        )

    frame = frame.dropna(
        subset=[TARGET, GROUP]
    ).copy()

    if frame.empty:
        raise ValueError(
            "No usable training rows were found."
        )

    return frame


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def fit_model():
    frame = load_data()

    X = frame[NUM_FEATURES + CAT_FEATURES]
    y = frame[TARGET].astype(float)

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

    model.fit(X, y)

    return model, frame


try:
    model, frame = fit_model()

except Exception as exc:
    st.error(
        f"Could not load or train the model: {exc}"
    )
    st.info(
        "Check that app.py and cu_ii_training_data.csv "
        "are both in the repository's root directory."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🧪 AI RESEARCH LAB")

    st.caption(
        "COPPER ADSORPTION | MACHINE LEARNING"
    )

    st.divider()

    st.markdown("### Model information")

    st.write(
        "**Target:** Adsorption capacity, qe (mg/g)"
    )

    st.write(
        "**Algorithm:** Random Forest Regressor"
    )

    st.write(
        "**Input variables:** pH, initial Cu(II), "
        "contact time, dose, adsorbent type, "
        "and synthesis method."
    )

    st.write(
        "**Training observations:** "
        f"{len(frame)}"
    )

    st.write(
        "**Source-paper groups:** "
        f"{frame[GROUP].nunique()}"
    )

    st.caption(
        "Temperature, BET surface area and O/C ratio "
        "are not model inputs in this prototype."
    )

    st.divider()

    st.caption(
        "Research prototype only. Predictions are "
        "exploratory estimates, not experimental results."
    )


# ============================================================
# MAIN DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div style="
        padding: 12px 0 8px 0;
    ">
        <div style="
            color: #FF7658;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 3px;
            margin-bottom: 10px;
        ">
            MACHINE LEARNING • MATERIALS SCIENCE
        </div>

        <h1 style="
            font-size: clamp(28px, 4vw, 42px);
            margin-bottom: 8px;
        ">
            Cu(II) Adsorption Predictor
        </h1>

        <p style="
            font-size: 16px;
            color: #B8B8C4;
            max-width: 780px;
            line-height: 1.7;
        ">
            A Random Forest research prototype for exploring
            copper adsorption using graphene-based materials,
            including GO, rGO and rGO–TiO₂.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.warning(
    "**Research prototype — not a validated scientific predictor.** "
    "The literature-derived dataset is small and unevenly "
    "distributed across source papers. Some extracted values "
    "require verification. Treat predictions as exploratory "
    "estimates, not measured results or engineering recommendations."
)


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

st.markdown("## Research overview")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Training observations",
        f"{len(frame):,}"
    )

with metric2:
    st.metric(
        "Source papers",
        f"{frame[GROUP].nunique():,}"
    )

with metric3:
    st.metric(
        "Mean observed qe",
        f"{frame[TARGET].mean():.2f} mg/g"
    )

with metric4:
    st.metric(
        "Median observed qe",
        f"{frame[TARGET].median():.2f} mg/g"
    )


st.divider()


# ============================================================
# NAVIGATION TABS
# ============================================================

tab_predict, tab_explore, tab_validation, tab_about = st.tabs([
    "🧪  Prediction",
    "📊  Data Explorer",
    "🤖  Model Evaluation",
    "🔬  About"
])


# ============================================================
# TAB 1: PREDICTION
# ============================================================

with tab_predict:

    st.subheader("Predict adsorption capacity")

    st.write(
        "Enter experimental conditions to obtain an "
        "exploratory estimate from the trained Random Forest model."
    )

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
        "Use values comparable to the training dataset. "
        "Predictions outside the observed experimental range "
        "may be particularly unreliable."
    )

    predict_clicked = st.button(
        "⚡ Predict adsorption capacity",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:

        input_row = pd.DataFrame([{
            "pH": ph,
            "Initial_Cu (mg/l)": initial_cu,
            "Contact_Time (min)": contact_time,
            "Dose (g/L)": dose,
            "Adsorbent_type": adsorbent,
            "Synthesis_method": synthesis
        }])

        prediction = float(
            model.predict(input_row)[0]
        )

        st.markdown("### Prediction result")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(
                "Predicted adsorption capacity",
                f"{prediction:.2f} mg/g"
            )

        with result_col2:
            st.markdown(
                f"""
                <div style="
                    background: rgba(25, 25, 32, 0.88);
                    border: 1px solid #34343F;
                    border-left: 4px solid #FF7658;
                    border-radius: 10px;
                    padding: 16px 18px;
                ">
                    <div style="
                        color: #FF9345;
                        font-weight: 700;
                        margin-bottom: 8px;
                    ">
                        MODEL OUTPUT — NOT AN EXPERIMENTAL VALUE
                    </div>

                    <div style="
                        color: #D1D1D8;
                        line-height: 1.7;
                    ">
                        This estimate is generated by a Random Forest
                        trained on the current literature-derived
                        dataset. Cross-paper validation has not
                        demonstrated reliable generalisation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Check whether numerical inputs are within the
        # observed training-data ranges.

        ranges = {}

        entered_values = [
            ("pH", ph),
            ("Initial_Cu (mg/l)", initial_cu),
            ("Contact_Time (min)", contact_time),
            ("Dose (g/L)", dose)
        ]

        for column, value in entered_values:

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
                f"{column}: entered {value:g}; "
                f"observed training range "
                f"{lower:g}–{upper:g}"
            )
            for column, (lower, upper, value)
            in ranges.items()
            if value < lower or value > upper
        ]

        if out_of_range:

            st.warning(
                "Some inputs are outside the observed training "
                "ranges:\n\n"
                + "\n\n".join(out_of_range)
            )

        known_adsorbents = set(
            frame["Adsorbent_type"]
            .dropna()
            .astype(str)
        )

        if adsorbent not in known_adsorbents:

            st.warning(
                "This adsorbent category was not represented "
                "in the training data. The estimate is "
                "especially unreliable."
            )


# ============================================================
# TAB 2: DATA EXPLORER
# ============================================================

with tab_explore:

    st.subheader("Explore the training dataset")

    st.write(
        "Inspect the observations used by the prototype "
        "and explore the distribution of reported adsorption "
        "capacities."
    )

    explore_col1, explore_col2 = st.columns(2)

    with explore_col1:

        st.metric(
            "Available target values",
            f"{frame[TARGET].notna().sum():,}"
        )

    with explore_col2:

        st.metric(
            "Distinct adsorbent categories",
            f"{frame['Adsorbent_type'].nunique():,}"
        )

    st.markdown("### Adsorption capacity distribution")

    qe_values = frame[TARGET].dropna()

    if not qe_values.empty:

        histogram = np.histogram(
            qe_values,
            bins=min(20, max(5, len(qe_values) // 3))
        )

        hist_frame = pd.DataFrame({
            "Frequency": histogram[0]
        })

        st.bar_chart(
            hist_frame,
            use_container_width=True
        )

    st.markdown("### Observations by adsorbent")

    adsorbent_counts = (
        frame["Adsorbent_type"]
        .fillna("Not specified")
        .astype(str)
        .value_counts()
    )

    st.bar_chart(
        adsorbent_counts,
        use_container_width=True
    )

    st.markdown("### Training data preview")

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = frame.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download training data as CSV",
        data=csv_bytes,
        file_name="cu_ii_training_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    st.markdown("### Validate another CSV")

    st.write(
        "Upload a CSV file to inspect its contents and "
        "check whether it contains the columns expected by "
        "the model. Uploading a file here does not retrain "
        "or replace the deployed model."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Upload a CSV file to preview and check its columns."
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            st.success(
                f"Loaded {len(uploaded_df):,} rows and "
                f"{len(uploaded_df.columns):,} columns."
            )

            expected_columns = (
                NUM_FEATURES
                + CAT_FEATURES
                + [TARGET, GROUP]
            )

            missing = [
                column
                for column in expected_columns
                if column not in uploaded_df.columns
            ]

            if missing:

                st.warning(
                    "The uploaded CSV is missing these "
                    "model-related columns: "
                    + ", ".join(missing)
                )

            else:

                st.success(
                    "All required model-related columns "
                    "are present."
                )

            st.dataframe(
                uploaded_df.head(100),
                use_container_width=True,
                hide_index=True
            )

        except Exception as exc:

            st.error(
                f"Could not read the uploaded CSV: {exc}"
            )


# ============================================================
# TAB 3: MODEL EVALUATION
# ============================================================

with tab_validation:

    st.subheader("Model evaluation and limitations")

    st.write(
        "Leave-one-paper-out validation tests how well the "
        "model predicts observations from a source paper "
        "that was excluded from training."
    )

    st.warning(
        "The previously obtained leave-one-paper-out "
        "metrics indicate poor generalisation to unseen "
        "source papers."
    )

    st.markdown("### Previously obtained validation results")

    val_col1, val_col2, val_col3 = st.columns(3)

    with val_col1:

        st.metric(
            "MAE",
            "61.24 mg/g"
        )

    with val_col2:

        st.metric(
            "RMSE",
            "64.07 mg/g"
        )

    with val_col3:

        st.metric(
            "R²",
            "-0.016"
        )

    st.caption(
        "These are the previously recorded validation results. "
        "They are displayed as historical results and are not "
        "recalculated by this section."
    )

    st.divider()

    st.markdown("### Run leave-one-paper-out validation")

    st.write(
        "The following calculation evaluates the current "
        "dataset using Paper_ID as the grouping variable. "
        "Each paper is held out in turn."
    )

    run_validation = st.button(
        "Run validation on current dataset",
        use_container_width=True
    )

    if run_validation:

        validation_data = frame.dropna(
            subset=[TARGET, GROUP]
        ).copy()

        unique_groups = validation_data[GROUP].nunique()

        if unique_groups < 2:

            st.error(
                "At least two distinct source-paper groups "
                "are needed for leave-one-paper-out validation."
            )

        else:

            X = validation_data[
                NUM_FEATURES + CAT_FEATURES
            ]

            y = validation_data[TARGET].astype(float)

            groups = validation_data[GROUP]

            logo = LeaveOneGroupOut()

            actual_values = []
            predicted_values = []
            group_results = []

            for train_index, test_index in logo.split(
                X,
                y,
                groups
            ):

                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]

                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]

                if X_train.empty or X_test.empty:
                    continue

                validation_model = Pipeline([
                    (
                        "preprocess",
                        ColumnTransformer(
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
                    ),
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

                validation_model.fit(
                    X_train,
                    y_train
                )

                predictions = validation_model.predict(
                    X_test
                )

                actual_values.extend(
                    y_test.tolist()
                )

                predicted_values.extend(
                    predictions.tolist()
                )

                paper_name = str(
                    groups.iloc[test_index[0]]
                )

                group_mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                group_rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

                group_results.append({
                    "Paper_ID": paper_name,
                    "Test rows": len(test_index),
                    "MAE (mg/g)": group_mae,
                    "RMSE (mg/g)": group_rmse
                })

            if actual_values:

                overall_mae = mean_absolute_error(
                    actual_values,
                    predicted_values
                )

                overall_rmse = np.sqrt(
                    mean_squared_error(
                        actual_values,
                        predicted_values
                    )
                )

                overall_r2 = r2_score(
                    actual_values,
                    predicted_values
                )

                result1, result2, result3 = st.columns(3)

                with result1:

                    st.metric(
                        "LOPO MAE",
                        f"{overall_mae:.2f} mg/g"
                    )

                with result2:

                    st.metric(
                        "LOPO RMSE",
                        f"{overall_rmse:.2f} mg/g"
                    )

                with result3:

                    st.metric(
                        "LOPO R²",
                        f"{overall_r2:.3f}"
                    )

                st.markdown(
                    "### Results by held-out paper"
                )

                results_df = pd.DataFrame(
                    group_results
                )

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "A negative R² indicates that the predictions "
                    "perform worse than predicting the overall "
                    "mean of the observed target values under "
                    "this validation setup. The model requires "
                    "more comparable, independently verified "
                    "data before it can be considered reliable."
                )

            else:

                st.error(
                    "Validation could not produce any predictions."
                )

    st.divider()

    st.markdown("### Interpretation")

    st.write(
        """
        - **MAE:** Average absolute difference between observed
          and predicted adsorption capacities.
        - **RMSE:** Error metric that penalises large prediction
          errors more heavily.
        - **R²:** Indicates how well predictions explain the
          variation in the observed target values.
        - **Leave-one-paper-out validation:** Tests generalisation
          to a source paper excluded from model training.
        """
    )

    st.error(
        "Do not use this prototype for engineering design, "
        "treatment decisions, safety decisions, scale-up, or "
        "replacement of laboratory measurements."
    )


# ============================================================
# TAB 4: ABOUT THE PROJECT
# ============================================================

with tab_about:

    st.subheader("About this research project")

    st.markdown(
        """
        ### Research objective

        To explore whether a Random Forest regression model can
        estimate Cu(II) adsorption capacity from selected
        experimental conditions and adsorbent characteristics
        extracted from scientific literature.

        ### Materials of interest

        - **GO:** Graphene oxide
        - **rGO:** Reduced graphene oxide
        - **rGO–TiO₂:** Reduced graphene oxide combined with
          titanium dioxide

        The availability of each material category depends on
        the current training dataset.

        ### Methodology

        1. Extract adsorption observations from published studies.
        2. Organise the available experimental variables into a
           structured dataset.
        3. Impute missing numerical values using training-set
           medians and missing categorical values using the
           most frequent category.
        4. One-hot encode categorical variables.
        5. Train a Random Forest regression model.
        6. Examine performance using leave-one-paper-out
           validation.

        ### Model inputs

        - Solution pH
        - Initial Cu(II) concentration
        - Contact time
        - Adsorbent dose
        - Adsorbent type
        - Synthesis method

        ### Important limitations

        The dataset is small, the distribution of observations
        across papers is uneven, and source-level verification
        remains important. Experimental conditions may differ
        between studies.

        Model outputs should therefore be treated as exploratory
        estimates rather than experimentally established values.

        ### Future improvements

        - Verify extracted values against the original papers.
        - Increase the number of comparable independent studies.
        - Document units and experimental conditions consistently.
        - Investigate additional validated material descriptors.
        - Compare baseline models and perform careful
          paper-level validation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 10px 0 24px 0;
        color: #898995;
        font-size: 12px;
        line-height: 1.8;
    ">
        CU(II) ADSORPTION • RANDOM FOREST RESEARCH PROTOTYPE
        <br>
        Biotechnology PBL | Exploratory analysis only
    </div>
    """,
    unsafe_allow_html=True
)
