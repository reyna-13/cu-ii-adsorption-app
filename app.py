1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
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
# All CSS is contained inside this single style block.
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- BLACK GRADIENT BACKGROUND ---------- */

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

    /* Keep the main Streamlit containers transparent
       so the gradient can show through. */

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    section.main {
        background: transparent !important;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 7, 9, 0.65) !important;
    }

    /* ---------- ANIMATED GLOWING DOTS ---------- */

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
