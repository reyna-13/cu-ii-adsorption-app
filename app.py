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
