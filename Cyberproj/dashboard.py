# dashboard.py

import streamlit as st
import tempfile
import os
import pandas as pd
import altair as alt
from scanners import engine, ml_scanner


def load_css():
    st.markdown("""
        <style>
            .stApp { background-color:#0E1117; color: #e6eef3; }
            .stMetric { 
                background-color:#262730;
                padding:15px;
                border-radius:12px;
                border:1px solid rgba(255,255,255,0.1);
            }
        </style>
    """, unsafe_allow_html=True)


def create_risk_chart(score_data):
    df = pd.DataFrame(score_data)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Score:Q", title="Risk Contribution"),
            y=alt.Y("Category:N", sort="-x"),
            color=alt.Color("Category:N", legend=None),
        )
        .properties(title="Static Risk Breakdown")
    )
    return chart


def main():
    st.set_page_config(page_title="PrivacyGuard Pro", page_icon="🛡️", layout="wide")
    load_css()

    st.image("https://i.imgur.com/v80Xf6b.png", width=100)
    st.title("PrivacyGuard Pro – MAST Platform")
    st.subheader("Static Analysis + AI Malware Detection")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload APK to begin analysis", type=["apk"]
    )

    if uploaded_file is None:
        st.info("Upload an APK file to continue.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".apk") as tmp:
        tmp.write(uploaded_file.getvalue())
        apk_path = tmp.name

    try:
        with st.spinner("Analyzing APK..."):

            apk_obj, dex_objs, analysis_obj = engine.analyze_apk(apk_path)
            static_report = engine.run_all_scans(apk_obj, analysis_obj)
            ml_report = ml_scanner.predict(analysis_obj)

        st.header("🧠 AI Malware Verdict")

        if "error" in ml_report:
            st.error(ml_report["error"])
        else:
            prob = ml_report["malware_probability"]
            verdict = ml_report["verdict"]
            source = ml_report["source"]

            icon = "🚨" if verdict == "Malicious" else "⚠️" if verdict == "Suspicious" else "✅"

            st.metric(
                f"AI Verdict ({source.upper()} model) {icon}",
                f"{prob:.1%}",
                help="Probability that this APK matches known malware patterns."
            )

            st.progress(min(max(prob, 0.0), 1.0))

        st.divider()
        st.header("⚙️ Static Analysis Findings")

        permissions_report = static_report["permissions"]
        leaks_report = static_report["leaks"]
        sca_report = static_report["sca"]
        storage_report = static_report["storage"]

        requested = permissions_report.get("requested", [])
        used = permissions_report.get("used_in_code", [])

        dangerous_used = [
            p for p in requested if p in engine.DANGEROUS_PERMISSIONS and p in used
        ]
        dangerous_unused = [
            p for p in requested if p in engine.DANGEROUS_PERMISSIONS and p not in used
        ]

        # scoring
        unjustified_score = len(dangerous_unused) * 25
        justified_score = len(dangerous_used) * 5
        secret_score = len(leaks_report.get("secrets", [])) * 15
        high_endpoints = len(leaks_report.get("high_risk_http", [])) * 20
        low_endpoints = len(leaks_report.get("low_risk_http", [])) * 5
        sca_score = len(sca_report["vulnerabilities"]) * 50
        storage_score = len(storage_report.get("findings", [])) * 10

        total_score = (
            unjustified_score + justified_score + secret_score + high_endpoints +
            low_endpoints + sca_score + storage_score
        )

        risk_level = (
            "Critical 🚨" if total_score > 100 else
            "High ⚠️" if total_score > 50 else
            "Medium" if total_score > 20 else
            "Low ✅"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Static Risk Score", total_score)
            st.metric("Overall Risk Level", risk_level)
            st.metric("Vulnerable Libraries (CVE)", len(sca_report["vulnerabilities"]))
            st.metric("High-Risk Findings", len(dangerous_unused) + len(leaks_report["high_risk_http"]))

        with col2:
            data = {
                "Category": [
                    "Unjustified Permissions",
                    "Justified Permissions",
                    "Secrets",
                    "High-Risk Endpoints",
                    "Low-Risk Endpoints",
                    "Vulnerable Libraries",
                    "Insecure Storage",
                ],
                "Score": [
                    unjustified_score, justified_score, secret_score,
                    high_endpoints, low_endpoints, sca_score, storage_score,
                ],
            }
            st.altair_chart(create_risk_chart(data), use_container_width=True)

        st.divider()
        st.subheader("📦 Vulnerable Libraries (SCA)")

        if sca_report["vulnerabilities"]:
            for v in sca_report["vulnerabilities"]:
                st.error(f"**{v['name']} ({v['cve']})**")
                st.caption(v["details"])
        else:
            st.success("No vulnerable libraries found.")

        st.subheader("🗄️ Insecure Storage")
        if storage_report.get("findings"):
            st.dataframe(pd.DataFrame(storage_report["findings"]), use_container_width=True)
        else:
            st.success("No storage issues detected.")

    finally:
        try:
            os.remove(apk_path)
        except:
            pass


if __name__ == "__main__":
    main()
