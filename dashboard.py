# dashboard.py (Stable version with modern UI)

import streamlit as st
import tempfile
import os
import pandas as pd
import altair as alt
from scanners import engine, ml_scanner

def load_css():
    """Injects custom CSS for modern styling and animations."""
    st.markdown("""
        <style>
            .stApp {
                background-color: #0E1117;
            }
            @keyframes fadeIn {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            .st-emotion-cache-1r4qj8v, .st-emotion-cache-1y4p8pa, .st-emotion-cache-ff22k5 {
                animation: fadeIn 0.5s ease-out;
            }
            .stMetric {
                background-color: #262730;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        </style>
    """, unsafe_allow_html=True)

def create_risk_chart(score_data):
    """Creates a dynamic bar chart for risk contribution."""
    df = pd.DataFrame(score_data)
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X('Score:Q', title='Contribution to Risk Score'),
        y=alt.Y('Category:N', sort='-x', title=None),
        color=alt.Color('Category:N',
                        scale=alt.Scale(
                            domain=['Vulnerable Libraries', 'High-Risk Endpoints', 'Unjustified Permissions', 'Insecure Storage', 'Secrets', 'Justified Permissions', 'Low-Risk Endpoints'],
                            range=['#c11d1d', '#d62728', '#ff4b4b', '#f56c02', '#ff8c00', '#ffc107', '#a0a0a0']),
                        legend=None)).properties(title='Static Analysis Risk Breakdown')
    return chart

def main():
    st.set_page_config(page_title="PrivacyGuard Pro", page_icon="🛡️", layout="wide")
    load_css()
    st.image("https://i.imgur.com/v80Xf6b.png", width=100)
    st.title("PrivacyGuard Pro MAST")
    st.markdown("##### Hybrid Analysis Platform: Static Analysis + AI-Powered Malware Detection")
    st.divider()

    uploaded_file = st.file_uploader("Drag and drop your APK here to begin analysis", type=["apk"], label_visibility="collapsed")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".apk") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            apk_path = tmp_file.name

        static_report = None
        ml_report = None

        with st.spinner('Running static and AI analysis suites...'):
            apk_obj, dex_objs, analysis_obj = engine.analyze_apk(apk_path)
            if not analysis_obj:
                st.error("Fatal error: Could not analyze the APK.")
                os.remove(apk_path)
                return

            static_report = engine.run_all_scans(apk_obj, analysis_obj)
            ml_report = ml_scanner.predict(analysis_obj)

        # --- AI Malware Analysis Section ---
        st.header("🧠 AI-Powered Threat Verdict")
        if ml_report.get("error"):
            st.error(f"AI analysis failed: {ml_report['error']}")
        else:
            prob = ml_report.get("malware_probability", 0)
            verdict = ml_report.get("verdict", "Unknown")
            
            if verdict == "Malicious": color = "red"; icon = "🚨"
            elif verdict == "Suspicious": color = "orange"; icon = "⚠️"
            else: color = "green"; icon = "✅"
            
            st.metric(label=f"AI Verdict: {verdict} {icon}", value=f"{prob:.1%}", help="The model's confidence that this app's patterns match known malware.")
            st.progress(prob, text=f"Malware Probability")

        # --- Static Analysis Section ---
        st.divider()
        st.header("⚙️ Static Analysis Deep Dive")

        permissions_report = static_report.get("permissions", {})
        leaks_report = static_report.get("leaks", {})
        sca_report = static_report.get("sca", {})
        storage_report = static_report.get("storage", {})

        requested_perms = permissions_report.get("requested", [])
        used_perms = permissions_report.get("used_in_code", [])
        dangerous_and_used = {p for p in requested_perms if p in engine.DANGEROUS_PERMISSIONS and p in used_perms}
        dangerous_and_unused = {p for p in requested_perms if p in engine.DANGEROUS_PERMISSIONS and p not in used_perms}

        unjustified_score = len(dangerous_and_unused) * 25
        justified_score = len(dangerous_and_used) * 5
        secrets_score = len(leaks_report.get("secrets", [])) * 15
        low_risk_endpoints_score = len(leaks_report.get("low_risk_http", [])) * 5
        high_risk_endpoints_score = len(leaks_report.get("high_risk_http", [])) * 20
        sca_score = len(sca_report.get("vulnerabilities", [])) * 50
        storage_score = len(storage_report.get("findings", [])) * 10
        total_score = (unjustified_score + justified_score + secrets_score + low_risk_endpoints_score + high_risk_endpoints_score + sca_score + storage_score)
        
        risk_level = "Low"
        if total_score > 100: risk_level = "Critical 🚨"
        elif total_score > 50: risk_level = "High ⚠️"
        elif total_score > 20: risk_level = "Medium"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Static Risk Score", value=total_score)
            st.metric(label="Static Risk Level", value=risk_level)
            st.metric(label="Known Vulnerabilities (CVEs)", value=len(sca_report.get("vulnerabilities", [])))
            st.metric(label="High-Risk Findings", value=len(dangerous_and_unused) + len(leaks_report.get("high_risk_http", [])))
        with col2:
            score_data = {"Category": ["Vulnerable Libraries", "High-Risk Endpoints", "Unjustified Permissions", "Insecure Storage", "Secrets", "Justified Permissions", "Low-Risk Endpoints"],"Score": [sca_score, high_risk_endpoints_score, unjustified_score, storage_score, secrets_score, justified_score, low_risk_endpoints_score]}
            risk_chart = create_risk_chart(score_data)
            st.altair_chart(risk_chart, use_container_width=True)

        st.divider()
        with st.expander(f"🔴 Vulnerable Libraries Found ({len(sca_report.get('vulnerabilities', []))})", expanded=True):
             if sca_report.get("vulnerabilities"):
                 for vuln in sca_report["vulnerabilities"]:
                     st.error(f"**{vuln['name']} ({vuln['cve']})**", icon=" L"); st.caption(vuln['details'])
             else:
                 st.success("No known vulnerable libraries detected.", icon="✅")
        
        with st.expander(f"🟠 Insecure Data Storage ({len(storage_report.get('findings', []))})"):
            if storage_report.get("findings"):
                df = pd.DataFrame(storage_report["findings"])
                st.dataframe(df, use_container_width=True)
            else:
                st.success("No common insecure storage patterns found.", icon="✅")

        os.remove(apk_path)

if __name__ == "__main__":
    main()