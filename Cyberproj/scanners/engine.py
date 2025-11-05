# scanners/engine.py

DANGEROUS_PERMISSIONS = {
    "android.permission.SEND_SMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.CAMERA",
    "android.permission.READ_CONTACTS",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.WRITE_EXTERNAL_STORAGE",
}


def analyze_apk(apk_path):
    """
    Stub — replace with real Androguard logic.
    """
    permissions = [
        "android.permission.INTERNET",
        "android.permission.SEND_SMS",
        "android.permission.CAMERA",
    ]

    api_calls = [
        "Landroid/telephony/SmsManager;->sendTextMessage",
        "Ljava/net/URL;->openConnection"
    ]

    analysis_obj = {
        "permissions": {
            "requested": permissions,
            "used_in_code": ["android.permission.INTERNET"]
        },
        "api_calls": api_calls,
        "static": { }   # not used now
    }

    return {}, {}, analysis_obj


def run_all_scans(apk_obj, analysis_obj):
    return {
        "permissions": analysis_obj["permissions"],
        "leaks": {"secrets": [], "high_risk_http": [], "low_risk_http": []},
        "sca": {"vulnerabilities": []},
        "storage": {"findings": []},
    }
