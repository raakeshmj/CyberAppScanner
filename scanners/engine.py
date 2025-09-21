# scanners/engine.py

import re
import requests
from androguard.misc import AnalyzeAPK

# --- CONFIGURATION ---

API_CALL_PERMISSION_MAP = {
    "Landroid/hardware/Camera;->open": "android.permission.CAMERA",
    "Landroid/media/AudioRecord;->startRecording": "android.permission.RECORD_AUDIO",
    "Landroid/telephony/SmsManager;->sendTextMessage": "android.permission.SEND_SMS",
    "Landroid/content/ContentResolver;->query": "android.permission.READ_CONTACTS",
    "Ljava/net/URL;->openConnection": "android.permission.INTERNET"
}

DANGEROUS_PERMISSIONS = {
    "android.permission.READ_CONTACTS": "Reads user's contacts.",
    "android.permission.CAMERA": "Accesses the camera.",
    "android.permission.RECORD_AUDIO": "Records audio with the microphone.",
    "android.permission.SEND_SMS": "Sends SMS messages.",
    "android.permission.ACCESS_FINE_LOCATION": "Accesses precise GPS location."
}

SECRET_PATTERNS = {
    "Google API Key": re.compile(r'AIza[0-9A-Za-z-_]{35}'),
    "Amazon AWS Key": re.compile(r'(A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}'),
    "Firebase URL": re.compile(r'https://[a-zA-Z0-9_-]+\.firebaseio\.com')
}

HTTP_ALLOWLIST = [
    re.compile(r'http://127\.0\.0\.1'),
    re.compile(r'http://localhost'),
    re.compile(r'http://schemas\.android\.com'),
    re.compile(r'http://192\.168\.\d+\.\d+'),
    re.compile(r'http://10\.\d+\.\d+\.\d+')
]

SENSITIVE_KEYWORDS = ["login", "password", "credential", "token", "auth", "secret", "key", "user", "account"]

# --- NEW: Configuration for new scanners ---
INSECURE_STORAGE_APIS = {
    "Landroid/content/SharedPreferences;": "Insecure SharedPreferences",
    "MODE_WORLD_READABLE": "World Readable File",
    "MODE_WORLD_WRITEABLE": "World Writeable File"
}

# --- CORE ANALYSIS ORCHESTRATOR ---

def analyze_apk(apk_path): # Keep this function as is
    try:
        a, d, dx = AnalyzeAPK(apk_path)
        return a, d, dx
    except Exception as e:
        print(f"[!] Critical error during APK analysis: {e}")
        return None, None, None

def run_all_scans(apk_obj, analysis_obj): # MODIFIED: accept objects instead of path
    """Main function to run all available scans and return a consolidated report."""
    if not analysis_obj:
        return {"error": "Invalid analysis object provided."}

    # Run each analysis module
    permissions_report = analyze_permissions(apk_obj, analysis_obj)
    leaks_report = find_leaks(analysis_obj)
    sca_report = find_vulnerable_libraries(analysis_obj)
    storage_report = find_insecure_storage(analysis_obj)

    # Consolidate results
    full_report = {
        "permissions": permissions_report,
        "leaks": leaks_report,
        "sca": sca_report,
        "storage": storage_report
    }
    return full_report

# --- ANALYSIS MODULES (Scanners) ---

def analyze_permissions(apk_obj, analysis_obj):
    requested_perms = apk_obj.get_permissions()
    used_perms_by_code = set()
    for method in analysis_obj.get_methods():
        if method.is_external(): continue
        for _, call, _ in method.get_xref_to():
            api_call_signature = f"{call.class_name}->{call.name}"
            if api_call_signature in API_CALL_PERMISSION_MAP:
                used_perms_by_code.add(API_CALL_PERMISSION_MAP[api_call_signature])
    
    return {
        "requested": requested_perms,
        "used_in_code": list(used_perms_by_code)
    }

def find_leaks(analysis_obj):
    found_secrets, low_risk_endpoints, high_risk_endpoints = [], [], []
    url_pattern = re.compile(r'https?://[^\s",\']+')
    strings_analysis = analysis_obj.get_strings_analysis()

    for str_value, str_info in strings_analysis.items():
        for key_type, pattern in SECRET_PATTERNS.items():
            if pattern.search(str_value):
                found_secrets.append(f"Type: {key_type}, Value: {str_value}")
        
        urls = url_pattern.findall(str_value)
        for url in urls:
            if url.startswith('http://'):
                if any(pattern.search(url) for pattern in HTTP_ALLOWLIST): continue
                is_high_risk = False
                for _, method_obj in str_info.get_xref_from():
                    method_source = method_obj.method.get_source()
                    if method_source and any(keyword in method_source.lower() for keyword in SENSITIVE_KEYWORDS):
                        is_high_risk = True
                        break
                if is_high_risk: high_risk_endpoints.append(url)
                else: low_risk_endpoints.append(url)

    return {
        "secrets": list(set(found_secrets)),
        "low_risk_http": list(set(low_risk_endpoints)),
        "high_risk_http": list(set(high_risk_endpoints))
    }

def find_vulnerable_libraries(analysis_obj):
    """
    Identifies common libraries and checks a (very basic) local database for vulnerabilities.
    A real-world tool would use a live CVE feed.
    """
    detected_libs = {}
    # A simplified local database of library package prefixes and their CVEs
    # Format: "package.prefix": {"name": "Library Name", "cve": "CVE-ID", "details": "..."}
    CVE_DATABASE = {
        "com.squareup.okhttp": {"name": "OkHttp (Older versions)", "cve": "CVE-2021-0341", "details": "Vulnerable to man-in-the-middle attacks."},
        "org.apache.cordova": {"name": "Apache Cordova (Older versions)", "cve": "CVE-2020-11999", "details": "Potential for arbitrary code execution."}
    }

    class_names = {cls.name.replace('/', '.') for cls in analysis_obj.get_classes()}
    for lib_prefix, cve_info in CVE_DATABASE.items():
        for cls_name in class_names:
            if cls_name.startswith(lib_prefix):
                detected_libs[cve_info['name']] = cve_info
                break # Found one class from this lib, no need to check others
    
    return {"vulnerabilities": list(detected_libs.values())}

def find_insecure_storage(analysis_obj):
    """Finds common insecure data storage patterns."""
    findings = []
    strings_analysis = analysis_obj.get_strings_analysis()

    for str_value, str_info in strings_analysis.items():
        for api, description in INSECURE_STORAGE_APIS.items():
            if api in str_value:
                # Find where this insecure API is being used
                for _, method_obj in str_info.get_xref_from():
                    location = f"In method: {method_obj.method.name}"
                    findings.append({"finding": description, "location": location})

    return {"findings": findings}