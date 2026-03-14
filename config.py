import os
from dotenv import load_dotenv

load_dotenv()

# --- NSL-KDD Feature Schema ---
# 41 traffic-flow features used for classification
FEATURE_NAMES = [
    # Basic features
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent",
    # Content features
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    # Time-based traffic features (same host, last 2 sec)
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # Host-based traffic features (same dest host, last 100 connections)
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]

# Attack categories (NSL-KDD multi-class)
ATTACK_CATEGORIES = {
    "normal": 0,
    "dos": 1,      # Denial of Service
    "probe": 2,    # Surveillance/probing
    "r2l": 3,      # Remote-to-Local
    "u2r": 4,      # User-to-Root
}
IDX_TO_CATEGORY = {v: k for k, v in ATTACK_CATEGORIES.items()}

# Specific attack types → category mapping
ATTACK_TYPE_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "xterm": "u2r", "ps": "u2r", "sqlattack": "u2r",
    "worm": "u2r",
}

# --- Model ---
MODEL_INPUT_DIM = 122       # after one-hot encoding of categorical features
MODEL_HIDDEN_DIMS = [256, 128, 64]
MODEL_NUM_CLASSES = 5
MODEL_DROPOUT = 0.3
MODEL_PATH = os.getenv("MODEL_PATH", "model/saint_classifier.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "model/encoder.pkl")

# --- Redis ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
INFERENCE_CACHE_TTL = 300   # seconds

# --- Flask ---
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5001))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# --- Dashboard ---
DASH_PORT = int(os.getenv("DASH_PORT", 8050))
DASHBOARD_REFRESH_INTERVAL_MS = 2000   # poll every 2s

# --- Thresholds for agent reasoning ---
HIGH_CONFIDENCE_THRESHOLD = 0.85
FLAG_FOR_REVIEW_THRESHOLD = 0.60
