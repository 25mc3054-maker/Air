# ATMOSAIR v2 — Security Policy & Credential Management Protocol
**Smart India Hackathon 2026 | Problem Statement 26082**

---

## 1. Credential Protection Principles
1. **Zero Secret Hardcoding:** API keys for Copernicus CDS, CAMS, NASA FIRMS, and Sentinel-5P are strictly managed through environment variables or secret managers (e.g. AWS Secrets Manager, HashiCorp Vault).
2. **Repository Sanitization:** The `.env` file is explicitly ignored in `.gitignore`. Only `.env.example` containing redacted placeholders is committed to version control.
3. **Log Masking:** Application loggers and exception handlers sanitize tokens, API keys, and authorization headers to ensure secrets never appear in execution logs.

---

## 2. API Key Rotation Protocol
If any secret key (such as `FIRMS_MAP_KEY` or `SENTINEL_API_CLIENT_SECRET`) has ever been committed, printed in console output, or exposed in an unencrypted environment:

1. **Immediate Revocation:**
   - Log into NASA EOSDIS FIRMS account and regenerate the MAP_KEY.
   - Access Copernicus Data Space Ecosystem portal, delete the compromised client credential, and generate a new pair.
2. **Environment Update:**
   - Update `.env` with the newly provisioned keys.
   - Restart the running service daemon (`sudo systemctl restart atmosair`).
3. **Git History Scrubbing:**
   - If a key was committed in git history, use `git filter-repo` or BFG Repo-Cleaner to scrub the credential from past commits.

---

## 3. Data Leakage & Integrity Auditing
* All data ingested through external adapters undergo strict temporal validation (`timestamp <= issue_time`).
* Automated tests (`tests/test_leakage.py`) run as mandatory CI/CD gates prior to deployment.
