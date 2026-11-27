"""Defensive self-audit tools for your own websites and local codebases.

Strictly passive and non-intrusive:
- Website posture audit performs a single read-only HTTP GET to check standard
  security response headers (HSTS, CSP, X-Frame-Options, etc.), cookie flags,
  TLS certificate validity, and server banner exposure.
- Local repo audit performs static pattern checks for accidentally committed
  secrets (always masked in output) and insecure coding patterns.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import ssl
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

# ---------------------------------------------------------------------------
# Security header policy checks
# ---------------------------------------------------------------------------

_SECURITY_HEADERS: List[tuple[str, int, str, str]] = [
    (
        "strict-transport-security",
        20,
        "Missing HSTS header allows SSL-stripping downgrade attacks.",
        "Strict-Transport-Security: max-age=31536000; includeSubDomains; preload",
    ),
    (
        "content-security-policy",
        25,
        "Missing Content-Security-Policy increases XSS and data-injection exposure.",
        "Content-Security-Policy: default-src 'self'; object-src 'none'; frame-ancestors 'none'",
    ),
    (
        "x-content-type-options",
        15,
        "Missing X-Content-Type-Options lets browsers MIME-sniff responses.",
        "X-Content-Type-Options: nosniff",
    ),
    (
        "x-frame-options",
        15,
        "Missing X-Frame-Options (or CSP frame-ancestors) leaves pages open to clickjacking.",
        "X-Frame-Options: DENY",
    ),
    (
        "referrer-policy",
        10,
        "Missing Referrer-Policy can leak sensitive URL paths in Referer headers.",
        "Referrer-Policy: strict-origin-when-cross-origin",
    ),
    (
        "permissions-policy",
        10,
        "Missing Permissions-Policy leaves browser features (mic/cam/geo) unrestricted.",
        "Permissions-Policy: camera=(), microphone=(), geolocation=()",
    ),
    (
        "cross-origin-opener-policy",
        5,
        "Missing Cross-Origin-Opener-Policy weakens cross-origin window isolation.",
        "Cross-Origin-Opener-Policy: same-origin",
    ),
]


@dataclass
class AuditFinding:
    severity: str  # "high" | "medium" | "low" | "info"
    title: str
    detail: str
    remediation: str
    location: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SiteAuditReport:
    url: str
    final_url: str
    status_code: int
    score: int
    grade: str
    tls: Dict[str, Any]
    headers_present: Dict[str, str]
    findings: List[AuditFinding] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["findings"] = [f.to_dict() for f in self.findings]
        return data


def _score_to_grade(score: int) -> str:
    if score >= 95:
        return "A+"
    if score >= 85:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def _validate_target_url(raw_url: str, *, allow_loopback: bool = True) -> str:
    url = raw_url.strip()
    if not url:
        raise ValueError("URL must not be empty.")
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid HTTP/HTTPS URL: {raw_url!r}")

    hostname = parsed.hostname.lower()
    # Always block cloud metadata & link-local addresses (SSRF protection).
    try:
        addr_infos = socket.getaddrinfo(
            hostname, parsed.port or 443, type=socket.SOCK_STREAM
        )
    except socket.gaierror:
        return url  # Let httpx report the DNS error cleanly.

    for info in addr_infos:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise ValueError(
                f"Refused target IP {ip_str}: link-local/metadata addresses are blocked."
            )
        if ip.is_loopback and not allow_loopback:
            raise ValueError(f"Loopback target {hostname} is disabled.")
    return url


def _inspect_tls(hostname: str, port: int = 443) -> Dict[str, Any]:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=5.0) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                cipher = ssock.cipher()
                not_after = cert.get("notAfter", "") if cert else ""
                days_remaining = None
                if not_after:
                    try:
                        dt = datetime.strptime(
                            not_after, "%b %d %H:%M:%S %Y %Z"
                        ).replace(tzinfo=timezone.utc)
                        days_remaining = (dt - datetime.now(timezone.utc)).days
                    except ValueError:
                        pass
                issuer_tuples = cert.get("issuer", ()) if cert else ()
                issuer = ", ".join(f"{k}={v}" for rdn in issuer_tuples for k, v in rdn)
                return {
                    "enabled": True,
                    "protocol": ssock.version() or "unknown",
                    "cipher": cipher[0] if cipher else "unknown",
                    "not_after": not_after,
                    "days_remaining": days_remaining,
                    "issuer": issuer,
                }
    except Exception as exc:
        return {"enabled": False, "error": str(exc)}


def audit_website_posture(
    raw_url: str,
    *,
    allow_loopback: bool = True,
    timeout: float = 8.0,
) -> SiteAuditReport:
    """Run a single passive read-only security header & TLS posture check."""
    try:
        url = _validate_target_url(raw_url, allow_loopback=allow_loopback)
    except ValueError as exc:
        return SiteAuditReport(
            url=raw_url,
            final_url="",
            status_code=0,
            score=0,
            grade="F",
            tls={"enabled": False},
            headers_present={},
            error=str(exc),
        )

    findings: List[AuditFinding] = []
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "OpenJarvis-DefensiveAudit/1.0"},
        ) as client:
            resp = client.get(url)
    except Exception as exc:
        return SiteAuditReport(
            url=url,
            final_url="",
            status_code=0,
            score=0,
            grade="F",
            tls={"enabled": False},
            headers_present={},
            error=f"Could not reach {url}: {exc}",
        )

    final_url = str(resp.url)
    parsed_final = urlparse(final_url)
    lower_headers = {k.lower(): v for k, v in resp.headers.items()}

    # TLS check
    if parsed_final.scheme == "https" and parsed_final.hostname:
        tls_info = _inspect_tls(parsed_final.hostname, parsed_final.port or 443)
        if (
            tls_info.get("days_remaining") is not None
            and tls_info["days_remaining"] < 14
        ):
            findings.append(
                AuditFinding(
                    severity="high",
                    title="TLS certificate expiring soon",
                    detail=f"Certificate expires in {tls_info['days_remaining']} days ({tls_info.get('not_after')}).",
                    remediation="Renew TLS certificate and enable automated renewal (e.g. ACME / Let's Encrypt).",
                    location="TLS",
                )
            )
    else:
        tls_info = {"enabled": False, "note": "Target responded over plain HTTP."}
        findings.append(
            AuditFinding(
                severity="high",
                title="Plaintext HTTP endpoint",
                detail="Site did not redirect to HTTPS.",
                remediation="Serve over HTTPS and configure a permanent 301 redirect from HTTP to HTTPS.",
                location="Transport",
            )
        )

    score = 100
    headers_present: Dict[str, str] = {}
    for header_name, weight, detail, remediation in _SECURITY_HEADERS:
        val = lower_headers.get(header_name)
        if val:
            headers_present[header_name] = val
        else:
            # Frame protection can also be satisfied via CSP frame-ancestors
            if (
                header_name == "x-frame-options"
                and "frame-ancestors"
                in lower_headers.get("content-security-policy", "").lower()
            ):
                headers_present[header_name] = "(covered by CSP frame-ancestors)"
                continue
            score -= weight
            findings.append(
                AuditFinding(
                    severity="high"
                    if weight >= 20
                    else ("medium" if weight >= 10 else "low"),
                    title=f"Missing {header_name}",
                    detail=detail,
                    remediation=remediation,
                    location="HTTP Response Headers",
                )
            )

    # Check server banner exposure
    for leaky in ("server", "x-powered-by", "x-aspnet-version"):
        if leaky in lower_headers:
            headers_present[leaky] = lower_headers[leaky]
            findings.append(
                AuditFinding(
                    severity="info",
                    title=f"Technology banner exposed ({leaky})",
                    detail=f"Header `{leaky}: {lower_headers[leaky]}` discloses server stack details.",
                    remediation=f"Strip or generalize the `{leaky}` response header at your reverse proxy.",
                    location="HTTP Response Headers",
                )
            )

    # Check Set-Cookie hardening
    set_cookies = resp.headers.get_list("set-cookie")
    for idx, cookie in enumerate(set_cookies):
        low = cookie.lower()
        cookie_name = cookie.split("=", 1)[0].strip() or f"cookie#{idx + 1}"
        missing_flags = [
            flag for flag in ("httponly", "secure", "samesite") if flag not in low
        ]
        if missing_flags:
            score = max(0, score - 5)
            findings.append(
                AuditFinding(
                    severity="medium",
                    title=f"Cookie '{cookie_name}' missing hardening flags",
                    detail=f"Missing attributes: {', '.join(missing_flags)}.",
                    remediation="Set Secure; HttpOnly; SameSite=Lax (or Strict) on session cookies.",
                    location="Set-Cookie",
                )
            )

    if parsed_final.scheme != "https":
        score = max(0, score - 15)

    score = max(0, min(100, score))
    return SiteAuditReport(
        url=url,
        final_url=final_url,
        status_code=resp.status_code,
        score=score,
        grade=_score_to_grade(score),
        tls=tls_info,
        headers_present=headers_present,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Local codebase static defensive hygiene scanner
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    (
        "GitHub Personal Access Token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    ),
    ("AWS Access Key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "Private Key Block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
]

_CODE_HYGIENE_PATTERNS: List[tuple[str, str, re.Pattern[str], str]] = [
    (
        "medium",
        "Subprocess with shell=True",
        re.compile(r"\bsubprocess\.\w+\([^)]*shell\s*=\s*True"),
        "Pass an argument list with shell=False to prevent command injection.",
    ),
    (
        "medium",
        "TLS verification disabled (verify=False)",
        re.compile(r"\bverify\s*=\s*False\b"),
        "Remove verify=False so TLS certificates are strictly validated.",
    ),
    (
        "medium",
        "Unsafe yaml.load() without SafeLoader",
        re.compile(r"\byaml\.load\([^)]*\)"),
        "Use yaml.safe_load() to prevent arbitrary object deserialization.",
    ),
    (
        "low",
        "Dynamic eval()/exec() call",
        re.compile(r"(?<!\w)(?:eval|exec)\s*\("),
        "Replace eval/exec with ast.literal_eval or explicit dispatch where possible.",
    ),
]

_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "target",
    ".mypy_cache",
    ".ruff_cache",
}

_TEXT_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".env",
    ".sh",
    ".cfg",
    ".ini",
    ".md",
}


def audit_local_repo(
    raw_path: str,
    *,
    max_files: int = 250,
) -> Dict[str, Any]:
    """Scan a local project directory for leaked secrets and common security anti-patterns."""
    target = Path(raw_path).expanduser().resolve()
    if not target.exists():
        return {"error": f"Path does not exist: {raw_path}", "findings": []}
    if not target.is_dir():
        return {"error": f"Path is not a directory: {raw_path}", "findings": []}

    findings: List[AuditFinding] = []
    files_scanned = 0

    for root, dirs, filenames in os.walk(target):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in filenames:
            if files_scanned >= max_files:
                break
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(target))

            if fname == ".env":
                findings.append(
                    AuditFinding(
                        severity="medium",
                        title="Unencrypted .env file present in workspace",
                        detail=f"Found `{rel}` — make sure it is listed in `.gitignore` and never committed.",
                        remediation="Add `.env` to `.gitignore` and keep only `.env.example` tracked.",
                        location=rel,
                    )
                )

            if fpath.suffix.lower() not in _TEXT_EXTS and fname != ".env":
                continue
            try:
                if fpath.stat().st_size > 256_000:
                    continue
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            files_scanned += 1
            for lineno, line in enumerate(content.splitlines(), start=1):
                if len(findings) >= 60:
                    break
                for label, pattern in _SECRET_PATTERNS:
                    m = pattern.search(line)
                    if m:
                        token = m.group(0)
                        masked = (
                            f"{token[:6]}...{token[-3:]}" if len(token) > 12 else "***"
                        )
                        findings.append(
                            AuditFinding(
                                severity="high",
                                title=f"Potential hardcoded {label}",
                                detail=f"Matched pattern ({masked}) in `{rel}:{lineno}`.",
                                remediation="Rotate the credential if real and load it via environment variables.",
                                location=f"{rel}:{lineno}",
                            )
                        )
                if fpath.suffix.lower() == ".py":
                    for sev, title, pattern, remediation in _CODE_HYGIENE_PATTERNS:
                        if (
                            title.startswith("Unsafe yaml.load")
                            and "SafeLoader" in line
                        ):
                            continue
                        if pattern.search(line):
                            findings.append(
                                AuditFinding(
                                    severity=sev,
                                    title=title,
                                    detail=f"Pattern detected in `{rel}:{lineno}`.",
                                    remediation=remediation,
                                    location=f"{rel}:{lineno}",
                                )
                            )
        if files_scanned >= max_files:
            break

    high_count = sum(1 for f in findings if f.severity == "high")
    med_count = sum(1 for f in findings if f.severity == "medium")
    risk = "HIGH" if high_count > 0 else ("MEDIUM" if med_count > 0 else "LOW")
    return {
        "path": str(target),
        "files_scanned": files_scanned,
        "truncated": files_scanned >= max_files,
        "risk_level": risk,
        "summary": {
            "high": high_count,
            "medium": med_count,
            "low": sum(1 for f in findings if f.severity == "low"),
            "total": len(findings),
        },
        "findings": [f.to_dict() for f in findings],
    }


__all__ = [
    "AuditFinding",
    "SiteAuditReport",
    "audit_local_repo",
    "audit_website_posture",
]
