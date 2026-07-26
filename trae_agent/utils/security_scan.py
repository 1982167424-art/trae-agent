"""Static security heuristics for MCP servers and skills.

Mirrors the desktop scanner. Best-effort, rule-based; not a substitute
for a real audit. Returns a risk level plus a list of findings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class Finding:
    level: str
    rule: str
    detail: str


# (level, rule, regex, detail)
_RULES: list[tuple[str, str, Any, str]] = [
    (
        "high",
        "destructive_fs",
        re.compile(r"\brm\s+-rf\b|\bformat\b|\bshutdown\b", re.I),
        "检测到可能删除/破坏文件系统的命令",
    ),
    (
        "high",
        "privilege_escalation",
        re.compile(r"\bsudo\b|\bsu\b|\bchmod\s+777\b", re.I),
        "检测到提权或放宽权限的操作",
    ),
    (
        "high",
        "remote_code_exec",
        re.compile(r"curl\s+[^\n|]*\|\s*(?:sh|bash)|wget\s+[^\n|]*\|\s*(?:sh|bash)|eval\s*\(\s*`", re.I),
        "检测到从网络下载并直接执行的管道命令（远程代码执行风险）",
    ),
    (
        "medium",
        "spawn_process",
        re.compile(r"child_process|subprocess|\bspawn\b|\bexec\b|\bexecFile\b", re.I),
        "会派生子进程执行系统命令",
    ),
    (
        "medium",
        "network_egress",
        re.compile(r"https?://|fetch\(|axios|urllib|requests\.(get|post)|http\.(get|post)", re.I),
        "存在对外网络请求，可能外泄数据",
    ),
    (
        "medium",
        "file_write",
        re.compile(r"fs\.(writeFile|appendFile)|writeFileSync|fs\.write", re.I),
        "会写入本地文件系统",
    ),
    (
        "low",
        "secret_access",
        re.compile(r"process\.env|os\.environ|API_KEY|SECRET|TOKEN", re.I),
        "读取环境变量或密钥，注意密钥泄露",
    ),
    (
        "low",
        "prompt_injection",
        re.compile(r"ignore (?:previous|above|all) (?:instructions|prompt)|disregard (?:previous|above)", re.I),
        "检测到提示注入特征文本",
    ),
]

_SCORE = {"high": 3, "medium": 2, "low": 1, "none": 0}


def scan_text(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for level, rule, pat, detail in _RULES:
        if pat.search(text):
            findings.append(Finding(level=level, rule=rule, detail=detail))
    return findings


def summarize(findings: list[Finding]) -> RiskLevel:
    score = max((_SCORE.get(f.level, 0) for f in findings), default=0)
    if score >= 3:
        return RiskLevel.HIGH
    if score == 2:
        return RiskLevel.MEDIUM
    if score == 1:
        return RiskLevel.LOW
    return RiskLevel.NONE


def scan_mcp_server(server: dict[str, Any]) -> tuple[RiskLevel, list[Finding]]:
    cmd = " ".join([str(server.get("command", ""))] + [str(a) for a in server.get("args", [])])
    env = server.get("env") or {}
    text = f"CMD: {cmd}\nENV: {env!r}"
    findings = scan_text(text)
    return summarize(findings), findings


def scan_plugin_source(source: str) -> tuple[RiskLevel, list[Finding]]:
    findings = scan_text(source)
    return summarize(findings), findings
