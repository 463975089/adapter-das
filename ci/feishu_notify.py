#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""把 HCU nightly 的判定结果作为飞书交互卡片发到群里。

由 .github/workflows/hcu-nightly-test.yml 的 notify job 调用，输入全部走环境变量：

  FEISHU_WEBHOOK      自定义机器人 webhook URL；为空时只把卡片 JSON 打到日志并
                      以 0 退出（fork / 未配置密钥的仓库不会因此报红）
  FEISHU_SIGN_SECRET  机器人开启"签名校验"时的密钥（可选；填了就带 timestamp+sign）
  NEEDS_JSON          ${{ toJSON(needs) }}：各 job 的 result，以及逐项判定
                      outputs（l15/vllm-lane 的 summary、l2 的 models）
  L2_MODELS           L2 模型 id，空格分隔（只用于展示顺序）
  RUN_URL / RUN_ID / REF_NAME / EVENT_NAME

卡片结构（Feishu 消息卡片 v1，msg_type=interactive）：
  标题栏（颜色随总体结论）+ 三个 job 的状态字段 + L2 各模型四道门 + 运行链接。
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.request

JOBS = [
    ("l15", "L1.5 上游门禁"),
    ("vllm-lane", "L1 vLLM 宿主车道"),
    ("l2", "L2 端到端模型"),
]
ICON = {"success": "✅", "failure": "❌", "cancelled": "⚪", "skipped": "⏭️"}
GATES = [
    ("G1-launch", "启动"),
    ("G2-generation", "生成"),
    ("G3-precision", "精度"),
    ("G4-long", "长文"),
]
TRIGGER = {"schedule": "定时任务", "workflow_dispatch": "手动触发"}

# 卡片里全是 emoji/中文；Windows(GBK) 下本地试跑会 UnicodeEncodeError，
# 统一把标准输出改成 UTF-8（CI 的 ubuntu-latest 本来就是 UTF-8，这里不影响）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def env(name):
    return (os.environ.get(name) or "").strip()


def load_needs():
    try:
        return json.loads(env("NEEDS_JSON") or "{}")
    except json.JSONDecodeError:
        return {}


def job_results(needs):
    return {jid: needs.get(jid, {}).get("result", "skipped") for jid, _ in JOBS}


def overall(results):
    states = set(results.values())
    if states == {"success"}:
        return "全部通过", "green"
    if "failure" in states or "cancelled" in states:
        return "存在失败", "red"
    return "未完整执行", "grey"


def l2_markdown(needs):
    raw = needs.get("l2", {}).get("outputs", {}).get("models") or ""
    try:
        models = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        models = {}
    if not models:
        return "**L2 判定**\n未取到 result 文件（suite 未执行或该 job 被取消）"

    order = env("L2_MODELS").split() or list(models)
    lines = ["**L2 判定**"]
    for mid in order:
        gates = models.get(mid) or {}
        if not gates:
            lines.append(f"**{mid}** — 未执行（无 result 文件）")
            continue
        parts = []
        for key, label in GATES:
            state = gates.get(key)
            if state == "PASS":
                parts.append(f"{label} ✅")
            elif state == "FAIL":
                parts.append(f"{label} ❌")
            else:
                # 文件被 judge_model 截断或 G1 之后就中断时会缺门
                parts.append(f"{label} ⏭️")
        lines.append(f"**{mid}**  " + "  ".join(parts))
    return "\n".join(lines)


def job_items(needs, jid):
    """读出某个 job 的逐项判定；没有该 output 时返回 None（区别于"有 output 但为空"）。"""
    raw = (needs.get(jid, {}).get("outputs") or {}).get("summary") or ""
    if not raw:
        return None
    try:
        return json.loads(raw).get("items") or {}
    except json.JSONDecodeError:
        return None


def details_markdown(needs, results):
    """L1.5 / vLLM 车道的逐项明细：全绿只报数量，有红就把失败项列出来。

    明细只覆盖脚本打印了判定行的项；job 还会因为装包失败、日志上传失败、
    超时/被取消等"判定项之外"的原因变红。所以最后那句结论必须再和 job
    result 对一遍，否则卡片上半截 ❌、下半截"全部通过"，自相矛盾
    （run 45573797349 就是这个形态）。
    """
    lines = []
    for jid, label in JOBS:
        if jid == "l2":
            continue
        result = results.get(jid, "skipped")
        items = job_items(needs, jid)
        if items is None:
            continue
        if not items:
            lines.append(
                f"**{label}** — 未产生明细（job 结果 {result}；脚本没跑到判定）"
            )
            continue
        # vLLM 车道的 EMPTY（import-only 模块、rc=5）不算失败，单独计数
        fails = [name for name, state in items.items() if state == "FAIL"]
        empty = [name for name, state in items.items() if state not in ("PASS", "FAIL")]
        total = len(items)
        passed = total - len(fails) - len(empty)
        if fails:
            shown = "、".join(fails[:6])
            if len(fails) > 6:
                shown += f" 等 {len(fails)} 项"
            line = f"**{label}** — {passed}/{total} 通过；失败：{shown}"
            if empty:
                line += f"（另有 {len(empty)} 项无用例）"
            lines.append(line)
            continue
        if empty:
            line = f"**{label}** — {passed}/{total} 通过（另有 {len(empty)} 项无用例）"
        else:
            line = f"**{label}** — 全部通过（{total} 项）"
        if result == "cancelled":
            line += "；但 job 被取消（超时或上游失败），明细可能不完整"
        elif result != "success":
            line += (
                f"；但 job 结果为 {result}——判定项之外还有失败"
                "（装包/收集/上传/超时），见日志"
            )
        lines.append(line)
    return "\n".join(lines)


def build_card(needs, results, status_text, template):
    fields = [
        {
            "is_short": True,
            "text": {
                "tag": "lark_md",
                "content": f"**{label}**\n{ICON.get(results[jid], '❔')} {results[jid]}",
            },
        }
        for jid, label in JOBS
    ]
    ref = env("REF_NAME") or "-"
    trigger = TRIGGER.get(env("EVENT_NAME"), env("EVENT_NAME") or "未知触发")
    run_url = env("RUN_URL")
    elements = [
        {"tag": "div", "fields": fields},
        {"tag": "hr"},
    ]
    details = details_markdown(needs, results)
    if details:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": details}})
        elements.append({"tag": "hr"})
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": l2_markdown(needs)}})
    if run_url:
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"🔗 [点击查看本次 CI 运行]({run_url})",
                },
            }
        )
    elements.append(
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"{trigger} · {ref} · run {env('RUN_ID') or '-'}",
                }
            ],
        }
    )
    if run_url:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "查看本次 CI"},
                        "type": "primary",
                        "url": run_url,
                    }
                ],
            }
        )
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": template,
                "title": {
                    "tag": "plain_text",
                    "content": f"HCU Nightly ｜ {ref} ｜ {status_text}",
                },
            },
            "elements": elements,
        },
    }


def main():
    needs = load_needs()
    results = job_results(needs)
    status_text, template = overall(results)
    payload = build_card(needs, results, status_text, template)

    webhook = env("FEISHU_WEBHOOK")
    if not webhook:
        print(
            "FEISHU_WEBHOOK 未配置，跳过发送；卡片内容如下：", file=sys.stderr
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    secret = env("FEISHU_SIGN_SECRET")
    if secret:
        timestamp = str(int(time.time()))
        digest = hmac.new(
            f"{timestamp}\n{secret}".encode("utf-8"), b"", hashlib.sha256
        ).digest()
        payload["timestamp"] = timestamp
        payload["sign"] = base64.b64encode(digest).decode()

    req = urllib.request.Request(
        webhook,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 -- 通知失败要给出原始原因
        print(f"飞书 webhook 请求失败：{exc}", file=sys.stderr)
        return 1

    print(f"飞书响应：{body}")
    try:
        code = json.loads(body).get("code", json.loads(body).get("StatusCode", 0))
    except json.JSONDecodeError:
        return 0
    if code not in (0, None):
        print("飞书返回非 0 code，通知未送达", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
