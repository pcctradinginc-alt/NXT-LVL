"""Email builder and sender (Gmail SMTP).

`build_email(result)` is pure (no I/O) so it can be unit-tested and reused
for the --dry-run HTML dump. `send()` performs the actual SMTP_SSL send.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

DISCLAIMER_HTML = """
Keine Anlageberatung. Optionen können wertlos verfallen. Das System liefert
datengetriebene Signale mit gemessener, aber nicht garantierter Trefferquote.
Die Entscheidung über jeden Trade liegt ausschließlich beim Menschen.
"""

BASE_STYLE = """
  body { background:#0d1117; color:#c9d1d9; font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif; margin:0; padding:24px; }
  .container { max-width:640px; margin:0 auto; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; margin-bottom:16px; }
  h1 { font-size:20px; color:#58a6ff; margin:0 0 8px 0; }
  h2 { font-size:15px; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em; margin:0 0 12px 0; }
  .pick-ticker { font-size:28px; font-weight:700; color:#3fb950; }
  .pick-meta { color:#8b949e; font-size:13px; margin-top:4px; }
  .thesis { margin-top:12px; line-height:1.5; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid #30363d; }
  th { color:#8b949e; font-weight:600; }
  .stat-row { display:flex; justify-content:space-between; padding:4px 0; font-size:14px; }
  .stat-label { color:#8b949e; }
  .footer { color:#6e7681; font-size:11px; line-height:1.5; margin-top:8px; }
  .badge { display:inline-block; background:#1f6feb; color:#fff; border-radius:4px; padding:2px 8px; font-size:12px; margin-left:8px; }
  .no-signal { color:#8b949e; font-size:15px; line-height:1.6; }
"""


def _stage_name(stage_id: int | None, stages_config: list[dict[str, Any]]) -> str:
    if stage_id is None:
        return "unbekannt"
    for stage in stages_config:
        if stage.get("id") == stage_id:
            return f"Stufe {stage_id}: {stage.get('name')}"
    return f"Stufe {stage_id}"


def _render_stage_distribution(
    stage_distribution: dict[str, Any] | None,
    stages_config: list[dict[str, Any]],
    stage_reasoning: str,
    llm_current_stage: int | None,
    llm_next_stage: int | None,
) -> str:
    """Render the probabilistic stage block (#5): top ~4 stages by probability
    plus a confidence line and a one-line LLM cross-check.

    Returns "" when `stage_distribution` has no probabilities at all, so the
    caller can fall back to the old single "Aktuell -> Als Nächstes" line for
    result dicts that don't carry this key (older callers / tests).
    """
    if not stage_distribution:
        return ""
    probabilities = stage_distribution.get("probabilities") or {}
    if not probabilities:
        return ""

    def _prob_value(item: tuple[Any, Any]) -> float:
        value = item[1]
        return value if isinstance(value, (int, float)) else 0.0

    top_stages = sorted(probabilities.items(), key=_prob_value, reverse=True)[:4]

    rows = []
    for stage_id_raw, prob in top_stages:
        try:
            stage_id: Any = int(stage_id_raw)
        except (TypeError, ValueError):
            stage_id = stage_id_raw
        name = _stage_name(stage_id, stages_config)
        pct = prob * 100 if isinstance(prob, (int, float)) else 0.0
        rows.append(
            f'<div class="stat-row"><span class="stat-label">{name}</span>'
            f'<span>{pct:.0f}%</span></div>'
        )

    confidence = stage_distribution.get("confidence")
    confidence_str = f"{confidence * 100:.0f}%" if isinstance(confidence, (int, float)) else "n/a"
    rows.append(
        f'<div class="stat-row"><span class="stat-label">Konfidenz</span><span>{confidence_str}</span></div>'
    )

    cross_check = ""
    if stage_reasoning or llm_current_stage is not None or llm_next_stage is not None:
        llm_stage_str = (
            f"{_stage_name(llm_current_stage, stages_config)} &rarr; {_stage_name(llm_next_stage, stages_config)}. "
            if (llm_current_stage is not None or llm_next_stage is not None)
            else ""
        )
        cross_check = (
            f'<div class="pick-meta" style="margin-top:8px;">LLM-Einschätzung: {llm_stage_str}{stage_reasoning}</div>'
        )

    return f"""
    <div class="thesis">Stufen-Wahrscheinlichkeiten (deterministisch, code-berechnet):</div>
    {''.join(rows)}
    {cross_check}
    """


def _render_candidate_table(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for c in candidates[:5]:
        scores = c.get("scores", {})
        rows.append(
            "<tr>"
            f"<td>{c.get('ticker', '')}</td>"
            f"<td>{c.get('total_score', 0)}</td>"
            f"<td>{scores.get('breadth', '-')}</td>"
            f"<td>{scores.get('momentum', '-')}</td>"
            f"<td>{scores.get('stage_fit', '-')}</td>"
            f"<td>{scores.get('divergence', '-')}</td>"
            f"<td>{scores.get('option_quality', '-')}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<tr><th>Ticker</th><th>Score</th><th>Breite</th><th>Momentum</th>"
        "<th>Stufen-Fit</th><th>Divergenz</th><th>Optionen</th></tr>"
        f"{''.join(rows)}"
        "</table>"
    )


def _render_track_record(track_record: dict[str, Any]) -> str:
    hit_rate = track_record.get("hit_rate")
    hit_rate_str = f"{hit_rate:.1f}%" if isinstance(hit_rate, (int, float)) else "n/a"
    avg_pnl = track_record.get("avg_pnl_pct")
    avg_pnl_str = f"{avg_pnl:+.1f}%" if isinstance(avg_pnl, (int, float)) else "n/a"
    return f"""
    <div class="stat-row"><span class="stat-label">Rollierende Hit Rate</span><span>{hit_rate_str}</span></div>
    <div class="stat-row"><span class="stat-label">Geschlossene Signale</span><span>{track_record.get('closed', 0)}</span></div>
    <div class="stat-row"><span class="stat-label">Offene Signale</span><span>{track_record.get('open', 0)}</span></div>
    <div class="stat-row"><span class="stat-label">Ø P/L (geschlossen)</span><span>{avg_pnl_str}</span></div>
    """


def _fmt_num(value: Any, suffix: str = "", decimals: int = 2) -> str:
    """Format a numeric value for the email, or '-' when not present."""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}{suffix}"
    return "-"


def _render_structure(
    structure: dict[str, Any] | None,
    earnings_trap: bool = False,
    iv_percentile: float | None = None,
) -> str:
    """Render the recommended options structure block (German), or "" if absent."""
    if not structure:
        return ""

    kind = structure.get("structure")
    kind_label = {
        "long_call": "Long Call",
        "call_spread": "Call-Spread",
        "stock": "Aktie (statt Call)",
    }.get(kind, kind or "unbekannt")

    metrics = structure.get("metrics") or {}
    reason = structure.get("reason", "")
    iv_expensive = structure.get("iv_expensive")
    realized_vol = structure.get("realized_vol")

    rows: list[str] = []
    rows.append(
        f"<div class=\"stat-row\"><span class=\"stat-label\">Empfohlene Struktur</span>"
        f"<span>{kind_label}</span></div>"
    )
    if reason:
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">Begründung</span>"
            f"<span>{reason}</span></div>"
        )

    break_even = metrics.get("break_even")
    if break_even is not None:
        move_pct = metrics.get("break_even_move_pct")
        move_str = f" ({_fmt_num(move_pct, '%', 1)} Bewegung nötig)" if move_pct is not None else ""
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">Break-even</span>"
            f"<span>{_fmt_num(break_even, '', 2)}{move_str}</span></div>"
        )

    # IV expensive? — show IV vs. realized vol when both are available.
    iv_val = metrics.get("iv")
    iv_expensive_label = "ja" if iv_expensive else "nein"
    iv_detail = ""
    if isinstance(iv_val, (int, float)) and isinstance(realized_vol, (int, float)):
        iv_detail = f" (IV {_fmt_num(iv_val, '', 2)} vs. realisierte Vola {_fmt_num(realized_vol, '', 2)})"
    elif isinstance(realized_vol, (int, float)):
        iv_detail = f" (realisierte Vola {_fmt_num(realized_vol, '', 2)})"
    if iv_expensive is not None:
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">IV teuer?</span>"
            f"<span>{iv_expensive_label}{iv_detail}</span></div>"
        )

    # IV-rank forward (#6): the option's IV vs. this ticker's OWN accumulated
    # history, once enough prior observations exist.
    if isinstance(iv_percentile, (int, float)):
        n_hint = ""
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">IV-Perzentil (eigene Historie)</span>"
            f"<span>{_fmt_num(iv_percentile, '%', 0)}{n_hint}</span></div>"
        )

    theta_per_day = metrics.get("theta_per_day")
    if isinstance(theta_per_day, (int, float)):
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">Theta/Tag</span>"
            f"<span>{_fmt_num(theta_per_day, '', 3)}</span></div>"
        )

    max_loss = metrics.get("max_loss")
    if isinstance(max_loss, (int, float)):
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">Max-Verlust (1 Kontrakt)</span>"
            f"<span>${_fmt_num(max_loss, '', 2)}</span></div>"
        )

    max_profit = metrics.get("max_profit")
    if isinstance(max_profit, (int, float)):
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">Max-Gewinn (1 Kontrakt)</span>"
            f"<span>${_fmt_num(max_profit, '', 2)}</span></div>"
        )

    warning_html = ""
    if earnings_trap:
        warning_html = (
            "<div class=\"thesis\" style=\"margin-top:8px;color:#f0883e;\">"
            "&#9888; Earnings-Termin in der frühen Laufzeit der Option &mdash; erhöhtes "
            "Theta-/IV-Crush-Risiko rund um den Termin.</div>"
        )

    return f"""
    <div class="card">
      <h2>Options-Struktur</h2>
      {''.join(rows)}
      {warning_html}
    </div>
    """


def _render_discovery(discovery: dict[str, Any] | None) -> str:
    """Render the "Warum entdeckt" section, or "" if no discovery data is present."""
    if not discovery:
        return ""
    via = discovery.get("via", "unbekannt")
    theme = discovery.get("theme") or discovery.get("theme_id") or ""
    drivers = discovery.get("drivers") or {}
    confirming_sources = discovery.get("confirming_sources") or discovery.get("sources") or []

    via_label = {
        "emergence": "Emergence-Erkennung (neues/beschleunigendes Thema)",
        "watchlist": "7-Stufen-Watchlist",
        "llm": "LLM-Analyse",
    }.get(via, via)

    driver_items = "".join(
        f"<div class=\"stat-row\"><span class=\"stat-label\">{k}</span><span>{v}</span></div>"
        for k, v in drivers.items()
    )
    sources_str = ", ".join(str(s) for s in confirming_sources) if confirming_sources else "n/a"

    return f"""
    <div class="card">
      <h2>Warum wurde der Kandidat entdeckt?</h2>
      <div class="stat-row"><span class="stat-label">Herkunft</span><span>{via_label}</span></div>
      <div class="stat-row"><span class="stat-label">Thema</span><span>{theme}</span></div>
      {driver_items}
      <div class="stat-row"><span class="stat-label">Bestätigende Quellen</span><span>{sources_str}</span></div>
    </div>
    """


def _render_invalidation(invalidation: dict[str, Any] | None) -> str:
    """Render the "These ungültig, wenn:" section (#14), or "" if absent."""
    if not invalidation:
        return ""
    below_50dma = invalidation.get("below_50dma")
    below_str = (
        f"Unterhalb {_fmt_num(below_50dma, '', 2)} (50-Tage-Durchschnitt)"
        if below_50dma is not None
        else "50-Tage-Durchschnitt nicht verfügbar"
    )
    theme_score_drop = invalidation.get("theme_score_drop")
    note = invalidation.get("note", "")

    rows = [
        f"<div class=\"stat-row\"><span class=\"stat-label\">Kurs</span><span>{below_str}</span></div>"
    ]
    if theme_score_drop:
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-label\">Thema</span><span>{theme_score_drop}</span></div>"
        )
    note_html = f"<div class=\"thesis\" style=\"margin-top:8px;\">{note}</div>" if note else ""

    return f"""
    <div class="card">
      <h2>These ungültig, wenn:</h2>
      {''.join(rows)}
      {note_html}
    </div>
    """


def _render_risks(risks: list[str] | None) -> str:
    """Render the "Risiken" section, or "" if no risks were flagged/present."""
    if not risks:
        return ""
    items = "".join(f"<li>{r}</li>" for r in risks)
    return f"""
    <div class="card">
      <h2>Risiken</h2>
      <ul>{items}</ul>
    </div>
    """


def _render_emergent_themes(emergent_themes: list[dict[str, Any]] | None) -> str:
    """Render the top-3 emergent themes section, or "" if none are present."""
    if not emergent_themes:
        return ""
    rows = []
    for theme in emergent_themes[:3]:
        drivers = theme.get("drivers", {})
        rows.append(
            "<tr>"
            f"<td>{theme.get('name', theme.get('theme_id', ''))}</td>"
            f"<td>{theme.get('emergence_score', '-')}</td>"
            f"<td>{theme.get('acceleration_z', drivers.get('acceleration_ratio', '-'))}</td>"
            f"<td>{theme.get('source_diversity', drivers.get('diversity', '-'))}</td>"
            "</tr>"
        )
    table = (
        "<table>"
        "<tr><th>Thema</th><th>Score</th><th>Beschleunigung</th><th>Diversität</th></tr>"
        f"{''.join(rows)}"
        "</table>"
    )
    return f"""
    <div class="card">
      <h2>Emergence-Themen</h2>
      {table}
    </div>
    """


def _render_filing_language(edgar_language: dict[str, Any] | None) -> str:
    """Render the compact "Filing-Sprache (Beschleunigung)" block (#8), or "" if absent.

    Shows the top ~4 phrases by positive aggregate delta (latest filing vs.
    prior filing, summed across companies) — i.e. which AI-buildout phrases
    are accelerating the most right now. Skipped entirely when there's no
    accelerating phrase to show.
    """
    if not edgar_language:
        return ""
    aggregate = edgar_language.get("aggregate") or {}
    positive = [
        (phrase, delta) for phrase, delta in aggregate.items() if isinstance(delta, (int, float)) and delta > 0
    ]
    if not positive:
        return ""
    positive.sort(key=lambda item: item[1], reverse=True)
    items = "".join(f"<li>&bdquo;{phrase}&ldquo; {delta:+.0f} ggü. Vorquartal</li>" for phrase, delta in positive[:4])
    return f"""
    <div class="card">
      <h2>Filing-Sprache (Beschleunigung)</h2>
      <ul>{items}</ul>
    </div>
    """


def _render_reward_status(reward: dict[str, Any] | None) -> str:
    """Render the reward-engine status section, or "" if not present."""
    if not reward:
        return ""
    feature_weights = reward.get("feature_weights") or {}
    history = reward.get("history") or []
    source_reliability = reward.get("source_reliability") or {}
    hit_rate_by_horizon = reward.get("hit_rate_by_horizon") or {}

    weight_rows = "".join(
        f"<div class=\"stat-row\"><span class=\"stat-label\">{k}</span><span>{v}</span></div>"
        for k, v in feature_weights.items()
    )
    reliability_rows = "".join(
        f"<div class=\"stat-row\"><span class=\"stat-label\">{k}</span><span>{v}</span></div>"
        for k, v in source_reliability.items()
    )
    hit_rate_rows = "".join(
        f"<div class=\"stat-row\"><span class=\"stat-label\">Horizont {k} Tage</span><span>{v}</span></div>"
        for k, v in hit_rate_by_horizon.items()
    )
    last_changes = history[-3:] if history else []
    history_items = "".join(
        f"<li>{h.get('date', '')}: {h.get('target', '')} {h.get('old', '')} &rarr; "
        f"{h.get('new', '')} ({h.get('reason', '')})</li>"
        for h in last_changes
    )
    history_html = f"<ul>{history_items}</ul>" if history_items else "<p>Noch keine protokollierten Änderungen.</p>"

    return f"""
    <div class="card">
      <h2>Reward-Status</h2>
      <div class="thesis">Aktuelle Feature-Gewichte:</div>
      {weight_rows}
      <div class="thesis" style="margin-top:8px;">Quellen-Zuverlässigkeit:</div>
      {reliability_rows}
      <div class="thesis" style="margin-top:8px;">Hit Rate je Horizont:</div>
      {hit_rate_rows}
      <div class="thesis" style="margin-top:8px;">Letzte Änderungen:</div>
      {history_html}
    </div>
    """


def build_email(result: dict[str, Any]) -> tuple[str, str]:
    """Build (subject, html_body) from a pipeline result dict.

    Expected keys in `result`:
      stages_config: list of stage configs (for names)
      current_stage, next_stage: int | None
      stage_reasoning: str
      top_pick: dict | None (candidate with `option` sub-dict) or None
      top5: list[dict] (scored candidates, may be empty)
      track_record: dict (from tracking.stats())
    """
    stages_config = result.get("stages_config", [])
    current_stage = result.get("current_stage")
    next_stage = result.get("next_stage")
    stage_reasoning = result.get("stage_reasoning", "")
    top_pick = result.get("top_pick")
    top5 = result.get("top5", [])
    track_record = result.get("track_record", {})

    # Probabilistic stage block (#5): prefer the code-computed distribution
    # over the old single "Aktuell -> Als Nächstes" line whenever it's
    # present; fall back to the old line for result dicts that lack it
    # (older callers / some existing tests), so those keep working unchanged.
    stage_block_html = _render_stage_distribution(
        result.get("stage_distribution"),
        stages_config,
        stage_reasoning,
        result.get("llm_current_stage"),
        result.get("llm_next_stage"),
    )
    if stage_block_html:
        stage_line = stage_block_html
    else:
        stage_line = (
            f'<div class="thesis">Aktuell: {_stage_name(current_stage, stages_config)} &rarr; '
            f'Als Nächstes: {_stage_name(next_stage, stages_config)}. {stage_reasoning}</div>'
        )

    if top_pick:
        ticker = top_pick.get("ticker", "?")
        option = top_pick.get("option") or {}
        strike = option.get("strike")
        expiration = option.get("expiration")
        mid = option.get("mid")
        dte = option.get("dte")
        delta = option.get("delta")
        thesis = top_pick.get("thesis", "")
        benchmark_symbol = top_pick.get("benchmark_symbol")

        if strike is not None and expiration:
            subject = f"NXT LVL Signal: {ticker} {strike}C {expiration}"
            option_line = (
                f"<div class=\"pick-meta\">Call {strike} &middot; Verfall {expiration} &middot; "
                f"DTE {dte} &middot; Delta {delta} &middot; Mid ${mid}</div>"
            )
        else:
            subject = f"NXT LVL Signal: {ticker} (nur Aktie, keine passende Option)"
            option_line = (
                "<div class=\"pick-meta\">Keine Option erfüllte die Liquiditäts-/Delta-Kriterien "
                "&mdash; Signal nur auf Aktienebene.</div>"
            )

        benchmark_line = (
            f"<div class=\"pick-meta\">Gemessen gegen Sektor-Benchmark: {benchmark_symbol}</div>"
            if benchmark_symbol
            else ""
        )

        # Claims-verification line (#18): "Belege geprüft: k/n" — how many of
        # the candidate's machine-checkable claims actually verified against
        # the digest. Shown whenever claims_verified is present on the
        # candidate (main.py sets it for every scored candidate).
        claims_verified = top_pick.get("claims_verified")
        claims_line = ""
        if isinstance(claims_verified, dict):
            claims_line = (
                f"<div class=\"pick-meta\">Belege geprüft: "
                f"{claims_verified.get('verified', 0)}/{claims_verified.get('total', 0)}</div>"
            )

        top_pick_html = f"""
        <div class="card">
          <h2>Top-Pick</h2>
          <div class="pick-ticker">{ticker}<span class="badge">Score {top_pick.get('total_score', 0)}</span></div>
          {option_line}
          {benchmark_line}
          {claims_line}
          <div class="thesis">{thesis}</div>
        </div>
        """
    else:
        subject = "NXT LVL: Kein Signal heute"
        no_signal_reason = result.get("no_signal_reason")
        no_signal_text = no_signal_reason or (
            "Kein Kandidat hat heute den Signal-Schwellwert (Score &ge; 70, "
            "mindestens 2 unabhängige Quellen) erreicht oder war im Cooldown. Kein Signal ist ein "
            "gültiges Ergebnis &mdash; Qualität vor Quantität."
        )
        top_pick_html = f"""
        <div class="card">
          <h2>Top-Pick</h2>
          <p class="no-signal">{no_signal_text}</p>
        </div>
        """

    top5_html = ""
    if top5:
        top5_html = f"""
        <div class="card">
          <h2>Top-5-Kandidaten</h2>
          {_render_candidate_table(top5)}
        </div>
        """

    # Optional Emergence & Reward Engine sections — all skipped gracefully
    # when the corresponding data is not present in `result` so older
    # result dicts (without these keys) render exactly as before.
    discovery_html = _render_discovery((top_pick or {}).get("discovery")) if top_pick else ""
    structure_html = (
        _render_structure(
            (top_pick or {}).get("structure"),
            bool((top_pick or {}).get("earnings_trap")),
            (top_pick or {}).get("iv_percentile"),
        )
        if top_pick
        else ""
    )
    risks_html = _render_risks((top_pick or {}).get("risks")) if top_pick else ""
    invalidation_html = _render_invalidation((top_pick or {}).get("invalidation")) if top_pick else ""
    emergent_themes_html = _render_emergent_themes(result.get("emergent_themes"))
    filing_language_html = _render_filing_language(result.get("edgar_language"))
    reward_html = _render_reward_status(result.get("reward"))

    html = f"""<!DOCTYPE html>
<html lang="de">
<head><meta charset="utf-8"><style>{BASE_STYLE}</style></head>
<body>
  <div class="container">
    <div class="card">
      <h1>NXT LVL &mdash; AI Next-Stage Beneficiary Scanner</h1>
      {stage_line}
    </div>
    {top_pick_html}
    {structure_html}
    {invalidation_html}
    {discovery_html}
    {risks_html}
    {top5_html}
    {emergent_themes_html}
    {filing_language_html}
    <div class="card">
      <h2>Track Record</h2>
      {_render_track_record(track_record)}
    </div>
    {reward_html}
    <div class="footer">{DISCLAIMER_HTML}</div>
  </div>
</body>
</html>"""

    return subject, html


def send(subject: str, html: str, mail_from: str, mail_to: str, app_password: str) -> None:
    """Send the HTML email via Gmail SMTP_SSL. Raises on failure (caller decides)."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.login(mail_from, app_password)
        server.sendmail(mail_from, [mail_to], msg.as_string())

    logger.info("mailer: email sent to %s", mail_to)
