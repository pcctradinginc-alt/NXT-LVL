# NXT LVL — AI Next-Stage Beneficiary Scanner

Ein vollautomatisches, kostenfreies System, das den aktuellen Stand des KI-Ausbaus anhand frei
verfügbarer Frühindikatoren verfolgt und **nicht die aktuellen Gewinner**, sondern die
**Profiteure der nächsten Stufe** (3–12 Monate Horizont) frühzeitig identifiziert. Für den
Top-Kandidaten wird via Tradier API eine passende Option ausgewählt und per E-Mail mit kurzer
Begründung versendet.

Die vollständige, verbindliche Spezifikation steht in [`CONCEPT.md`](./CONCEPT.md) — dieses
README beschreibt nur Setup und Betrieb.

**Kein Trade wird automatisch ausgeführt.** Das System liefert Signale, die Entscheidung bleibt
beim Menschen. Siehe [Disclaimer](#disclaimer).

## Architektur

```
                    ┌─────────────────────────────────────────────┐
                    │        GitHub Actions (werktäglich,          │
                    │        11:30 UTC, vor US-Börsenöffnung)      │
                    └───────────────────┬───────────────────────────┘
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  1. COLLECT (5 kostenlose Quellen, fehlertolerant, sequenziell)     │
   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
   │  │  EDGAR   │ │  GitHub  │ │ HN Jobs  │ │  arXiv   │ │ HN Buzz  │  │
   │  │  Capex   │ │  Trends  │ │(hiring)  │ │  Trends  │ │(stories) │  │
   │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
   │       └────────────┴────────────┴────────────┴────────────┘       │
   │                              ▼                                     │
   │                   2. AGGREGATE (lokal, kompaktes Digest)            │
   └──────────────────────────────┬───────────────────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  3. LLM-ANALYSE (Claude Haiku   │
                    │     4.5, 1 Call/Tag)            │
                    │     -> current_stage,           │
                    │        next_stage, Kandidaten   │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  4. SCORING (deterministisch,   │
                    │     src/analysis/scoring.py)    │
                    │     breadth/momentum/stage_fit/ │
                    │     divergence/option_quality    │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  5. OPTIONSAUSWAHL (Tradier)    │
                    │     DTE 90-180, Delta 0.60-0.70 │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  6. E-MAIL (Gmail SMTP)         │
                    │     Top-Pick + Score-Tabelle +  │
                    │     Track Record                │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  7. TRACK RECORD UPDATE          │
                    │     data/signals.json committed  │
                    │     zurück ins Repo              │
                    └───────────────────────────────┘
```

## Das Stufenmodell

| # | Stufe | Beispiel-Profiteure |
|---|-------|---------------------|
| 1 | Compute / Halbleiter (Training) | NVDA, AMD, AVGO, TSM |
| 2 | Datacenter-Infrastruktur | VRT, DELL, ANET, MU, SMCI |
| 3 | Energie / Kühlung / Netz | VST, CEG, GEV, ETN, PWR, MOD |
| 4 | Inference / Netzwerk / Edge | MRVL, CRDO, ALAB, COHR |
| 5 | Software / Agenten / Daten | PLTR, NOW, DDOG, MDB, SNOW |
| 6 | Vertikale KI-Adoption | ISRG, CRWD, PANW, TER |
| 7 | Robotik / Physical AI | SYM, ROK, TER, TSLA |

Details zu Datenquellen, Scoring-Formel und Optionsauswahl: siehe [`CONCEPT.md`](./CONCEPT.md).

## Setup

### 1. Repository nutzen

Forke dieses Repository oder nutze es direkt in deinem eigenen GitHub-Account (öffentliches
Repo empfohlen, damit GitHub Actions kostenlos läuft).

### 2. GitHub Secrets anlegen

Gehe im Repo zu **Settings → Secrets and variables → Actions → New repository secret** und lege
folgende Secrets an:

| Secret | Zweck | Woher? |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM-Analyse (Claude Haiku 4.5) | [console.anthropic.com](https://console.anthropic.com) → "API Keys". Pay-as-you-go, aber sehr günstig (~1 Call/Tag ≈ wenige Cent/Monat). |
| `TRADIER_API_KEY` | Optionsdaten & Kurse | [tradier.com](https://tradier.com) → Konto → API-Zugang → Access Token erzeugen (Brokerage- oder Sandbox-Account) |
| `GMAIL_APP_PASSWORD` | Mail-Versand | Google-Konto → Sicherheit → 2-Faktor-Auth aktivieren → "App-Passwörter" → neues Passwort für "Mail" erzeugen |
| `MAIL_FROM` | Absenderadresse | Deine Gmail-Adresse (z.B. `pcctradinginc@gmail.com`) |
| `MAIL_TO` | Empfängeradresse | In der Regel dieselbe Gmail-Adresse |

Optional als **Repository Variable** (nicht Secret, da kein Geheimnis): `TRADIER_ENV=sandbox`,
falls du zunächst gegen die Tradier-Sandbox statt gegen den Live-Account testen willst (Default:
`prod`).

`GITHUB_TOKEN` wird von GitHub Actions automatisch bereitgestellt — hier ist nichts zu tun.

### 3. Lokal testen

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Läuft komplett ohne Secrets: Collector greifen auf echte kostenlose APIs zu,
# LLM/Tradier/Mail werden gestubbt bzw. übersprungen.
python -m src.main --dry-run
```

Ergebnis: `data/last_digest.json` (aggregierte Rohsignale) und `data/last_email.html`
(die Mail, die im Live-Betrieb verschickt würde) werden geschrieben. Öffne
`data/last_email.html` im Browser, um das Ergebnis anzusehen.

Für einen komplett netzwerkfreien Test (z.B. in restriktiven Sandboxes):

```bash
NXT_OFFLINE=1 python -m src.main --dry-run
```

### 4. Tests ausführen

```bash
pip install pytest
pytest tests/ -q
```

### 5. Workflow manuell auslösen

Im Repo unter **Actions → Daily AI Scan → Run workflow**. Der Workflow läuft danach automatisch
werktäglich um 11:30 UTC (kurz vor US-Börsenöffnung).

## Konfiguration

Alle strategischen Parameter — Stufen, Ticker-Universum, Keywords, Scoring-Gewichte,
Schwellwerte, Options-Filter, Tracking-Regeln — stehen in [`config.yaml`](./config.yaml) und
lassen sich ohne Code-Änderung anpassen. Wichtige Stellschrauben:

- `scoring.signal_threshold` (Standard 60): Mindest-Score für ein Signal.
- `scoring.min_sources` (Standard 2): Mindestanzahl unabhängiger Quellen.
- `scoring.cooldown_days` (Standard 14): Sperrfrist, bevor derselbe Ticker erneut signalisiert
  werden darf.
- `options.delta_min` / `delta_max` (Standard 0.60–0.70): Ziel-Delta für die Call-Auswahl.

## Validierung & Backtest

**Ehrliche Ausgangslage:** Die Scoring-Gewichte (`config.yaml` → `scoring.weights`) wurden nie
gegen realisierte Renditen validiert. Ein echter historischer Backtest der *vollständigen*
Signal-Logik (Breadth, Momentum, Emergence, ...) ist rückwirkend **nicht möglich** — die
kostenlosen Datenquellen (GitHub/EDGAR/HN/arXiv) liefern nur den aktuellen Stand ("point-in-time")
und wurden nie archiviert. Es gibt also kein "Was hätte der Scanner am 1.1.2024 gesagt"-Replay.
Statt das zu verschweigen, liefert dieses Projekt drei ehrliche Bausteine:

### A) Mechanischer Preis-Backtest (`src/backtest/price_backtest.py`)

Testet auf **echten historischen Kursdaten** die zwei mechanischen Hypothesen, die das Scoring
kodiert: (1) schlagen Divergenz-Buckets tatsächlich durch — sind Aktien, die seitwärts/leicht
gelaufen sind, bessere Call-Kandidaten als solche, die schon stark gestiegen sind? (2) ist ein
leicht-im-Geld-Call (Delta ≈ 0,60) strukturell sinnvoll? Die Optionswerte sind dabei eine
theoretische Black-Scholes-Neubewertung mit realisierter Volatilität als IV-Proxy (Tradier liefert
keine historischen Optionsketten) — eine belastbare Näherung, kein echter historischer Fill.

```bash
# Komplett offline, deterministische synthetische Kursreihen, kein Netzwerk nötig:
python -m src.backtest --mock

# Live gegen echte Tradier-Historie (braucht TRADIER_API_KEY):
python -m src.backtest --years 3 --horizon 90 --entry-dte 120 --delta 0.60
```

Ergebnis wird als Tabelle ausgegeben und nach `data/backtest_report.json` geschrieben.
**Was das NICHT testet:** die Collector-Signale (Breadth, Momentum, Emergence) — dafür fehlt die
historische Archivierung der Rohdaten komplett.

### B) Digest-Archivierung (`data/digest_history.jsonl`)

Jeder Lauf (auch `--dry-run`) hängt eine kompakte JSON-Zeile an `data/digest_history.jsonl` an:
Datum, Stufen-Einschätzung, Top-Pick, alle gescorten Kandidaten mit Einzel-Scores, Theme-Scores.
Diese Datei ist **nicht** in `.gitignore` (nur `last_email.html`/`last_digest.json` sind
Laufzeit-Artefakte, die ignoriert werden) — der tägliche GitHub-Actions-Workflow committet
`data/` automatisch, die Historie wächst also von selbst. Das ist die Grundlage für einen
**echten Forward-Backtest**, der mit der Zeit entsteht.

### C) Forward-IC-Kalibrierung (`src/backtest/calibrate.py`)

Sobald `digest_history.jsonl` genug gereifte Beobachtungen enthält (Laufdatum älter als der
Horizont), misst dieses Tool die **Information Coefficient (IC)** — die Spearman-Rangkorrelation
zwischen jedem Scoring-Bestandteil (breadth, momentum, stage_fit, divergence, option_quality,
emergence, total_score) und der tatsächlich realisierten Forward-Rendite des Tickers. Ein IC nahe
0 heißt: diese Komponente hat nicht besser als Zufall sortiert — Kandidat fürs Herunterwichten.

```bash
python -m src.backtest.calibrate --horizon 90
```

Mit weniger als 30 gereiften Kandidaten-Beobachtungen gibt das Tool ehrlich
`insufficient_history` zurück, statt eine Korrelation mit falscher Präzision vorzutäuschen.

## Track Record

Jedes erzeugte Signal wird in `data/signals.json` gespeichert (Ticker, OCC-Optionssymbol,
Einstiegs-Mittelkurs, Score, These). Bei jedem Lauf werden offene Signale mit aktuellen
Tradier-Quotes neu bewertet und nach ca. 60 Handelstagen oder bei 40 Tagen Restlaufzeit (DTE)
geschlossen. **Hit = Options-Mittelkurs beim Schließen liegt über dem Einstiegs-Mittelkurs.**
Die rollierende Hit Rate und der durchschnittliche P/L stehen in jeder E-Mail und werden durch
den committeten Verlauf in `data/signals.json` öffentlich nachvollziehbar dokumentiert.

## Disclaimer

Keine Anlageberatung. Optionen können wertlos verfallen. Das System liefert datengetriebene
Signale mit gemessener, aber nicht garantierter Trefferquote. Eine Hit Rate über 50 % kann kein
System garantieren — die Design-Entscheidungen (leicht-im-Geld-Calls, lange Laufzeiten,
Mehrquellen-Bestätigung, Signal-Schwellwert) sollen die Trefferwahrscheinlichkeit erhöhen, ohne
sie zu garantieren. Jeder Trade bleibt eine eigenverantwortliche Entscheidung des Menschen.
