# NXT LVL — Emergence & Reward Engine (Erweiterung)

Ergänzt den Basis-Scanner (siehe [`CONCEPT.md`](./CONCEPT.md)) um zwei Fähigkeiten:

1. **Emergence Detection** — erkennt *neue* und *beschleunigende* Themen, Firmen und
   Profiteur-Cluster aus mehreren kostenlosen Datenströmen, statt nur eine feste Watchlist
   abzuklopfen. Kandidaten können so **außerhalb der ursprünglichen 7-Stufen-Watchlist**
   entstehen.
2. **Reward Engine** — ein *regelbasiertes, erklärbares* Feedback-System, das die
   Prognosequalität vergangener Signale misst und daraus die Feature-Gewichte und
   Quellen-Zuverlässigkeiten **innerhalb fester Grenzen** und **vollständig protokolliert**
   nachjustiert. Keine Blackbox.

Leitprinzip: **Alles bleibt kostenlos, deterministisch und erklärbar.** Jede Gewichtsänderung
ist im Log begründet; jeder emergente Kandidat trägt seine Entdeckungs-Belege mit sich.

---

## Teil A — Emergence Detection

### A.1 Themen-Taxonomie (`config.yaml → themes`)

Ein gegenüber den 7 Stufen **breiteres** Themenraster. Jedes Thema hat `id`, `name`,
`keywords` (Phrasen für die Textzählung) und `tickers` (bekannte Profiteure — ein Universum,
das deutlich über die Watchlist hinausgeht, damit neue Namen auftauchen können). Startthemen:

| id | Thema | Beispiel-Keywords |
|----|-------|-------------------|
| ai_inference | AI Inference | "inference", "serving", "vllm", "tokens per second", "kv cache" |
| dc_cooling | Data Center Cooling | "liquid cooling", "immersion cooling", "direct-to-chip", "CDU" |
| power_grid | Power Grid / Transformers | "transformer", "switchgear", "grid interconnect", "substation" |
| optical_networking | Networking / Optical Interconnects | "optical interconnect", "co-packaged optics", "800G", "silicon photonics" |
| hbm_memory | HBM / Memory | "HBM", "HBM3E", "high bandwidth memory", "memory bandwidth" |
| custom_silicon | Custom Silicon / ASIC | "ASIC", "custom silicon", "accelerator", "TPU", "tape-out" |
| edge_ai | Edge AI | "edge ai", "on-device", "npu", "edge inference" |
| ai_security | AI Security | "ai security", "model security", "prompt injection", "llm firewall" |
| enterprise_automation | Enterprise Automation | "agentic workflow", "rpa", "process automation", "ai agents" |
| ai_workflow_sw | AI Workflow Software | "orchestration", "llmops", "eval", "observability", "vector database" |
| data_infra | Data Infrastructure | "data infrastructure", "lakehouse", "streaming", "feature store" |

Themen und Ticker sind reine Konfiguration — Erweiterung ohne Code-Änderung.

### A.2 Datenströme (alle kostenlos)

Wiederverwendet werden die bestehenden Collector-Korpora, ergänzt um eine neue keyless Quelle:

| Quelle | Rolle in Emergence | Zugriff |
|--------|--------------------|---------|
| **SEC EDGAR Full-Text Search** (neu) | Themen-Häufigkeit in Filings (10-K/10-Q/8-K) als Näherung für „SEC-Filings & Earnings-Kommentare" | `https://efts.sec.gov/LATEST/search-index?q="<phrase>"&forms=10-K,10-Q,8-K&startdt=&enddt=` → `hits.total.value`; UA-Pflicht, keyless |
| GitHub | Themen-Aktivität (Repo-/Code-Treffer je Keyword) + Firmen-Nennungen in Repo-Namen/Beschreibungen | bestehende REST-Suche |
| HN „Who is hiring" | Job-Nennungen je Thema + Firmen-Co-Occurrence in Kommentaren | bestehender Algolia-Collector |
| arXiv | Forschungs-Häufigkeit je Thema | bestehender Atom-Collector |
| HN Buzz | Aufmerksamkeits-Häufigkeit je Thema | bestehender Algolia-Collector |

### A.3 Beobachtung, Baseline & Beschleunigung

Pro Lauf entsteht je Thema eine **ThemeObservation**:

```
{ theme_id, date, per_source_counts:{edgar_fts, github, jobs_hn, arxiv, hn_buzz},
  frequency (Summe), source_diversity (# Quellen mit count>0) }
```

Diese werden in `data/baseline.json` als **rollierende Historie** je Thema gespeichert
(gekappt auf `emergence.baseline_window`, z. B. 30 Beobachtungen). Aus der Historie:

- **baseline_mean / baseline_std** der Frequenz (bzw. je Quelle),
- **acceleration_z** = (frequency − mean) / max(std, 1),
- **acceleration_ratio** = frequency / max(mean, 1),
- **novelty** ∈ [0,1]: hoch, wenn das Thema wenige historische Beobachtungen hat
  (`< min_history_for_baseline`) oder erst kürzlich erstmals über 0 lag
  (`novelty_window_days`); sinkt mit wachsender Historie.

### A.4 Emergence Score (0–100, deterministisch)

```
emergence_score = 100 * Σ_k  w_k * norm_k
```
mit `emergence.score_weights` (Default): `frequency 0.30`, `acceleration 0.35`,
`diversity 0.20`, `novelty 0.15`. Normalisierung dokumentiert und geclippt:

- `norm_frequency` = frequency relativ zum Max über alle Themen dieses Laufs,
- `norm_acceleration` = clip(acceleration_z / 3, 0..1) (3σ = voll),
- `norm_diversity` = source_diversity / 5,
- `norm_novelty` = novelty.

Ein Thema gilt als **emergent**, wenn `emergence_score ≥ emergence.theme_threshold`
(Default 60) **und** `source_diversity ≥ emergence.min_sources` (Default 2). Das erzwingt
Mehrquellen-Bestätigung und filtert Einzelquellen-Rauschen.

### A.5 Firmen-/Entity-Erkennung (neue, wiederkehrende Namen)

Ein `names_dictionary` (aus `themes[].tickers` plus optionaler `config.yaml → entity_aliases`)
bildet Firmenname-Varianten → Ticker. Über die Freitext-Korpora (HN-Job-Kommentare,
GitHub-Repo-Namen/Beschreibungen, arXiv-Titel) werden Nennungen je Firma und Quelle gezählt.

Eine Firma wird zum **emergenten Kandidaten**, wenn:
1. sie in **≥ `min_sources` unabhängigen** Quellen genannt wird, **und**
2. ihr zugeordnetes Thema **emergent** ist (A.4), **und**
3. sie **kein aktueller Mega-Cap-Gewinner** ist (Ausschlussliste NVDA/MSFT/GOOGL/AMZN/META…).

Weil das Themen-Universum breiter ist als die 7-Stufen-Watchlist, tauchen so **Namen
außerhalb der Original-Watchlist** auf. Zusätzlich werden die **Top-emergenten Themen an das
LLM** übergeben, das für diese Themen weitere, auch völlig neue Ticker vorschlagen darf →
zweiter Weg für Nicht-Watchlist-Kandidaten. Jeder vom LLM genannte Ticker wird vom Code gegen
Tradier auf handelbare, liquide Optionen validiert.

### A.6 Ausgabe der Emergence-Schicht

```
{ emergent_themes:[{theme_id, name, emergence_score, acceleration_z, source_diversity,
                    novelty, drivers:{...}, confirming_sources:[...]}],
  emergent_candidates:[{ticker, theme_id, sources:[...], emergence_score,
                    drivers:{frequency, acceleration, diversity, novelty},
                    first_seen, in_watchlist:bool}] }
```

Diese Kandidaten werden mit den LLM-Kandidaten des Basissystems **zusammengeführt**
(Dedupe per Ticker; emergente Herkunft wird am Kandidaten vermerkt: `discovery.via="emergence"`).
Das Scoring erhält eine zusätzliche Teilkomponente **`emergence`** (der Emergence Score des
zugehörigen Themas), sodass emergente Kandidaten regulär mitbewertet werden.

---

## Teil B — Reward Engine

### B.1 Erweitertes Signal-Objekt (rückwärtskompatibel zu `tracking.py`)

Beim Erzeugen eines Signals wird zusätzlich gespeichert:

```
date, ticker(Kandidat), score, data_sources[], reasoning(Begründung),
recommended_horizon_days, price_at_signal(Underlying), benchmark_symbol,
benchmark_at_signal, option_idea{occ_symbol, strike, exp, mid, delta, oi, spread},
data_quality_score,                     # 0-100, s. B.2
source_attribution[],                   # welche Quellen den Kandidaten stützten
feature_attribution{feature: beitrag},  # gewichteter Score-Beitrag je Feature
discovery{via, theme_id, drivers},      # watchlist | emergence
horizon_evals{30:{},60:{},90:{},180:{}} # gefüllt durch B.3
```

### B.2 Datenqualitäts-Score (0–100)

Deterministisch aus dem Digest zum Signalzeitpunkt: Anteil aktiver Quellen, Vollständigkeit
der Capex-Reihen, ob GitHub durch Rate-Limit degradiert war, Anzahl HN-Kommentare etc.
Fließt in den Report ein und dient der Reward-Engine als Vertrauensgewicht (schwache Datenlage
→ Signal wird bei der Belohnung geringer gewichtet).

### B.3 Horizont-Auswertung (30 / 60 / 90 / 180 Tage)

Bei jedem Lauf prüft der Evaluator alle Signale und füllt jeden **fälligen** Horizont
(Alter ≥ Horizont, noch nicht ausgewertet) über Tradier-History (Underlying **und** Benchmark):

- **absolute Rendite** (Underlying seit Signal),
- **Rendite ggü. Benchmark** (Alpha = Underlying-Rendite − Benchmark-Rendite),
- **maximaler Drawdown** im Zeitraum,
- **Hit** (Alpha > 0 am jeweiligen Horizont),
- **Signal kam vor breiter Marktbewegung** (ja/nein: Underlying bewegte sich vor dem
  Benchmark — frühe Relativstärke),
- **Optionsidee theoretisch profitabel** (ja/nein: Black-Scholes-Neubewertung der ursprüngl.
  Option mit aktuellem Underlying/Restlaufzeit vs. Einstiegs-Mid — falls Optionsidee vorhanden),
- **Liquiditätsqualität der Option** (aus OI/Spread zum Signalzeitpunkt),
- **ursprüngliche Begründung eingetroffen** (ja/nein: Heuristik — blieb das zugehörige Thema
  emergent bzw. stieg der Emergence Score weiter?).

Weil der Evaluator rein datумbasiert über die gesamte `signals.json` läuft, können **alte
Signale jederzeit nachträglich (retroaktiv) bewertet** werden — Akzeptanzkriterium erfüllt.

### B.4 Reward-Logik (regelbasiert, begrenzt, protokolliert)

Zwei Ledger in `data/weights.json`:
- **feature_weights** — die Scoring-Gewichte (breadth, momentum, stage_fit, divergence,
  option_quality, **emergence**). Werden adaptiv nachjustiert.
- **source_reliability** — Multiplikator ∈ `[reliability_min, reliability_max]` (Default
  0.5–1.5) je Datenquelle, der in Breadth/Attribution einfließt.

Aus den ausgewerteten Signalen werden **Attributions-Ledger** gebildet: je Feature und je
Quelle `{n, wins, sum_alpha}` (gewichtet mit `data_quality_score`). „Win" = positives Alpha am
`primary_horizon` (Default 90 T). **Overheated** = Score/Emergence hoch (≥
`overheated_score_threshold`), aber Alpha negativ → zählt als Strafe für die treibenden
Features/Quellen.

`update_weights()` (bei jedem Lauf, nach der Evaluation):
- nur wenn `n ≥ min_samples` (Default 5) für das Feature/die Quelle — sonst **dokumentierter
  Skip**;
- Nudge = `learning_rate * (win_rate − 0.5)` + kleiner Alpha-Term, **hart geclippt** auf
  `step_max` (Default ±0.02) pro Lauf und auf absolute Grenzen `weight_bounds`
  (Default 0.05–0.45);
- Feature-Gewichte werden anschließend **renormalisiert** (Summe = 1);
  Source-Reliabilities werden nur geclippt (keine Renormalisierung);
- **jede** Änderung (und jeder Skip) wird an `weights.json → history` angehängt:
  `{date, target, old, new, delta, reason, evidence:{n, win_rate, avg_alpha}}` und zusätzlich
  ins Log geschrieben. Keine Optimierung ohne Erklärbarkeit; keine Änderung außerhalb der
  Grenzen.

Das Scoring liest die Gewichte künftig aus `weights.json` (Fallback: `config.yaml`), sodass die
gelernten Anpassungen wirksam werden — transparent und jederzeit im Repo nachvollziehbar
(Actions committet `data/` zurück).

### B.5 Erklärbarkeits-Report (E-Mail-Erweiterung)

Zusätzliche Abschnitte in der bestehenden HTML-Mail:
- **„Warum wurde der Kandidat entdeckt?"** — Herkunft (Watchlist vs. Emergence), Thema,
  Treiber (Frequenz/Beschleunigung/Diversität/Novelty), bestätigende Quellen.
- **Risiken** — automatische Flags: bereits gelaufen (niedrige Divergenz), Einzelquellen-Risiko,
  geringe Options-Liquidität, überhitztes Thema, hohe Novelty (unbewährt), schwache Datenqualität.
- **Reward-Status** — aktuelle Feature-Gewichte, letzte protokollierte Änderungen,
  Quellen-Zuverlässigkeiten, Hit-Rate je Horizont.

---

## Neue/erweiterte Dateien

```
config.yaml                     # + themes, entity_aliases, emergence, reward
data/baseline.json              # rollierende ThemeObservation-Historie
data/weights.json               # feature_weights + source_reliability + history
data/signals.json               # erweitertes Schema (rückwärtskompatibel)
src/collectors/edgar_fts.py     # SEC Full-Text-Search Themenzählung (neu, keyless)
src/emergence/detector.py       # Themen-Emergence + Score
src/emergence/entities.py       # Firmen-Co-Occurrence-Erkennung
src/emergence/baseline.py       # Laden/Aktualisieren der Baseline-Historie
src/reward/evaluator.py         # 30/60/90/180-Tage-Auswertung inkl. Benchmark
src/reward/engine.py            # Attribution + begrenzte, geloggte Gewichtsanpassung
src/reward/weights.py           # Laden/Speichern/History von weights.json
src/analysis/options_math.py    # Black-Scholes-Neubewertung (Options-Profitabilität)
```

## Akzeptanzkriterien → Tests

1. **Neues Thema wird erkannt** → `test_emergence_detects_new_theme` (Baseline niedrig →
   Sprung hoch ergibt hohen Emergence Score + hohe Beschleunigung).
2. **Kandidat aus neuem Cluster außerhalb der Watchlist** →
   `test_emergent_candidate_outside_watchlist` (Firma nicht in 7-Stufen-Watchlist, aber in
   ≥2 Quellen bei emergentem Thema → erscheint als Kandidat).
3. **Alte Signale retroaktiv bewertbar** → `test_retroactive_evaluation` (Signal 200 T alt +
   Fake-History → 30/60/90/180-Evals gefüllt, Alpha/Drawdown/Hit korrekt).
4. **Gewichte nachvollziehbar angepasst** → `test_weight_update_bounded_and_logged` (Nudge
   innerhalb Step/Grenzen, Renormalisierung, History-Eintrag mit Begründung; Skip bei zu
   wenig Samples dokumentiert).
5. **Report erklärt den Kandidaten** → `test_report_explains_candidate` (E-Mail enthält
   „Warum", „Quellen", „Risiken").

## Kosten & Betrieb

Unverändert **0 €/Monat**: EDGAR-FTS und alle Zählungen sind keyless; das LLM bleibt bei
**einem** Call pro Lauf (der Digest wird lediglich um die Emergence-Zusammenfassung ergänzt).
Baseline/Weights/Signals werden von GitHub Actions ins Repo zurückcommittet und bilden den
transparenten, wachsenden Track Record.
