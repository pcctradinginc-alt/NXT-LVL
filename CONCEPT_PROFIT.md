# NXT LVL — Weg zur profitablen Kante (Konzept v3)

## Ausgangslage (gemessen, nicht vermutet)

Der erste reale Walk-Forward-Backtest (2026-07-12, 9 Monate, n=96) ergab:

| Komponente | IC @90d | Befund |
|---|---|---|
| total_score | **−0.276** | Composite ist auf dieser Stichprobe **anti-prädiktiv** |
| breadth | **−0.305** | stärkster Negativtreiber |
| divergence | −0.169 | „noch nicht gelaufen" kämpft gegen Momentum |
| theme_momentum | +0.051 | einziger (schwach) positiver Kandidat |

Hit-Rate nach Score-Quartil war **invertiert** (Q4=33 %, Q2=71 %). Caveats: n winzig,
arXiv 429-gedrosselt, GitHub ausgeschlossen, nur 9 Monate, nur Aktienrendite.

**Konsequenz:** Nicht mehr Features auf Plausibilität bauen. Stattdessen: (A) die Messung
schnell und belastbar machen, (B) Scoring **datengetrieben** neu kalibrieren, (C) bekannte,
empirisch dokumentierte Faktoren als Kandidaten testen, (D) ein **Validierungs-Gate**: das
System empfiehlt Trades nur aus einem Scoring, das out-of-sample positive Kante gezeigt hat.

**Ehrlichkeit:** Auch dieser Prozess garantiert keinen Gewinn. Er garantiert nur, dass kein
Geld auf ein nachweislich kantenloses Scoring gesetzt wird — und dass jede behauptete Kante
aus einer out-of-sample-Messung stammt, nicht aus einer Story.

## Phase A — Messinfrastruktur: schnell + belastbar (Voraussetzung für alles)

Problem: 3-Jahres-Lauf hing nach ~2 h; arXiv drosselt (429) → degradierte Daten.

1. **Monats-Bucketing**: Historische Zählungen (arXiv, EDGAR-FTS, HN-Buzz, Jobs) werden pro
   `(Thema, Kalendermonat)` **einmal** abgefragt statt pro `(Thema, Fenster, Stichtag)`.
   Fenster-Zählungen = Summe der Monats-Buckets. Reduziert API-Calls um ~10–20×.
2. **Disk-Cache** (`data/backtest_cache/*.json`): Monats-Buckets und Tradier-Tageshistorien
   werden gecacht — innerhalb eines Laufs und über Läufe hinweg (GitHub `actions/cache`).
   Abgeschlossene Monate sind unveränderlich → Cache nie stale.
3. **Sanfte Drosselung**: arXiv ≥3 s Abstand + exponentieller Backoff bei 429; Zählung via
   `totalResults` (max_results=1) bleibt.
4. **Options-P/L statt nur Aktienrendite**: Der Walk-Forward bewertet zusätzlich einen
   synthetischen 120-DTE-Δ0.60-Call (Black-Scholes, realisierte Vola als IV-Proxy, minus
   halber Spread als Kosten) — validiert wird das Instrument, das wirklich gehandelt wird.

Ziel: 2–3 Jahre × monatliche Stichtage × ~60 Ticker in < 30 Min. → n ≈ 1500–2500.

## Phase B — Kandidaten-Faktoren (empirisch dokumentiert, alle frei)

Neue Komponenten, die im Walk-Forward **mitgemessen** werden (nicht blind live geschaltet):

1. **Preis-Momentum (12-1)**: 12-Monats-Rendite ohne letzten Monat — der am besten
   dokumentierte freie Faktor. Ersatz-Hypothese für die gescheiterte Divergenz-Logik.
2. **Trend-Filter**: Kurs > 50-Tage-Linie UND 50 > 200 („kein fallendes Messer, kein
   Gegen-Trend-Call").
3. **Regime-Filter**: Long-Calls nur, wenn SPY > 200-Tage-Linie (Risk-on). In Risk-off:
   kein Trade — Nicht-Handeln ist ein gültiges, oft profitables Ergebnis.
4. **Theme-Momentum** (einziger positiver IC): bleibt, wird aber gegen die neue Messung
   erneut geprüft.
5. Bestehende Komponenten (breadth, divergence, emergence, stage_fit) bleiben messbar,
   bekommen aber **kein Vertrauen mehr per Default** — sie müssen sich im IC beweisen.

## Phase C — Datengetriebene Kalibrierung mit echtem Out-of-Sample

`src/backtest/optimize.py` (Walk-Forward-Optimierung, keine Blackbox):

1. Zeitachse in **Folds** teilen (z. B. 6-Monats-Blöcke).
2. Auf Fold k: IC je Komponente messen → Gewichte ∝ max(IC, 0) (negative IC ⇒ Gewicht 0;
   **kein** naives Invertieren — Overfitting-Schutz), normiert, Mindest-/Höchstgrenzen.
3. Auf Fold k+1 (**nie gesehen**): das so kalibrierte Scoring bewerten — IC(total),
   Top-Quartil-Alpha, Options-Hit-Rate nach Kosten.
4. **Adoptionskriterien** (alle müssen über die Validierungs-Folds hinweg gelten):
   - IC(total) > 0 im Median der Folds,
   - Top-Quartil-Options-Hit-Rate ≥ 55 % nach Kosten,
   - Top-Quartil schlägt Bottom-Quartil konsistent.
5. Ergebnis: `data/scoring_calibration.json` (Gewichte + Validierungsreport + Zeitstempel).

## Phase D — Validierungs-Gate im Produktivsystem

- `config.yaml → scoring.validation_gate: true` (Default): Ein Trade-Signal (Option) wird
  nur ausgesprochen, wenn eine **gültige, bestandene** Kalibrierung vorliegt; sonst wird die
  Mail als „Beobachtung — Kante nicht validiert" gekennzeichnet (Kandidaten + Scores werden
  weiter berichtet und archiviert, damit die Forward-Messung wächst).
- Produktiv-Scoring lädt die kalibrierten Gewichte aus `scoring_calibration.json`
  (Fallback: config) und wendet Trend-/Regime-Filter als harte Gates an.
- Reward-Engine bleibt als Langfrist-Feedback; die Kalibrierung ist der schnelle Pfad.

## Arbeitsteilung (auf Nutzerwunsch)

- **Sonnet**: Phase A (Cache/Bucketing/Options-P/L im Walk-Forward) und Phase C/D
  (Optimizer, Gate, Integration).
- **Haiku**: Phase B als isoliertes Modul (`src/analysis/trend.py`: Momentum/Trend/Regime,
  pure Funktionen + Tests) sowie **Code-Review** aller Diffs (nur Bugs/Fehler).
- **Opus (Hauptagent)**: Konzept, Aufgabenschnitt, Verifikation, finaler Bug-Review, Läufe.

## Erfolgskriterien & Abbruchkriterien

- **Erfolg**: Kalibriertes Scoring besteht die Adoptionskriterien out-of-sample → Gate
  öffnet, Signale gelten als „validierte Kante" (und werden forward weiter gemessen).
- **Abbruch/Eskalation**: Besteht kein Kandidaten-Set die Kriterien, sagt das System das
  offen („keine validierte Kante — kein Trade") statt zu handeln. Dann sind bessere Daten
  (Bezahlquellen) oder andere Instrumente die Diskussion — nicht mehr Feature-Bau.
