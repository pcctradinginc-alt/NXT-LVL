# NXT LVL — AI Next-Stage Beneficiary Scanner

## Ziel

Ein vollautomatisches, kostenfreies System, das den aktuellen Stand des KI-Ausbaus anhand
frei verfügbarer Frühindikatoren verfolgt und **nicht die aktuellen Gewinner**, sondern die
**Profiteure der nächsten Stufe (3–12 Monate Horizont)** frühzeitig identifiziert.
Für den Top-Kandidaten wird via Tradier API eine passende Option mit passendem Zeithorizont
ausgewählt und per E-Mail mit kurzer Begründung versendet.

**Wichtig / Ehrlichkeit:** Eine Hit Rate > 50 % kann kein System garantieren. Das System ist
deshalb so gebaut, dass es (a) durch Design-Entscheidungen die Trefferwahrscheinlichkeit
maximiert (ITM-Optionen, lange Laufzeiten, Mehrquellen-Bestätigung, Signal-Schwellwert) und
(b) seine eigene Hit Rate **misst und in jeder Mail ausweist**, sodass Schwellwerte
datenbasiert nachgeschärft werden können. Kein Trade wird automatisch ausgeführt — das System
liefert Signale, die Entscheidung bleibt beim Menschen.

## Kernidee: Das Stufenmodell des KI-Ausbaus

Der KI-Ausbau verläuft in Wertschöpfungs-Wellen. Wer die aktuelle Welle erkennt, kann die
nächste antizipieren, bevor der Markt sie einpreist:

| # | Stufe | Beispiel-Profiteure |
|---|-------|---------------------|
| 1 | Compute / Halbleiter (Training) | NVDA, AMD, AVGO, TSM |
| 2 | Datacenter-Infrastruktur | VRT, DELL, ANET, MU, SMCI |
| 3 | Energie / Kühlung / Netz | VST, CEG, GEV, ETN, PWR, MOD |
| 4 | Inference / Netzwerk / Edge | MRVL, CRDO, ALAB, COHR |
| 5 | Software / Agenten / Daten | PLTR, NOW, DDOG, MDB, SNOW |
| 6 | Vertikale KI-Adoption | ISRG, CRWD, PANW, TER |
| 7 | Robotik / Physical AI | SYM, ROK, TER, TSLA |

Das LLM bestimmt aus den gesammelten Signalen, **welche Stufe gerade läuft** und **welche als
nächstes kommt** — und kann auch Ticker außerhalb dieser statischen Liste vorschlagen. Der
Code validiert jeden Vorschlag (handelbare, liquide Optionen via Tradier).

## Datenquellen (alle kostenlos)

1. **SEC EDGAR (Capex der Tech-Giganten)** — offizielle XBRL-API, kein Key nötig.
   Quartalsweise `PaymentsToAcquirePropertyPlantAndEquipment` für MSFT, GOOGL, AMZN, META,
   ORCL, NVDA. Metrik: Capex-Wachstum QoQ/YoY → misst, wie viel Geld in den Ausbau fließt
   (Frühindikator für Stufe 2–3).
2. **GitHub-Trends (Developer-Metriken)** — GitHub REST API (im Actions-Runner kostenlos via
   `GITHUB_TOKEN`). Stern-Wachstum neuer AI-Repos, Aktivität in Schlüssel-Ökosystemen
   (Inference, Agents, Robotics), trending Topics → misst, woran Entwickler *jetzt* arbeiten
   (Frühindikator für Stufe 4–7).
3. **Stellenanzeigen** — Hacker-News-„Who is hiring"-Threads via Algolia API (kostenlos, kein
   Key) + öffentliche Greenhouse/Lever-Job-Boards ausgewählter AI-Firmen. Keyword-Zählung
   (inference, agents, robotics, GPU, datacenter, …) → misst, wofür Firmen einstellen.
4. **arXiv (Forschungstrends)** — kostenlose Atom-API. Themen-Häufigkeit in cs.AI/cs.LG/cs.RO
   der letzten 30 Tage → misst, was in 6–18 Monaten in Produkte fließt.
5. **Hacker-News-Diskussionen** — Algolia API. Buzz-Zählung pro Thema/Stufe → Stimmungs- und
   Aufmerksamkeitsindikator.

Alle Collector sind fehlertolerant: Fällt eine Quelle aus, läuft der Rest weiter.

## Pipeline (läuft werktäglich vor US-Börsenöffnung via GitHub Actions)

```
Collect (5 Quellen) → Aggregate (lokal, tokensparend) → LLM-Analyse (Gemini Free)
→ Scoring (deterministisch, im Code) → Optionsauswahl (Tradier) → E-Mail (Gmail SMTP)
→ Track-Record-Update (committed ins Repo)
```

### 1. Aggregation (Kosteneffizienz)
Rohdaten werden **lokal** zu kompakten Kennzahlen verdichtet (Wachstumsraten, Zähler,
Top-Listen). Nur dieses Digest (~3–4k Tokens) geht an Gemini — ein einziger LLM-Call pro
Lauf, weit unter dem Free-Tier-Limit.

### 2. LLM-Analyse (Gemini 2.5 Flash, Free Tier)
Ein Call mit strukturiertem JSON-Output:
- Einschätzung der **aktuellen Stufe** des KI-Ausbaus,
- Identifikation der **nächsten Stufe** (3–12 Monate),
- 5–10 Kandidaten-Ticker der nächsten Stufe mit Kurzthese,
- pro Kandidat: Signalstärke-Einschätzung je Datenquelle.

### 3. Scoring (deterministisch, 0–100)
Das LLM liefert Thesen — die Bewertung macht der Code, damit sie reproduzierbar ist:

| Komponente | Gewicht | Messung |
|---|---|---|
| Signalbreite | 25 % | Wie viele unabhängige Quellen bestätigen den Kandidaten? |
| Signal-Momentum | 25 % | Wachstumsraten der zugrunde liegenden Metriken |
| Stufen-Fit | 20 % | Gehört der Wert zur identifizierten *nächsten* Stufe (nicht zur aktuellen)? |
| Divergenz | 20 % | Signal stark, Kurs noch nicht gelaufen (3-Monats-Performance via Tradier) → noch nicht eingepreist |
| Options-Qualität | 10 % | Liquidität: Open Interest, Bid-Ask-Spread der Zieloption |

Nur wenn der Top-Kandidat **Score ≥ 70** erreicht UND von **≥ 2 unabhängigen Quellen**
bestätigt ist, wird ein Signal erzeugt. Sonst: „Kein Trade heute" (kein Signal ist ein
gültiges Ergebnis — Qualität vor Quantität, das schützt die Hit Rate).
Cooldown: derselbe Ticker wird frühestens nach 14 Tagen erneut signalisiert.

### 4. Optionsauswahl (Tradier API, Key vorhanden)
Für den Top-Kandidaten:
- **Laufzeit:** 90–180 Tage (DTE), passend zum 3–12-Monats-Horizont der These — Zeitwertverfall
  trifft die These nicht sofort.
- **Strike:** Call mit Delta ≈ 0,60–0,70 (leicht im Geld) — höhere Gewinnwahrscheinlichkeit
  als OTM-Lotterielose; das ist die wichtigste Design-Entscheidung für Hit Rate > 50 %.
- **Liquiditätsfilter:** Open Interest ≥ 100, Spread ≤ 10 % des Mittelkurses.
- Fallback: erfüllt keine Option die Kriterien → Signal ohne Options-Empfehlung (nur Aktie).

### 5. E-Mail (Gmail SMTP, App-Passwort)
Kompakte HTML-Mail an pcctradinginc@gmail.com:
Top-Pick + konkreter Options-Kontrakt + 3-Satz-Begründung + Score-Tabelle der Top 5 +
aktuelle rollierende Hit Rate + Stufen-Einschätzung („wo stehen wir, was kommt als
nächstes").

### 6. Track Record (Selbstmessung der Hit Rate)
Jedes Signal wird in `data/signals.json` protokolliert (Ticker, OCC-Optionssymbol,
Einstiegs-Mittelkurs, Underlying-Kurs, These, Score, Horizont). Bei jedem Lauf werden offene
Signale neu bewertet (Tradier-Quotes); nach 60 Handelstagen oder bei 40 DTE Restlaufzeit wird
ein Signal geschlossen: **Hit = Options-Mittelkurs über Einstieg**. Rollierende Hit Rate und
Ø-P/L stehen in jeder Mail. Die Actions-Pipeline committet die Daten zurück ins Repo —
vollständige, öffentliche Nachvollziehbarkeit.

## Betrieb & Kosten

| Posten | Kosten |
|---|---|
| GitHub Actions (öffentliches Repo, ~5 Min/Werktag) | 0 € |
| SEC EDGAR, GitHub API, HN/Algolia, arXiv | 0 € (keine Keys) |
| Gemini 2.5 Flash Free Tier (1 Call/Tag) | 0 € |
| Tradier API (vorhanden) | 0 € |
| Gmail SMTP | 0 € |
| **Summe** | **0 €/Monat** |

### Benötigte GitHub Secrets
| Secret | Zweck |
|---|---|
| `GEMINI_API_KEY` | LLM-Analyse (aistudio.google.com → „Get API key") |
| `TRADIER_API_KEY` | Optionsdaten (vorhanden) |
| `GMAIL_APP_PASSWORD` | Mail-Versand (Google-Konto → Sicherheit → App-Passwörter) |
| `MAIL_FROM` / `MAIL_TO` | Absender/Empfänger (Gmail-Adresse) |

Optional: `TRADIER_ENV=sandbox` für Sandbox-Basis-URL.

## Technik

- Python 3.12, einzige Abhängigkeit: `requests` (schnelle, robuste CI-Installs).
- LLM & Tradier über REST direkt — kein SDK-Ballast.
- `--dry-run`-Modus: läuft ohne Secrets (Collector echt, LLM/Tradier/Mail gestubbt) für
  lokale Tests und CI-Smoke-Tests.
- Konfiguration (Universum, Stufen, Gewichte, Schwellwerte) in `config.yaml` — Strategie
  nachschärfen ohne Code-Änderung.
- Workflow committet `data/` zurück (`contents: write`).

## Disclaimer

Keine Anlageberatung. Optionen können wertlos verfallen. Das System liefert
datengetriebene Signale mit gemessener, aber nicht garantierter Trefferquote.
