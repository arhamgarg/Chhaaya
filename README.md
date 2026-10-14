# Chhaaya

Chhaaya is a WhatsApp health assistant for rural patients in India. You send it a voice note, a message or a photo, in Hindi, English or Hinglish. It answers from Indian public-health guidelines, explains your lab report in plain words, reminds you to take your medicines, and passes anything urgent or uncertain to your ASHA worker. It does not diagnose.

The project is at an early stage: the service skeleton runs, the features do not exist yet. [docs/design.md](docs/design.md) is the specification, and the open issues are the work plan.

## How it works

A message arrives through the WhatsApp Cloud API and is queued in Postgres. A worker then:

1. turns voice into text with Sarvam's speech-to-text;
2. checks the message against a fixed list of danger signs;
3. routes it with a small intent classifier we fine-tune from MuRIL.

Questions are answered by retrieving passages from MoHFW, ICMR and NHM documents and having an LLM answer only from those passages. Any answer that cites nothing it was given is thrown away. Lab reports are read with OCR and compared against the ranges printed on them. Prescriptions are confirmed by the ASHA before the patient hears anything about them. Replies go back as text and as a voice note.

## Running it locally

`docker compose up` starts the app, the worker and Postgres 16 with pgvector. Settings come from environment variables; `.env.example` lists them all. The service refuses to start with a blank `WHATSAPP_*` value; without a Meta app, any non-blank placeholders work locally, but never deploy with guessable ones. `GET /health` on port 8000 answers `{"status": "ok"}` once the app is up.

```bash
cp .env.example .env   # then set POSTGRES_PASSWORD and the WHATSAPP_* values
docker compose up --build
curl localhost:8000/health
```

## Database and migrations

Everything Chhaaya stores lives in one Postgres 16 database with pgvector: users (patients and ASHAs), every WhatsApp message in and out, escalation cases, medicine reminders and their doses, and the knowledge-base chunks with their 1024-dimensional embeddings. Inbound messages double as the worker's queue: each has a status the worker claims with `SELECT ... FOR UPDATE SKIP LOCKED`, and the unique WhatsApp message id means a retried webhook delivery is never stored twice. Chunks are searched by cosine similarity through an HNSW index. The tables are defined in `src/chhaaya/db.py` and built by Alembic migrations in `migrations/`; `docker compose up` applies them before the app and worker start, and the tests apply them to an empty database in a throwaway container (Docker must be running). Compose passes the password separately through `PGPASSWORD`, preserving characters that have special meanings in URLs. After changing a model, rebuild the migration image and generate the revision inside Compose, where Postgres is reachable; the bind mount saves the new migration in your checkout. Review the generated file before applying it.

```bash
docker compose build migrate
docker compose run --rm --volume ./migrations:/app/migrations migrate alembic revision --autogenerate -m "add x"
docker compose run --rm --volume ./migrations:/app/migrations migrate  # apply migrations
uv run pytest tests/test_migrations.py tests/test_compose.py
```

## Danger-sign detection

Every message is screened for danger signs by rules rather than a model, so anyone can read why a message was flagged. `data/danger_signs.yaml` lists each sign (for example bleeding in pregnancy, convulsions, a baby not feeding, difficulty breathing, unconsciousness) with the page of the MCP card, IMNCI or ASHA module it comes from, its phrasings in Hindi, English and Hinglish, and the urgent reply (go to the nearest health centre or call 108) in all three. `chhaaya.danger_signs.detect(text)` folds case, Unicode forms and common Hinglish spellings ("nhi", "dudh", "rha"), then looks for each phrasing in order with up to six other words between its words, so "pet mein bahut tez dard" still matches "pet tez dard". A negated mention such as "no bleeding" or "pregnancy mein khoon nahi aaya" does not fire, but because a missed sign is far worse than a false alarm, negation only suppresses a match in those narrow patterns: "pregnancy mein bleeding nahi ruk rahi" (the bleeding won't stop) fires. Wiring it into the worker comes with the worker itself (issue #10).

```bash
uv run python -c 'from chhaaya.danger_signs import detect; print(detect("bachcha doodh nahi pee raha"))'
uv run pytest tests/test_danger_signs.py
```

## Knowledge corpus

The Q&A agent answers only from a small set of Indian public-health documents, so they are chosen by hand. `data/sources.csv` lists each one (NHM ASHA modules and handbook, MoHFW guidelines on diarrhoea, immunisation, pregnancy, NCDs and TB, and ICMR Standard Treatment Workflows) with its title, publisher, year, the URL it was fetched from, why it is in, and the SHA-256 of the reviewed PDF. `chhaaya.corpus` downloads the PDFs to `data/pdfs/` and writes `data/chunks.jsonl`; neither is committed. A PDF is accepted, whether downloaded or already in `data/pdfs/`, only if its checksum matches the CSV, so the per-PDF fixes below never touch another edition. It reads each page with PyMuPDF, turns sideways pages upright, and drops running headers and footers (margin lines repeated on many pages), page numbers, contents pages, symbol-font bullets and small figure labels; a small line that contains a number stays, because it can be a dose or a value. Ligatures are re-joined ("ﬁ ve" becomes "five"), and in three PDFs whose fonts map glyphs wrongly, reviewed glyph and word-spacing repairs apply; an unmapped glyph stops the build instead of reaching a chunk. Tables become one line per row, "Table row: caption; header: value; header: value", so each value keeps its condition and column label; a row too long for one chunk is split on its largest cell and repeats every other cell. Reviewed pages are left out by page: flowcharts that only make sense as drawings, a quiz with wrong answers, blank planning and reporting forms, and one infant-positioning table the parser cannot split. From the ICMR posters only reviewed prose boxes (symptoms, risk factors, lifestyle advice, monitoring, danger signs and when to refer) are kept, not their algorithms or dose tables. A new section starts at each heading set in a font larger than the body text; bold body-size lines such as "Remember:" stay inside their section. Each section is cut into chunks of 300 to 500 tokens, counted with the BGE-M3 tokenizer the embeddings will use, at paragraph or sentence boundaries; a chunk never crosses a section, so a short section stays one smaller chunk. Every chunk keeps the document's title, publisher, year and URL, its section heading, and the PDF page it starts on. `data/chunks.jsonl` is replaced only once every source has been chunked, so a failed build leaves the previous one in place. The first run downloads the tokenizer from Hugging Face.

```bash
uv run python -m chhaaya.corpus          # download what's missing, then rebuild data/chunks.jsonl
uv run pytest tests/test_corpus.py
```

## Knowledge base search

Retrieval finds the guideline passages a question needs, even when the question is in Hindi or Hinglish and the guideline is in English. `python -m chhaaya.retrieval` embeds every chunk in `data/chunks.jsonl` with BGE-M3 (a multilingual embedding model, run on the CPU through sentence-transformers) and writes them to `kb_chunks` in one transaction. It first deletes the guideline rows, so a rerun replaces the corpus instead of duplicating it, and ASHA answers stored in the same table are kept. `KnowledgeBase(engine, embed).search(query, k=5)` embeds the query the same way and returns the five nearest chunks by cosine similarity through the HNSW index, best first, each with its score (1 means the same direction as the query) and its title, publisher, year, page, URL and source type. The first run downloads the model, about 2 GB, into the Hugging Face cache; on Linux the project installs the CPU-only PyTorch build. Build the corpus first, then load it through Compose, where Postgres is reachable. The default tests use a small stand-in embedder; `-m retrieval` runs twelve hand-written questions, four each in Hindi, English and Hinglish, against the real model and corpus and prints the hit rate and query latency.

```bash
uv run python -m chhaaya.corpus
docker compose run --rm --volume ./data:/app/data --volume ~/.cache/huggingface:/root/.cache/huggingface worker python -m chhaaya.retrieval
uv run pytest tests/test_retrieval.py
uv run pytest tests/test_retrieval.py -m retrieval -s   # real model and corpus
```

## Medicine lexicon and critical lab values

The document agent checks the medicine names it reads off a prescription against a lexicon. It also checks lab values against a short list of critical thresholds. `data/lexicon_sources.csv` names the two lexicon sources with their URL, licence and the SHA-256 of the reviewed file. The first is the National List of Essential Medicines 2022 (MoHFW). The second is the Common Drug Codes for India flat files from NRCeS at C-DAC, which map Indian brand names to generics. CDCI is licensed CC BY 4.0, but redistribution and derived works are limited to India, and its generic names are SNOMED CT terms covered by the SNOMED CT Affiliate Licence. The generated `data/lexicon.csv` inherits these terms. Neither is committed. `python -m chhaaya.lexicon` downloads both to `data/lexicon/` and refuses a file whose checksum has changed. It then writes `data/lexicon.csv`. Each NLEM 2022 medicine gets one row per section that lists it, with its dosage forms and strengths and the section number and page it came from, read from the PDF's tables with PyMuPDF. Each CDCI brand gets one row with the generic it contains (strength and dose form included) and its CDCI id. The 14 brands whose generic is missing from the package are skipped and counted. `Lexicon.load().lookup(text)` drops a leading dosage-form word ("Tab", "Cap", "Syp", "Inj") from both the query and the names. It then scores every name with rapidfuzz's token-set ratio (0 to 100; 100 when every word of the shorter name is in the longer). It returns the best entry and its score; ties go to the name closest overall, so "Tab Telma 40" finds Telma 40 mg rather than a Telma combination. A strength written without a space ("650mg", "1gm") is read as "650 mg" and "1 g". A lone "T" or "C" is kept, because brand names such as T-Bact and C-Tax start with it. Some brand names belong to more than one generic in CDCI (324 do, for example "Seradol 10 mg oral tablet"), and lookup returns one of them. A medicine that is not in the lexicon still returns its nearest entry, with a low score. `data/critical_values.csv` is committed. Each row gives a test, unit, threshold, direction (`below` or `above`), population (`adult`, `pregnant` or `child`), and an age range in years (from inclusive, to exclusive, blank when open). It also cites the page and URL of its source. The first rows are the severe-anaemia haemoglobin cut-offs from the Anemia Mukt Bharat Abhiyaan Operational Guidelines (MoHFW, June 2026), which follow the WHO 2024 cut-offs. The default tests use a small stand-in lexicon. `-m lexicon` builds the real one, checks that each of the 384 medicines in the NLEM's alphabetical index is in its section tables, and looks up eight common brands.

```bash
uv run python -m chhaaya.lexicon          # download what's missing, then rebuild data/lexicon.csv
uv run python -c 'from chhaaya.lexicon import Lexicon; print(Lexicon.load().lookup("Tab Dolo 650"))'
uv run pytest tests/test_lexicon.py tests/test_critical_values.py
uv run pytest tests/test_lexicon.py -m lexicon -s   # real sources
```

## WhatsApp webhook

Messages reach Chhaaya through the WhatsApp Cloud API. `POST /webhook` checks Meta's `X-Hub-Signature-256` signature against the app secret and rejects anything unsigned or forged. It stores each text, voice note and photo as a pending message keyed by its WhatsApp message id, so a retried delivery is ignored, and answers `200` straight away; the worker does the rest. `GET /webhook` answers Meta's verification challenge. `src/chhaaya/whatsapp.py` is the one client for sending text and audio, uploading media and downloading what patients send.

To connect a test number:

1. At developers.facebook.com create a Business app and add the WhatsApp product. API Setup gives you a test number: copy its phone number id and a temporary access token (valid 24 hours).
2. Under App settings, Basic, copy the app secret.
3. In API Setup add up to five recipient phones and confirm each with the code WhatsApp sends. The test number can only message these.
4. Put the four values in `.env`: `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_APP_SECRET`, and a `WHATSAPP_VERIFY_TOKEN` of your choosing.
5. In Cloudflare Zero Trust create a tunnel, add a public hostname that points to `http://app:8000`, and put its token in `.env` as `CLOUDFLARE_TUNNEL_TOKEN`.
6. In the Meta app, WhatsApp, Configuration, set the callback URL to `https://<your-hostname>/webhook` and the verify token to your `WHATSAPP_VERIFY_TOKEN`, then subscribe to the `messages` field.

```bash
docker compose --profile tunnel up --build
```

Send a text, a voice note and a photo from a recipient phone; each should appear once in the `messages` table.

```bash
docker compose exec db psql -U chhaaya -c "SELECT wa_message_id, kind, status FROM messages"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [AGENTS.md](AGENTS.md) for the rules every change and coding agent follows.

## License

AGPL-3.0. See [LICENSE](LICENSE).
