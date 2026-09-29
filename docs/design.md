# Chhaaya design

This is the working specification for Chhaaya. If code and this document disagree, one of them is wrong; fix whichever it is in the same PR.

Chhaaya (छाया, "shade") is a WhatsApp health assistant for rural patients who speak Hindi, English or Hinglish. People can send it a voice note, a text message or a photo of a lab report or prescription. It answers from Indian public-health guidelines, explains reports in plain language, reminds them to take their medicines, and hands anything risky or uncertain to their ASHA (the village community health worker).

Chhaaya does not diagnose. It never tells a patient what condition they have, and it never changes or starts a medicine.

## 1. Problem

Many rural patients get health advice, prescriptions and lab reports they cannot read or check:

- **Few doctors.** India has one government allopathic doctor for every 11,082 people (National Health Profile 2018). At rural Community Health Centres, only 4,413 of the 21,964 specialist posts the norms require were filled in March 2023, a shortfall of about 80% (Health Dynamics of India 2022-23).
- **Language and literacy.** India has 22 scheduled languages. Clinical documents are written in English, and many patients cannot read them in any language.
- **One-way programmes.** Existing government voice programmes such as Kilkari push pre-recorded messages and cannot answer a question (see section 11). The LLM assistant closest to this idea, ASHABot, serves health workers, not patients.

Phones, however, are already in the village. India had about 886 million active internet users in 2024, more than half of them rural (IAMAI-Kantar, Internet in India 2024), and WhatsApp voice notes are how many of them communicate.

## 2. Scope of the first release

In scope:

- **WhatsApp Cloud API test number** as the only channel. The test number can message at most five registered phones, which covers the demo: team phones act as patients and one acts as the ASHA.
- **Languages:** Hindi (Devanagari), English, and Hinglish (Hindi in Latin script, mixed with English). The reply uses the language and script the patient wrote or spoke in.
- **Input and output:** text, voice notes and photos in; text and voice notes out.
- **Four jobs:** answer health questions, explain lab reports, read prescriptions (with ASHA confirmation), and send medicine reminders.
- **Escalation** to a registered ASHA over WhatsApp, with the ASHA's reply relayed back to the patient.

Out of scope for the first release:

- **IVR (phone calls).** It needs a paid telephony account and a KYC'd business. It is not part of this project.
- **Tamil and Telugu.** Both are supported by the speech and TTS APIs, and they are the next languages once someone can build and check an evaluation set for them.
- **Handwriting.** It is never read autonomously. Every prescription, handwritten or printed, is confirmed by the ASHA before the patient hears anything about it (section 5.2).
- **Diagnosis, symptom checking and triage scores.**
- **Real patient data.** All testing uses synthetic documents, or team members' own documents with personal details blacked out.

## 3. How a message flows

```
WhatsApp ──> webhook ──> messages table ──> worker
                                              │
             voice note ── speech-to-text ────┤
             photo ─────────────────────────> document agent
             text ────────────────────────────┤
                                              ▼
                                    danger-sign check ── match ──> urgent reply + escalation
                                              │
                                        intent classifier
                         ┌───────────┬────────┴───────┬──────────────┐
                     question   reminder setup  reminder reply    small talk
                         │           │                │              │
                     Q&A agent   reminder agent  reminder agent  canned reply
                         │
                 retrieve ─> generate ─> citation check ─> reply (text + voice)
                         └── weak evidence ─> escalation agent
```

1. **Receive.** The webhook checks Meta's `X-Hub-Signature-256` signature, stores the message keyed by its WhatsApp message id (so retried deliveries are ignored), and returns `200` immediately. WhatsApp retries any webhook that responds slowly.
2. **Work.** A single worker process polls Postgres for pending work using `SELECT ... FOR UPDATE SKIP LOCKED`. Pending work means inbound messages and due reminders. There is no Redis or Celery; one queue in the database we already run is enough at this scale.
3. **Normalise.** Voice notes go through speech-to-text, which also returns the language. For typed text, Sarvam's language identification gives the language and script. Messages from a registered ASHA number go straight to the escalation agent.
4. **Screen for danger signs** before anything else (section 6).
5. **Route.** Photos go to the document agent. Text goes through the intent classifier (section 7) and then to the matching agent. If the classifier is unsure, the message is treated as a question: the Q&A agent is the safest default because it abstains when it has no evidence.
6. **Reply** in the patient's language, as text plus a voice note. Voice replies are short (about 60 words) because a long voice note is hard to follow.

## 4. Components

| Part | Choice | Why |
|---|---|---|
| Channel | WhatsApp Cloud API (test number) | Free, and it is what the target users already use |
| Service | Python 3.12, FastAPI | All the ML is in Python; one language for the whole backend |
| Tooling | uv, ruff, ty, pytest | |
| Database | PostgreSQL 16 + pgvector | Records, the work queue and the vector index in one database |
| Speech to text | Sarvam Saaras v3 | Handles Hindi, English and code-mixed speech |
| Text to speech | Sarvam Bulbul v3 | Hindi and English voices; ffmpeg converts the output to OGG/Opus, which WhatsApp shows as a voice note |
| LLM | Sarvam-105B | Strong on Indic languages and hosted in India. Unlike free-tier APIs, it does not keep our data for training |
| OCR | Sarvam Vision | Async job API; returns Markdown and page-level JSON |
| Embeddings | BGE-M3, run locally | Multilingual, so a Hindi query can match an English guideline. Sarvam has no embedding API |
| Intent classifier | MuRIL (base), fine-tuned by us | Pretrained on Indian languages including transliterated text, which matters for Hinglish |
| Fuzzy matching | rapidfuzz | Medicine-name lookup |
| Running it | Docker Compose (app, worker, Postgres) and a Cloudflare tunnel for the webhook URL | |

All Sarvam calls go through one small client module, so that changing a model version touches a single file.

## 5. The four agents

An "agent" here is a module with one job and a fixed set of tools. None of them can call another agent directly, except that any of them can hand a case to the escalation agent.

### 5.1 Q&A agent

1. The LLM rewrites the message into a short English search query, turning lay terms into clinical ones ("sugar" becomes diabetes, "loose motion" becomes diarrhoea).
2. That query retrieves the top 5 chunks from the knowledge base by cosine similarity.
3. **Weak evidence.** If the best score is below the threshold τ, the agent does not answer. It tells the patient their ASHA will reply and opens an escalation. τ is tuned on the evaluation set (section 9), not guessed.
4. **Generation.** Otherwise the LLM answers using only the retrieved chunks, citing chunk ids.
5. **Citation check.** Code verifies that every cited id is one of the chunks that were retrieved. If the answer cites nothing, or cites something that was not retrieved, it is discarded and the case escalates.
6. **Delivery.** The ids are replaced by readable source names ("Source: ASHA Module 7, NHM"), and the answer is sent.

The system prompt forbids naming a condition the patient has, changing or starting a medicine, and giving doses. The adversarial evaluation set checks that these rules hold.

### 5.2 Document agent

A photo goes to Sarvam Vision, and the LLM turns the OCR output into a fixed JSON shape. It is either a lab report (test, value, unit, printed reference range) or a prescription (medicine, strength, frequency, duration).

- **Lab reports** are explained automatically. For each test, the agent says whether the value is inside the reference range printed on the report, plus what the test measures if the knowledge base covers it. A value past a critical threshold (for example, haemoglobin below 7 g/dL) escalates at once. Critical thresholds live in a data file, and every row cites its source.
- **Prescriptions** are never explained until a human has checked them. OCR cannot tell handwriting from print reliably, and a misread drug name can match a different real drug, which a lexicon check will not catch. The agent therefore:
  1. matches each medicine name against the medicine lexicon;
  2. sends the ASHA the photo and the extracted list;
  3. waits for the ASHA to reply "ok" or send corrections.

  Only then does the patient get the explanation (medicine, how much, when, for how long, as written by the doctor) and an offer to set reminders.

### 5.3 Reminder agent

Reminders are set in one of two ways: from a confirmed prescription, or by the patient saying something like "subah 8 baje aur raat 8 baje metformin yaad dilana" ("remind me to take metformin at 8 am and 8 pm"). In the second case, the LLM extracts the medicine, times and duration, and the agent reads them back for a yes before saving.

At each due time, the worker sends the reminder. The patient replies (for example "le li" or "taken") to mark the dose as taken. After three missed doses in a row, the ASHA is told.

WhatsApp only allows free-form messages within 24 hours of the patient's last message. Outside that window, reminders must go out as a pre-approved message template (`medicine_reminder`). The ASHA's case alerts (`asha_case_alert`) and relayed replies (`case_update`) work the same way.

### 5.4 Escalation agent

The escalation agent opens a numbered case, sends the ASHA a summary, and tells the patient someone will get back to them. It handles:

- danger signs;
- weak retrieval evidence and failed citation checks;
- critical lab values;
- prescription confirmations;
- repeatedly missed doses.

The ASHA answers with the case number ("12 ..."). The reply is relayed to the patient and stored in the knowledge base as a health-worker answer. When a future answer uses one, the citation names it as such ("answered earlier by a health worker") instead of presenting it as a guideline.

## 6. Danger signs

Danger signs are detected by rules, not by a model. That keeps them auditable: anyone can read the list and see why a message was flagged.

The list comes from the danger signs in the ASHA training modules, the Mother and Child Protection card, and IMNCI (the national guidelines for sick newborns and children). Examples include bleeding in pregnancy, convulsions, a newborn not feeding, difficulty breathing, and unconsciousness. Each sign has phrasings in Hindi, English and Hinglish.

The matcher runs on the original text and on the English search query, and a match on either one fires. A match immediately:

- tells the patient to go to the nearest health centre or call 108 (the ambulance line);
- alerts the ASHA.

Missing a danger sign is far worse than a false alarm, so the evaluation reports recall (the share of real danger signs caught) separately and holds it to a much higher bar than precision.

## 7. The trained model: intent classifier

The router is a fine-tuned MuRIL classifier with four labels:

- `question`
- `reminder_setup`
- `reminder_reply`
- `smalltalk` (greetings, thanks and anything off-topic)

Photos never reach it.

**Training data.** No suitable labelled dataset exists, so we build one. Training examples may be drafted with an LLM and then checked by hand. The test set is written entirely by people, in all three languages, and is never used for training or for choosing thresholds.

**Evaluation.** We report macro-F1 (the F1 score averaged equally over the four labels) per language. We compare against two baselines:

- a keyword rule baseline;
- zero-shot classification by Sarvam-105B.

The classifier earns its place only if it beats the keyword rules and comes close to the LLM at a fraction of the latency and cost. If it does not, we report that honestly.

## 8. Data

| Data | Source | Used for |
|---|---|---|
| Knowledge base | MoHFW and ICMR guidelines, NHM ASHA training modules, the ICMR Standard Treatment Workflows | Retrieval. Documents are downloaded by a script with their URLs recorded, not committed. Each chunk keeps its title, publisher, year and page |
| Medicine lexicon | National List of Essential Medicines 2022 (generic names), plus a public brand-to-generic list | Prescription extraction |
| Danger-sign list | ASHA modules, MCP card, IMNCI | Section 6 |
| Critical lab values | Cited per row (for example, anaemia cut-offs from Anemia Mukt Bharat) | Section 5.2 |
| Intent data | Built by the team | Section 7 |
| Evaluation sets | Built by the team | Section 9 |

**Datasets we dropped from the original proposal, and why:**

- **Symptom-to-disease and diagnosis-to-remedy tables** (IEEE DataPort). They exist to predict a diagnosis, which Chhaaya does not do.
- **MedQA, MedMCQA, PubMedQA** (the BELIEF collection). They are exam multiple-choice questions and look nothing like what a patient asks. Our own question set measures what matters.
- **Doctor-AI dialogue data.** There is no generation model to train, so there is no use for it.
- **Handwriting datasets** (IEEE prescription crops, RxHandBD, IAM). Handwriting goes to the ASHA instead of a model.

## 9. Evaluation

The evaluation harness runs every set below against the running system. It writes `eval/report.md` and one CSV per set, so any result in this document can be regenerated with one command. Each research question maps to a set:

| Question | Set | Size | Reported per language |
|---|---|---|---|
| **RQ1:** Are answers in Hindi, English and Hinglish accurate? | Patient-style questions, each written in all three languages | 40 × 3 | Retrieval hit@5 (was the expected source in the top 5?), answer rating, abstention rate |
| **RQ2:** Is it safe to put in front of patients? | Danger-sign messages and look-alikes without danger signs | 30 + 30 | Recall and false-alarm rate |
| | Adversarial messages (dose changes, "can I stop", diagnosis requests, prompt injection, off-topic) | 20 | Unsafe-response rate |
| **RQ3:** Can it read reports and prescriptions? | Synthetic lab reports and prescriptions | 10 + 10 | Field-level extraction accuracy, medicine-name accuracy before and after lexicon matching, critical-value escalation |
| **RQ4:** Does voice work? | Team-recorded voice questions | 15 per language | Word error rate; whether the voice path gives the same answer as the text path |
| **RQ5:** Does escalation stay manageable? | All of the above | | Share of messages escalated, per cause |
| Router | Human-written intent test set | 40 per language | Macro-F1 against both baselines |

**How answers are rated.** Two team members rate each answer independently. Each answer is rated correct or not, supported by its sources or not, safe or not, and understandable or not. Disagreements are settled by discussion, and the agreement rate is reported.

**Tuning.** τ is tuned on half of the question set and reported on the other half.

## 10. Privacy

The project uses no real patient data (section 2). Audio and images are deleted once they have been processed. Messages and case records stay in the local database, and the phone numbers in it belong to team members.

A real deployment would need consent and data handling under the Digital Personal Data Protection Act, 2023. That is outside this project.

## 11. Prior work

| System | What it does | What Chhaaya takes or changes |
|---|---|---|
| ASHABot (Khushi Baby, Microsoft Research) | WhatsApp assistant for ASHAs in Udaipur, Rajasthan: GPT-4 with retrieval over public-health documents; Hindi, English and Hinglish; voice notes; uncertain questions go to nurses. 869 ASHAs, over 24,000 messages | Same channel and grounding, but for patients, with document reading and reminders |
| CataractBot | WhatsApp assistant for cataract surgery patients at an eye hospital. Every answer is checked by an expert, and corrections go back into the knowledge base | Write-back of expert answers. Escalation is selective, not every answer, so a single ASHA can keep up |
| Kilkari / mMitra (ARMMAN) | Scheduled, pre-recorded voice calls on maternal and child health, at national scale | Shows voice reaches this population. Chhaaya is two-way and answers the patient's own question |
| Ada | Symptom checker. In an emergency-department study, 14% of its triage decisions were judged unsafe by at least two of three physicians | Chhaaya deliberately does not triage or diagnose |
| SnehAI (Population Foundation of India) | Chatbot for adolescent sexual and reproductive health | Single topic; Chhaaya covers the everyday rural-health journey |

Findings from the literature that shaped this design:

- **Grounding matters more than model size.** Med-PaLM 2 reached 86.5% on MedQA, and most of its gain came from grounding and answer refinement, not scale. It has no way to abstain, so a deployed system has to add its own deferral policy [1].
- **Non-English accuracy drops.** Medical QA accuracy falls about ten points outside English even for well-resourced European languages, and retrieval does not close the gap. So we report every metric per language and keep the corpus small and authoritative [2].
- **Identify the language first.** Mixed-language Indian speech recognition works better when the language is identified before recognition. Speech-to-text returns the language, and we use it [3].
- **One-way content has limits.** A randomised trial of Kilkari found high listening rates but no effect on its primary outcome. One-way content is the baseline an interactive system has to beat [4].
- **Human checking builds trust but must be selective.** Users trusted CataractBot because they could see experts checking it. Checking every answer depends on specialist time, which is why Chhaaya only escalates selectively [5].

References:

1. K. Singhal et al., "Toward expert-level medical question answering with large language models," *Nature Medicine* 31(3), 943-950, 2025. doi:10.1038/s41591-024-03423-7
2. I. Alonso, M. Oronoz, R. Agerri, "MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering," *Artificial Intelligence in Medicine* 155, 102938, 2024. doi:10.1016/j.artmed.2024.102938
3. S. Soni, S. Lalitha, "Effective Multilingual and Mixed-lingual DSR System for Healthcare Application in Indian Languages," *Procedia Computer Science* 258, 1219-1231, 2025. doi:10.1016/j.procs.2025.04.356
4. A. E. LeFevre et al., "The impact of a direct to beneficiary mobile communication program on reproductive and child health outcomes: a randomised controlled trial in India," *BMJ Global Health* 6(Suppl 5), e008838, 2022. doi:10.1136/bmjgh-2022-008838
5. B. Sachdeva et al., "Utility of an LLM-powered experts-in-the-loop chatbot for pre- and post-operative care of cataract surgery patients," *European Journal of Ophthalmology*, 2025. doi:10.1177/11206721251396664

## 12. Changes from the original proposal

| Original | Now | Reason |
|---|---|---|
| WhatsApp and IVR | WhatsApp only | IVR needs a paid, KYC'd telephony account |
| "All 22 languages" via IndicBERT, IndicTrans2, IndicWhisper, MuRIL | Hindi, English, Hinglish | The claim was also wrong: IndicWhisper covers 12 languages and MuRIL 17. Every language needs its own checked evaluation set |
| IndicWhisper for voice output | Saaras v3 in, Bulbul v3 out | Whisper models only transcribe; they cannot speak |
| Self-hosted AI4Bharat models | Sarvam APIs | No GPU needed; one vendor for speech, OCR and the LLM |
| Symptom-disease datasets | Rule-based danger signs | Chhaaya does not diagnose |
| Trained handwriting recognition | ASHA confirms every prescription | A misread drug name can match another real drug |
| FastAPI or Node.js; Redis and Celery; FAISS | FastAPI; one Postgres with pgvector and a job table | Fewer moving parts |
| "Notification & Escalation" / "Danger-Sign Escalation" agent | Escalation agent | One name for one thing |
| "Trained and evaluated on" the datasets | One trained model (the intent classifier); everything else evaluated on our own sets | The earlier wording described work that had not happened |
| Rural doctor ratio 1:11,082; ~70% specialist vacancy | One government allopathic doctor per 11,082 people nationally; ~80% specialist shortfall at CHCs | Corrected against the sources |
| ASHABot "50k+ questions, 11k+ ASHAs" | 869 ASHAs, 24,000+ messages | Microsoft Research's published figures |
