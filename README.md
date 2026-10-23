# Chhaaya

A Hindi, English and Hinglish health assistant with guideline-grounded answers, lab-report reading, ASHA-confirmed prescriptions and medicine reminders. Urgent or unsupported questions go to the ASHA. Use synthetic data only; this is not a clinical service.

Claude Sonnet 5.5 handles text and photos. Sarvam is a manual fallback; speech is off by default. A fine-tuned MuRIL classifier routes messages, and local BGE-M3 embeddings retrieve guidelines. MuRIL is the only model we train.

## Run

Use Python 3.12, Docker, a configured `.env`, and the saved checkpoint in `data/intent/model/`. Then:

```bash
uv sync
DEMO_DATABASE_URL=postgresql://chhaaya:demo@127.0.0.1:5433/chhaaya_demo \
  HF_HUB_OFFLINE=1 uv run python -m chhaaya.demo
```

Open the printed link. The browser runs the patient and ASHA flows without sending WhatsApp messages. Select **Models** for saved test metrics and explanations.

```bash
uv run python -m chhaaya.model_metrics   # evaluate the saved checkpoint and cache SHAP
uv run python -m chhaaya.model_report    # update README and the evaluation report
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

<!-- model-evidence:start -->
## Model performance

Held-out intent test set: 120 messages. MuRIL is the only model trained here; keywords and Sarvam are comparison baselines. Bold marks the best score in each column, including ties. P = precision, R = recall. AUC uses one-vs-rest class probabilities; saved hard-label baselines have no AUC.

| Model | Accuracy | P macro | P weighted | R macro | R weighted | F1 macro | F1 weighted | AUC macro | AUC weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fine-tuned MuRIL (900 training rows) | 0.908 | 0.911 | 0.911 | 0.908 | 0.908 | 0.907 | 0.907 | **0.983** | **0.983** |
| Keyword rules | 0.475 | 0.581 | 0.581 | 0.475 | 0.475 | 0.380 | 0.380 | N/A | N/A |
| Sarvam-105B zero-shot | **0.967** | **0.968** | **0.968** | **0.967** | **0.967** | **0.967** | **0.967** | N/A | N/A |

## Explainability

MuRIL uses SHAP Partition with token masking. The UI shows cached global importance and local explanations for held-out or entered text. See the [worked example and figures](eval/report.md#explainability). These explain routing, not clinical decisions.
<!-- model-evidence:end -->

<details>
<summary>First-time setup</summary>

Save the provider keys in the ignored `.env`. Claude needs `ANTHROPIC_API_KEY`; Sarvam is optional. Reuse the saved MuRIL checkpoint, or train it once with `uv run python -m chhaaya.intent train`. Do not train on the test set.

```bash
cp -n .env.example .env
docker run -d --name chhaaya-browser-demo-db -e POSTGRES_USER=chhaaya \
  -e POSTGRES_PASSWORD=demo -e POSTGRES_DB=chhaaya_demo \
  -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16
export DEMO_DATABASE_URL=postgresql://chhaaya:demo@127.0.0.1:5433/chhaaya_demo
DATABASE_URL=$DEMO_DATABASE_URL uv run alembic upgrade head
uv run python -m chhaaya.corpus
DATABASE_URL=$DEMO_DATABASE_URL uv run python -m chhaaya.retrieval
uv run python -m chhaaya.lexicon
```

For WhatsApp, fill `WHATSAPP_*` and `POSTGRES_PASSWORD` in `.env`, then run `docker compose up --build`. Voice additionally needs a Sarvam key, ffmpeg and `SPEECH_ENABLED=true`; the browser always keeps speech off.

</details>

[Full evaluation and dataset provenance](eval/report.md) · [Saved metrics](data/intent/model_evaluation.json) · [Intent dataset on IEEE DataPort](https://doi.org/10.21227/m81y-kj83)

AGPL-3.0. See [LICENSE](LICENSE).
