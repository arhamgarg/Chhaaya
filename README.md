# Chhaaya

Chhaaya is a WhatsApp health assistant for rural patients in India. You send it a voice note, a message or a photo, in Hindi, English or Hinglish. It answers from Indian public-health guidelines, explains your lab report in plain words, reminds you to take your medicines, and passes anything urgent or uncertain to your ASHA worker. It does not diagnose.

The project is at the design stage and there is nothing to run yet. [docs/design.md](docs/design.md) is the specification, and the open issues are the work plan.

## How it works

A message arrives through the WhatsApp Cloud API and is queued in Postgres. A worker then:

1. turns voice into text with Sarvam's speech-to-text;
2. checks the message against a fixed list of danger signs;
3. routes it with a small intent classifier we fine-tune from MuRIL.

Questions are answered by retrieving passages from MoHFW, ICMR and NHM documents and having an LLM answer only from those passages. Any answer that cites nothing it was given is thrown away. Lab reports are read with OCR and compared against the ranges printed on them. Prescriptions are confirmed by the ASHA before the patient hears anything about them. Replies go back as text and as a voice note.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

AGPL-3.0. See [LICENSE](LICENSE).
