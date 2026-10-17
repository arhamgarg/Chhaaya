# Intent labelling guide

Every message from a patient gets exactly one of four labels. The router (design section 7) sends it to the matching agent. Photos never reach it, and neither do messages from ASHAs. Label the message as written, without guessing what happened earlier in the chat.

| Label | The patient is... | Goes to |
|---|---|---|
| `question` | asking for health information: symptoms, medicines, food, pregnancy, children, tests, vaccines, when or how to take something | Q&A agent |
| `reminder_setup` | asking Chhaaya to remind them to take a medicine, or to change or stop a reminder | Reminder agent |
| `reminder_reply` | answering a reminder that Chhaaya sent: taken, not taken, will take later | Reminder agent |
| `smalltalk` | greeting, thanking, saying goodbye, or writing something off-topic | Canned reply |

## Rules

1. **Asking to be reminded is `reminder_setup`; asking about timing is `question`.** "BP ki goli kab leni hai?" asks for information, so it is a `question`. "Roz subah 8 baje BP ki goli yaad dilana" asks for a reminder, so it is `reminder_setup`. The ask ("remind me", "yaad dilana", "alarm lagao", "set a reminder") decides, not the medicine or time words.
2. **A reply reports on a dose.** "Le li", "taken", "abhi nahi li", "will take at night" are `reminder_reply`, including when the dose was missed or skipped. The same words with a new time to be reminded at ("nahi li, 2 ghante baad yaad dilana") are `reminder_setup`.
3. **When unsure, label `question`.** The router does the same, because the Q&A agent abstains when it has no evidence.
4. **Thanks plus a question is a `question`.** "Thanks, aur ek baat batao, bukhar mein kya khayein?" asks something.
5. **Bare "ok", "haan", "yes", "no" are left out of the dataset.** Confirming a reminder read-back is handled by the conversation state, not by the classifier.
6. **Off-topic means `smalltalk`:** cricket scores, jokes, "who made you". Anything that mentions a symptom or medicine is not off-topic.
7. **Language and script do not change the label.** Hindi is in Devanagari, English in the Latin script, and Hinglish is Hindi written in the Latin script, mixed with English words.

## Borderline cases

| Message | Label | Why |
|---|---|---|
| BP ki goli kab leni hai | `question` | Asks when, not for a reminder (rule 1) |
| remind me to take my BP tablet at 8 | `reminder_setup` | Asks for a reminder |
| mujhe iron ki goli roz leni hai, kaise yaad rakhun | `reminder_setup` | "How do I remember" asks for help being reminded |
| iron ki goli ke baad ulti aa rahi hai | `question` | Describes a symptom; no reminder involved |
| le li, par chakkar aa raha hai | `question` | A symptom belongs with the Q&A agent, even inside a dose report |
| nahi li, kal se reminder band kar do | `reminder_setup` | Asks to stop the reminder |
| nahi li, kal se goli band kar du | `question` | Asks about stopping the medicine, which Chhaaya never decides (rule 3) |
| taken. thank you | `reminder_reply` | Reports the dose; the thanks is secondary |
| thank you, bahut madad mili | `smalltalk` | Only thanks |
| aap kaun ho | `smalltalk` | About the bot, not about health |
| kya main goli band kar sakti hoon | `question` | Chhaaya will not advise, but the router's job is only to send it to the Q&A agent |

## Labelling the agreement sample

Two people label the same 50 messages from `data/intent/agreement_sample.csv` on their own, without seeing each other's labels or discussing them, and save them as `data/intent/agreement/<name>.csv` with columns `id,label`. Run the agreement command from the README; the PR reports Cohen's kappa. Settle disagreements only after the score is recorded, and add the case to the table above when it shows a rule is unclear.
