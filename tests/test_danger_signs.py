"""Danger-sign matcher.

Ways it can fail, each covered below:

- a sign phrased in Hindi, English or Hinglish, or described indirectly, is missed;
- case, Unicode forms (precomposed or decomposed nukta, zero-width joiners,
  full-width letters) or punctuation stop a phrase matching;
- a common Hinglish spelling variant ("nhi", "dudh", "rha", "bacha") is missed;
- a sign split across words ("pet mein bahut tez dard") or across a full stop
  is missed;
- a negated mention ("no bleeding", "khoon nahi aa raha") fires, or a negation
  word in the sign itself ("doodh nahi pee raha", "not breathing") stops it
  firing, or a negation of something else nearby hides a real sign;
- a harmless look-alike ("I am fit", "khoon ki jaanch") fires;
- an entry in the list has no source or is missing a language, or a listed
  phrasing does not match its own sign after normalisation.
"""

import pytest

from chhaaya.danger_signs import SIGNS, URGENT_REPLY, detect

LANGUAGES = {"en", "hi", "hi-Latn"}


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("My baby is not feeding since morning", "not_feeding"),
        ("bachcha doodh nahi pee raha", "not_feeding"),
        ("बच्चा दूध नहीं पी रहा है", "not_feeding"),
        ("I am 7 months pregnant and bleeding", "bleeding_in_pregnancy"),
        ("pregnancy mein khoon aa raha hai", "bleeding_in_pregnancy"),
        ("गर्भावस्था में खून आ रहा है", "bleeding_in_pregnancy"),
        ("usko jhatke aa rahe hain", "convulsions"),
        ("मेरे बच्चे को दौरे पड़ रहे हैं", "convulsions"),
        ("She had a fit an hour ago", "convulsions"),
        ("baby is breathing very fast", "difficulty_breathing"),
        ("saans lene mein dikkat ho rahi hai", "difficulty_breathing"),
        ("उसे सांस लेने में तकलीफ है", "difficulty_breathing"),
        ("my mother is unconscious", "unconscious"),
        ("wo behosh ho gayi", "unconscious"),
        ("दादी बेहोश हो गई हैं", "unconscious"),
        ("delivery ke baad bahut khoon beh raha hai", "bleeding_after_delivery"),
        ("bacche ki hatheli aur talve peele hain", "jaundice"),
        ("baby is vomiting everything she eats", "vomits_everything"),
        ("bachcha doodh pi nahi raha", "not_feeding"),
        ("बच्चा दूध पी नहीं रहा", "not_feeding"),
        ("bacha dudh pee nahi rha hai", "not_feeding"),
        ("baby does not feed", "not_feeding"),
        ("baby is not taking milk", "not_feeding"),
        ("jhatke aaye", "convulsions"),
        ("baby ko jhatke aate hain", "convulsions"),
        ("usko daura pada", "convulsions"),
        ("pregnancy mein khoon aaya", "bleeding_in_pregnancy"),
        ("गर्भावस्था में खून आया", "bleeding_in_pregnancy"),
    ],
)
def test_detects_sign_in_each_language(text, sign):
    assert sign in detect(text)


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("BABY IS NOT FEEDING!!!", "not_feeding"),
        ("ｂａｂｙ ｉｓ ｎｏｔ ｆｅｅｄｉｎｇ", "not_feeding"),
        # Precomposed ड़ (U+095C) and the decomposed ड + nukta both match.
        ("मेरे बच्चे को दौरे प\u095c रहे हैं", "convulsions"),
        ("मेरे बच्चे को दौरे पड़ रहे हैं", "convulsions"),
        ("बच्चा दूध न‍हीं पी रहा", "not_feeding"),
        ("bacha dudh nhi pi rha", "not_feeding"),
        ("bachha doodh nahin pee rahaaa", "not_feeding"),
        ("wo behosh ho gyi", "unconscious"),
        ("saas lene me dikat", "difficulty_breathing"),
    ],
)
def test_normalises_case_unicode_and_spelling(text, sign):
    assert sign in detect(text)


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("pregnancy ke saatve mahine mein khoon aa raha", "bleeding_in_pregnancy"),
        ("I'm pregnant. Since last night, bleeding", "bleeding_in_pregnancy"),
        ("bachcha doodh bilkul nahi pee raha", "not_feeding"),
        (
            "pregnancy mein khoon dekha, haan wahi khoon bahut zyada tez aa raha hai",
            "bleeding_in_pregnancy",
        ),
        ("pregnancy mein bleeding nahi ruk rahi", "bleeding_in_pregnancy"),
    ],
)
def test_detects_sign_split_across_words(text, sign):
    assert sign in detect(text)


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("I am pregnant, no bleeding so far", "bleeding_in_pregnancy"),
        ("pregnancy mein khoon nahi aa raha", "bleeding_in_pregnancy"),
        ("pregnancy mein koi bleeding nahi hai", "bleeding_in_pregnancy"),
        ("गर्भावस्था में खून नहीं आया", "bleeding_in_pregnancy"),
        ("she did not have any fits", "convulsions"),
        ("jhatke nahi aaye", "convulsions"),
    ],
)
def test_negated_mention_does_not_fire(text, sign):
    assert sign not in detect(text)


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("no fever, but she is unconscious", "unconscious"),
        ("bukhar nahi hai lekin wo behosh hai", "unconscious"),
        ("I don't know why she is bleeding, I am pregnant", "bleeding_in_pregnancy"),
        ("baby is not breathing", "difficulty_breathing"),
        ("baby, not moving since morning", "reduced_fetal_movement"),
        (
            "main pregnant hoon, tabiyat theek nahi hai, khoon aa raha hai",
            "bleeding_in_pregnancy",
        ),
        (
            "I am pregnant and not feeling well, bleeding since morning",
            "bleeding_in_pregnancy",
        ),
    ],
)
def test_negation_elsewhere_does_not_hide_sign(text, sign):
    assert sign in detect(text)


@pytest.mark.parametrize(
    "text",
    [
        "I am fit and healthy",
        "pregnancy mein khoon ki jaanch kab karani hai",
        "delivery ke baad khoon ki kami ho gayi",
        "bachcha doodh pi raha hai, koi dikkat nahi",
        "बच्चा दूध पी रहा है, कोई दिक्कत नहीं",
        "meri saas ko sugar hai",
        "baby is feeding well and sleeping",
        "breathing exercises in pregnancy are good?",
        "बच्चा दूध पी रहा है",
        "no postpartum haemorrhage was found",
        "the baby has no poor feeding",
        "नवजात का शरीर पीला नहीं है",
        "",
    ],
)
def test_harmless_look_alike_does_not_fire(text):
    assert detect(text) == []


def test_every_sign_is_sourced_and_phrased_in_every_language():
    assert SIGNS
    for sign in SIGNS:
        assert "p." in sign.source, sign.id
        assert set(sign.phrases) == LANGUAGES, sign.id
        assert all(sign.phrases[lang] for lang in LANGUAGES), sign.id


def test_every_phrase_matches_its_own_sign():
    for sign in SIGNS:
        for phrases in sign.phrases.values():
            for phrase in phrases:
                assert sign.id in detect(phrase), (sign.id, phrase)


def test_sign_ids_are_unique():
    ids = [sign.id for sign in SIGNS]
    assert len(ids) == len(set(ids))


def test_urgent_reply_in_every_language_names_108():
    assert set(URGENT_REPLY) == LANGUAGES
    for text in URGENT_REPLY.values():
        assert "108" in text
