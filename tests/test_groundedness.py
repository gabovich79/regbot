from services.groundedness import build_premise, score_answer_groundedness


def test_build_premise_concatenates_verbatim_evidence():
    evidence = [
        {"raw_text": "סכומים שמשך עובד מחשבונו"},
        {"raw_text": "הפניה לסעיף 121"},
    ]

    assert build_premise(evidence) == "סכומים שמשך עובד מחשבונו\n\nהפניה לסעיף 121"


def test_score_answer_groundedness_passes_premise_and_hypothesis_to_predictor():
    calls = []

    def fake_predict(pairs):
        calls.append(pairs)
        return [0.42]

    evidence = [{"raw_text": "שש שנים"}]
    result = score_answer_groundedness("שלוש שנים", evidence, fake_predict)

    assert calls == [[("שש שנים", "שלוש שנים")]]
    assert result["groundedness_score"] == 0.42
