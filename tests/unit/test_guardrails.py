import pytest

from rag_chatbot.chat.ChatCopy import ChatCopy
from rag_chatbot.guardrails.OutputGuard import OutputGuard
from rag_chatbot.guardrails.QuestionGuard import QuestionGuard

OPEN_TO_WORK_ANSWERS = [
    "Alex is open to work and would welcome a call.",
    "He is currently looking for a new role in applied AI.",
    "He is available for hire starting next month.",
    "Alex is seeking new opportunities in data science.",
    "You should hire him for your team.",
    "He would be a great fit for your company.",
    "Feel free to reach out about a role.",
    "His next role could be in a larger team.",
    "His salary expectations are not listed.",
]
CONTACT_ANSWERS = [
    "You can call him at (585) 555-0123.",
    "His number is 585.555.0123.",
    "Email him at someone.name@gmail.com for details.",
]


@pytest.mark.parametrize("answer", OPEN_TO_WORK_ANSWERS)
def test_output_guard_blocks_open_to_work(answer):
    result = OutputGuard().review(answer)
    assert not result.allowed
    assert result.reason == "open_to_work"
    assert result.text == ChatCopy.BLOCKED


@pytest.mark.parametrize("answer", CONTACT_ANSWERS)
def test_output_guard_blocks_contact_details(answer):
    result = OutputGuard().review(answer)
    assert not result.allowed
    assert result.reason == "contact_details"


def test_output_guard_passes_normal_answers_and_tidies_them():
    result = OutputGuard().review(
        "He studied at the Syracuse iSchool \u2014 2019\u20132022 \U0001f680 and built tools."
    )
    assert result.allowed
    assert "iSchool" not in result.text and "Syracuse University" in result.text
    assert "\u2014" not in result.text and "\u2013" not in result.text
    assert "\U0001f680" not in result.text


def test_output_guard_allows_ordinary_work_language():
    answer = "He built a forecasting platform and looked for patterns in loyalty data. The code is available on GitHub."
    assert OutputGuard().review(answer).allowed


@pytest.mark.parametrize(
    "question,reason",
    [
        ("Is he open to work?", "job_status"),
        ("Is Alex available for a new role?", "job_status"),
        ("What salary does he expect?", "job_status"),
        ("Why should we hire him?", "job_status"),
        ("What is his phone number?", "contact"),
        ("Can I get his email?", "contact"),
        ("", "empty"),
        ("x" * 600, "too_long"),
    ],
)
def test_question_guard_declines_off_limits_questions(question, reason):
    result = QuestionGuard(max_chars=500).check(question)
    assert not result.allowed
    assert result.reason == reason


PROJECT_QUESTIONS = [
    "What did he build at Signature Aviation?",
    "How does the news topic explorer work?",
    "What is his experience with demand forecasting?",
    "Which tools and languages does he use?",
    "Where did he study?",
]


def test_question_guard_allows_project_questions():
    for question in PROJECT_QUESTIONS:
        assert QuestionGuard().check(question).allowed
