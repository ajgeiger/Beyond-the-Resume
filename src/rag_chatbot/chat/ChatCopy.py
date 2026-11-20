"""Every fixed reply the chatbot gives, kept in one place so the tone stays consistent."""


class ChatCopy:
    """Plain, understated wording for answers that skip the language model."""

    NOT_COVERED = (
        "I don't have details on that, but I can tell you about Alex's resume, "
        "his work at Signature Aviation and earlier roles, and his projects such as Headline Radar "
        "and the News Topic Explorer."
    )
    GREETING = "Thanks for taking the time to visit. Alex would be glad to connect through LinkedIn."
    DECLINED = (
        "I only describe Alex's projects and experience, so questions about his job plans or pay are "
        "outside what I cover. You could ask what he built at Signature Aviation instead."
    )
    CONTACT = "The best way to reach Alex is through the LinkedIn and GitHub links on his site."
    BLOCKED = (
        "I could not give a reliable answer to that from the material I have. "
        "Try asking about a specific project or role."
    )
    BUDGET_REACHED = (
        "The assistant has reached its usage limit for this month. The resume and project pages "
        "on the site cover the same material."
    )
    TOO_LONG = "Please keep questions under {limit} characters."
    EMPTY = "Please type a question."
    PROVIDER_ERROR = "The assistant could not answer just now. Please try again in a moment."
