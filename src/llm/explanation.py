from langchain_openai import ChatOpenAI

_llm: ChatOpenAI | None = None


def get_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=model)
    return _llm


def generate_explanation(query: str, description: str) -> str:
    prompt = (
        f"User is looking for: {query}\n\n"
        f"Book description:\n{description}\n\n"
        "Explain in 1 short sentence why this book is a good recommendation."
    )
    return get_llm().invoke(prompt).content
