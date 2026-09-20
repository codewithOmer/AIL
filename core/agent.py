from interfaces.llm import LLM


class MockLLM(LLM):
    """Temporary model used to test AIL architecture."""

    def generate(self, prompt: str) -> str:
        return f"AIL received: {prompt}"


class Agent:
    """AIL's initial reasoning/orchestration layer."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def respond(self, user_input: str) -> str:
        return self.llm.generate(user_input)