from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq
from rag_eval_poc.src import config


class GroqModel(DeepEvalBaseLLM):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.load_model()   # initialize client

    def load_model(self):
        """Required by DeepEval"""
        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature= config.EVAL_TEMPERATURE
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name