from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai
import replicate

from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMWrapper(ABC):
    @abstractmethod
    def generate_response(self, prompt):
        """
        Generates a response to the given prompt.

        :param prompt: The input prompt to generate a response for.
        :return: The generated response as a string.
        """
        pass

class GptApi(LLMWrapper):

    def __init__(self, api_key, model):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt):

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a teacher, skilled at explaining complex AI decisions to general audiences."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return completion.choices[0].message.content
    
class GeminiAPI(LLMWrapper):

    def __init__(self, api_key, model="gemini-pro"):
        self.model = model
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def generate_response(self, prompt):

        return self.client.generate_content(prompt).text

# Not advised to use
class HfGemma(LLMWrapper):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

    def generate_response(self, prompt, max_tokens=1500):
        input_ids = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(**input_ids, max_new_tokens=max_tokens)
        print(self.tokenizer.decode(outputs[0]))

class LlamaAPI():
    def __init__(self, api_key, model="meta/meta-llama-3-70b-instruct"):
        self.api_key = api_key
        self.model = model
        self.client = replicate.Client(api_key)

    def generate_response(self, prompt, max_tokens=512):
        output = self.client.run(
            self.model,
            input={
                "top_p": 0.9,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "min_tokens": 0,
                "temperature": 0.6,
                "prompt_template": "system\n\nYou are a helpful assistantuser\n\n{prompt}assistant\n\n",
                "presence_penalty": 1.15,
                "frequency_penalty": 0.2
            }
        )
        response_text = ""
        for item in output:
            response_text += str(item)
        return response_text 