from langchain_openai import ChatOpenAI   # Remove ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()

class Classifier:
    def __init__(self, model_name="gpt-4", temperature=0.0):
        # Use OpenAI instead of ChatOpenAI
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key =os.getenv('OPENAI_API_KEY'))

    def classify_zero_shot(self, prompt_template):
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(input_variables=["review"], template=prompt_template))
        return chain

    def classify_one_shot(self, prompt_template):
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(input_variables=["review"], template=prompt_template))
        return chain

    def classify_few_shot(self, prompt_template):
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(input_variables=["review"], template=prompt_template))
        return chain
