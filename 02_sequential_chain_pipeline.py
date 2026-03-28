from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel , Field
from langchain_core.runnables import RunnableLambda
import sys
sys.stdout.reconfigure(encoding='utf-8')

class LinkdinPostOutput(BaseModel):
    title : str = Field(description = "Title of the Post")
    content : str = Field(description = "Content of the Post. YOU MUST use plain text and standard bullet points. NO HTML tags allowed.")
    hashtags : list[str] = Field(description = "List of Hashtags")
    emoji_reaction : str = Field(description = "A single emoji representing the post's vibe (e.g., 🚀, 💡, 🔥)  ")



llm_ollama = ChatOllama(model = "llama3.1:8b", temperature = 0.2, num_gpu = 100)

user_input = input("Enter Your Linkdin Post Topic: ")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a very helpful Ai Assistant"),
    ("user", "Generate a clear discription for {topic}")
])

strng_parser = StrOutputParser()

def dict_maker(text:str)-> dict:
    return {"text" : text}

dict_runnable = RunnableLambda(dict_maker)

Final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Professional Linkdin Post Writer and a great AI with a Tech Humor"),
    ("user", "Generate a Linkdin Post With the given - {text} context")
])


llm_structured_output = llm_ollama.with_structured_output(LinkdinPostOutput)

chain = (
    prompt
    | llm_ollama
    | strng_parser
    | dict_runnable
    | Final_prompt
    | llm_structured_output
)

result = chain.invoke({"topic" : user_input})

print("\n--- YOUR LINKDIN POST ---")
print(f"Title: {result.title}")
print(f"Content: {result.content}")
print(f"Hashtags: {', '.join(result.hashtags)}")
print(f"Emoji Reaction: {result.emoji_reaction}")