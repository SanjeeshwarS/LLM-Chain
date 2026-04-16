from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel , Field


user_input = input("Enter your project description : ")

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a professional and intelligent Ai assistant who has a great knowledge in technology and who helps Users on project guiding"),
    ("user", "This is the {project_discription}  give step by step guide on it and the needed Tech Stack and Tech Skills ")
])

class ProjectGuide(BaseModel):
    project_name : str = Field(description = "Name of the Project")
    difficulty : str = Field(description = "Just One Word - Beginner, Intermediate, Advanced")
    core_tools: list[str] = Field(description="A list of the core technologies, libraries, or languages used")
    description : str = Field(description = "Give A one line description of the project. YOU MUST include Emoji in the description")
    step_guide : list[str] = Field(description = "Step by step guide on the project. YOU MUST represent each line with numbers and dots like 1. , 2. , 3. ")

llm_ollama = ChatOllama(model = "llama3.1:8b", temperature = 0, num_gpu = 100)

llm_structured_output = llm_ollama.with_structured_output(ProjectGuide)

chain = prompt | llm_structured_output

result = chain.invoke({"project_discription" : user_input})

print("\n--- YOUR PROJECT BLUEPRINT ---")
print(f"Name: {result.project_name}")
print(f"Difficulty: {result.difficulty}")
print(f"Tools: {', '.join(result.core_tools)}") # This joins your list into a nice clean string!
print(f"Description: {result.description}")

print("\n--- STEP BY STEP GUIDE ---")
# NEW: Loop through the list to print each step on its own line
for step in result.step_guide:
    print(step)
