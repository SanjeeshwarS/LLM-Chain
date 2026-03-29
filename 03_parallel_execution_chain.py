from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables  import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')


#FLOW 1
user_input = input("Enter The Reason For Taking Leave: ")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior analyst assistant, renowned for providing comprehensive context, background, and nuances in all responses. Before providing direct answers, first analyze the user's query for implicit needs, supply necessary context or historical background, and explain the rationale. If the query is ambiguous, you must ask clarifying questions to fill in missing information. When providing answers, structure them to highlight the 'what', 'why', and 'how'."),
    ("user", "{topic}")
])

llm_ollama = ChatOllama(model="llama3.1:8b", temperature=0, num_gpu = 100)

strng_parser = StrOutputParser()

def dict_maker(text:str)-> dict:
    return {"text" : text}

dict_runnable = RunnableLambda(dict_maker)

chain1 = (
    prompt
    | llm_ollama
    | strng_parser
    | dict_runnable
)

#FLOW 2
systemhr_prompt = """
You are a professional corporate employee writing a formal leave request email to the Human Resources (HR) department.

Your task is to draft a clear, concise, and strictly professional leave request.

Guidelines:
- Use a formal, official, and respectful tone
- Follow proper business email structure
- Clearly mention:
  - Reason for leave (brief and professional)
  - Leave start date and end date
  - Total number of leave days
- Ensure the wording is neutral and appropriate for official records
- Do not include casual or overly friendly language
- Avoid unnecessary details
- Keep the content precise and to the point

Structure:
- Subject line (formal and specific)
- Salutation (e.g., Dear HR Team)
- Purpose of the email
- Leave details (dates and reason)
- Optional: mention compliance with company policy
- Closing with gratitude
- Professional sign-off

Output must be:
- Formal and polished
- Grammatically correct
- Suitable for official documentation
- Ready to send without edits

If any details are missing, make reasonable professional assumptions.

DONT USE HTML TAGS
"""

hr_prompt = ChatPromptTemplate.from_messages([
    ("system",systemhr_prompt),
    ("user","This is the context : {text}")
])

chain_hr = (
    hr_prompt
    | llm_ollama
    | strng_parser
)

#FLOW 3 Using Func 
def TeamLead(text: str):

    systemtl_prompt =  """

    You are a responsible employee writing a leave request message to your Team Leader.

    Your task is to draft a professional yet slightly friendly and human-centered leave request.

    Guidelines:
    - Maintain a respectful but approachable tone
    - Clearly communicate:
    - Reason for leave (simple and honest)
    - Leave dates (start and end)
    - Emphasize work responsibility:
    - Mention task status or progress
    - Mention handover or delegation if needed
    - Show accountability and team awareness
    - Offer availability for urgent issues if appropriate
    - Avoid overly formal/legal tone
    Structure:
    - Subject line (clear but not overly rigid)
    - Greeting (e.g., Dear [Team Leader’s Name])
    - Purpose of the message
    - Leave details (dates + reason)
    - Work status / handover plan
    - Optional: availability during leave
    - Closing with appreciation
    - Friendly professional sign-off

    Output must be:
    - Clear and engaging
    - Professional but not stiff
    - Supportive and responsible in tone
    - Ready to send without edits

    If details are missing, make logical workplace assumptions.

    """


    tl_prompt = ChatPromptTemplate.from_messages([
        ("system",systemtl_prompt),
        ("user","This is the context : {text}")
    ])

    chain_tl = (
        tl_prompt
        | llm_ollama
        | strng_parser
    )

    result = chain_tl.invoke(text)

    return result

TeamLead_runnable = RunnableLambda(TeamLead)


Link_chain = (
    chain1
    |  RunnableParallel(branches = {"Hr Letter" : chain_hr,"Team Lead Letter": TeamLead_runnable })
    
)

def beautify(Final_response: str)-> dict:
    Hr_Letter = Final_response["branches"]["Hr Letter"]
    Tl_Letter = Final_response["branches"]["Team Lead Letter"]

    return f"\n{'='*40}\n🏢 HR LEAVE REQUEST\n{'='*40}\n{Hr_Letter}\n\n{'='*40}\n🤝 TEAM LEAD MESSAGE\n{'='*40}\n{Tl_Letter}\n"

beautify_runnable = RunnableLambda(beautify)

Final_chain = (
    Link_chain
    | beautify_runnable
)
result = Final_chain.invoke({"topic" : user_input})

print(result)