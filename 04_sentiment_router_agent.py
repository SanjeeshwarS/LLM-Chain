from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables  import RunnableLambda, RunnableParallel,  RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')

#FLOW 1
llm_ollama = ChatOllama(model = "llama3.1:8b", temperature=0, num_gpu = 100)

user_input = input("Enter Movie Name : ")
user_review = input("Enter Your Review : ")

strng_parser = StrOutputParser()

#Structuring OutPut 
class MovieReviewOutput(BaseModel):
    review_summary: Literal["Positive", "Negative", "Neutral"]

llm_structured_output = llm_ollama.with_structured_output(MovieReviewOutput)


classifier_sysprompt = ChatPromptTemplate.from_messages([
    ("system", 
     "This is the Movie {topic}"
     "You are an expert movie review sentiment classifier.\n"
     "Your task is to analyze the sentiment of a movie review.\n\n"
     
     "Rules:\n"
     "- Classify the sentiment strictly as either 'Positive' or 'Negative'.\n"
     "- Do NOT return anything except the required structured output.\n"
     "- Do NOT explain your answer.\n"
     "- Base your decision on overall tone, emotions, and opinion.\n"
     "- If the review is mixed, choose the dominant sentiment.\n"
     
     "Output format:\n"
     "review_summary: 'Positive' or 'Negative' or 'Neutral'"
    ),
    
    ("user", "Classify The Review For the Movie as Positive or Negative or Neutral : {review}")
])


def parse_MovieReviewSummary(input: MovieReviewOutput)-> str:
    return input.model_dump()["review_summary"]

pydantic_runnable = RunnableLambda(parse_MovieReviewSummary)

classifier_chain = (
    classifier_sysprompt
    | llm_structured_output
    | pydantic_runnable
)


#FLOW 2

positive_sysprompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are the ultimate cinematic hype-man and a world-class movie buff. 
    The user just watched a movie and loved it. Your job is to validate their opinion and match their energy!
    
    Guidelines:
    - Be highly enthusiastic, energetic, and engaging.
    - Agree with their specific points from the review.
    - Validate their great taste in cinema.
    - Keep it concise (2-3 short paragraphs max).
    - Drop 2 or 3 relevant emojis to keep the vibe fun.
    - Do NOT sound like a robotic AI. Sound like a passionate movie fan talking to a friend.
    """),
    ("user", "Movie: {topic}\nMy Review: {review}")
])


chain_1 = (
    positive_sysprompt
    | llm_ollama
    | strng_parser
)


#FLOW 3

def ModelNegReview(inputs: dict):

    negative_sysprompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a hilariously harsh, razor-sharp movie critic. 
        The user just watched a movie, hated it, and left a negative review. Your job is to commiserate with them and roast the movie!
        
        Guidelines:
        - Be witty, sarcastic, and dramatically disappointed in the film.
        - Agree with the user's critiques and amplify them.
        - Make a joke about how they deserve a refund or a written apology from the director.
        - Keep it concise (2-3 short paragraphs max).
        - Drop a few funny/dramatic emojis (like 💀, 🗑️, 📉, 🥱).
        - Do NOT defend the movie. You are on the user's side 100%.
        """),
        ("user", "Movie: {topic}\nMy Review: {review}")
    ])

    chain_2  = (
        negative_sysprompt
        | llm_ollama
        | strng_parser
    )

    result = chain_2.invoke({
        "topic": inputs["text"],
        "review": inputs["review"]
    })

    return result

negative_custom_runnable = RunnableLambda(ModelNegReview)

#FLOW 4
neutral_sysprompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a perfectly balanced, highly analytical movie reviewer. 
    The user just watched a movie and thought it was completely average (neither great nor terrible). 
    Your job is to agree with their middle-of-the-road take!
    
    Guidelines:
    - Be calm, diplomatic, and realistic.
    - Acknowledge that not every movie is a masterpiece, and sometimes a "just okay" popcorn flick is totally fine.
    - Keep it concise (2-3 short paragraphs max).
    - Drop a couple of perfectly neutral emojis (like 🤷‍♂️, 🍿, ⚖️, 😐).
    """),
    ("user", "Movie: {topic}\nMy Review: {review}")
])

chain_3 = (
    neutral_sysprompt
    | llm_ollama
    | strng_parser  
)


router = RunnableBranch(
    (lambda x: x["sentiment"] == "Positive", chain_1),
    (lambda x: x["sentiment"] == "Negative", negative_custom_runnable),
    (lambda x: x["sentiment"] == "Neutral", chain_3),
    chain_3 # <-- Fallback Chain (default chain)
)

main_chain = (
    RunnablePassthrough.assign(sentiment = classifier_chain)
    | router
)

print("\n🤖 Analyzing Review & Routing...\n" + "="*50)

final_output = main_chain.invoke({"topic": user_input, "review": user_review})

print(final_output)