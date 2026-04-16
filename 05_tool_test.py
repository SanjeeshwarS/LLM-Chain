from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage

# 1. Define the tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

# 2. Initialize the LLM and bind tools
llm = ChatOllama(model="llama3.1:8b", temperature=0)
tools = [multiply]
llm_with_tools = llm.bind_tools(tools)

# 3. Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to perform calculations."),
    ("human", "{input}"),
])

# 4. Define the chain
chain = prompt | llm_with_tools

# 5. Execute the chain
user_input = "What is 123 times 456?"
print(f"Input: {user_input}")

# Invoke the chain
ai_msg = chain.invoke({"input": user_input})

print("\n--- LLM Response ---")
print(f"Tool Calls: {ai_msg.tool_calls}")

# 6. Step-by-step Tool Execution (Optional but helpful for testing)
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        # Actually execute the tool
        selected_tool = {"multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        
        print(f"\nTool Execution ({tool_call['name']}):")
        print(f"Result: {tool_output}")
        
        # We could then feed this back to the LLM if we wanted a final natural language response
else:
    print(ai_msg.content)
