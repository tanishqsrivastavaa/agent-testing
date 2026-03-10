from pyexpat import model
from pydantic_ai import Agent, providers
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from settings import settings
import re

GROQ_API_KEY= settings.GROQ_API_KEY

react_agent = Agent(model=GroqModel(model_name="llama-3.3-70b-versatile",provider=GroqProvider(api_key=GROQ_API_KEY)))



@react_agent.tool_plain
def get_order_status(order_id:str):
    database = {
        "ORD-123":"Delivered",
        "ORD-456":"Undelivered"
    }

    return database.get(order_id,"Order ID not found")


SYSTEM_PROMPT = """
You operate in a loop of Thought, Action, Observation.
Use Thought to describe your plan.
Use Action to run a tool. Format: Action: tool_name[input]
Available tools: get_order_status[order_id]

Example:
Thought: I need to check the status of order ORD-123.
Action: get_order_status[ORD-123]
Observation: Delivered
Final Answer: The order has been delivered.
"""

def run_agent(user_query):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": user_query}]
    
    for _ in range(5):
        
        llm_output = call_llm(messages)
        print(f"LLM: {llm_output}")

        if "Final Answer:" in llm_output:
            return llm_output


        match = re.search(r"Action: (\w+)\[(.*?)\]", llm_output)
        if match:
            tool_name, tool_input = match.groups()
            

            if tool_name == "get_order_status":
                observation = get_order_status(tool_input)
                

                obs_str = f"Observation: {observation}"
                print(obs_str)
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({"role": "user", "content": obs_str})

user_query=input("Enter your query\n")
result = run_agent(user_query)
print(result)