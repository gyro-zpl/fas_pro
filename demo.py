from agents import Agent, WebSearchTool, function_tool
import datetime
import db


@function_tool

def save_results(output):
    db.insert({"output": output, "time": datetime.datetime.now()})
    return "File saved"


search_agent = Agent(
    name= "search_agent",
    instructions="You are a search agent that can search the web for information.",
    tools=[WebSearchTool(),save_results],
)

result = search_agent.run("What is the capital of France?")
print(result)