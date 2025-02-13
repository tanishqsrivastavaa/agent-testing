from smolagents import SmolAgent, HfApiModel


class DuckDuckGoSearchTool(Tool):
    def run(self,query : str) -> str:
        return f'Results for : {query}'
    

agent = SmolAgent(tools=[DuckDuckGoSearchTool()])

prompt = input("Enter a prompt:")
response = agent.run(prompt)

print(response)
