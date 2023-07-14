import os
import re
from typing import List, Union
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType, LLMSingleActionAgent, AgentOutputParser
from langchain.tools import YouTubeSearchTool
from langchain import SerpAPIWrapper, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, HumanMessage

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
SERPAPI_API_KEY = os.environ['SERPAPI_API_KEY']

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
)

vectorstore = FAISS.load_local("arxiv_vectorstore", embeddings=embed)

llm = OpenAI(model_name="text-davinci-003", temperature=0)

papers_chunks_tool = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)
search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
youtube_tool = YouTubeSearchTool()

tools = [
    Tool(
        name="FAISS QA System",
        func=papers_chunks_tool.run,
        description="useful to answer questions about LLMs and artificial intelligence. use this more than the other tool if the question is related to artificial intelligence and/or LLMS",
    ),
    Tool(
        name = "Search",
        func=search_tool.run,
        description="useful for when you need to look for some resources such as web articles and educational courses"
    ),
    Tool(
        name = "Youtube Search",
        func=youtube_tool.run,
        description="useful for finding educational videos"
    )
]

# Set up the base template
template = """Complete the objective as best you can. You are an AI agent designed to help users study the topic of LLMs. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

After finding the answer to the original question, search for a related web article.
After finding the web article, search for one related Youtube video's url link. Output the original question's answer and the Youtube urls.
After having both the answer to the question and the additional resources, output the answer and the resources.

These were previous tasks you completed:


Begin!

Question: {user_input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["user_input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names,
    # verbose=True,
    return_intermediate_steps=False
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

answer_ = agent_executor.run("What are LLMs?")

print(answer_)