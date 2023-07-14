import streamlit as st
import openai
import os
import re
from typing import List, Union
import openai
#from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import FAISS
from langchain.agents import load_tools
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import initialize_agent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.tools import YouTubeSearchTool
from langchain import  LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.tools import BraveSearch
from langchain.memory import ConversationBufferWindowMemory

#load_dotenv(find_dotenv())
openai.api_key = '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'
SERPAPI_API_KEY = 'b63d40b3acabad62185803ff03dc52738ab3748e56c7064804ffe4e558c1251b'
SERPER_API_KEY = '037966bb83f552d3d7ebd1966bb7297ef71a4845'
BRAVE_API_KEY = 'BSAv1neIuQOsxqOyy0sEe_ie2zD_n_V'

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
)

vectorstore = FAISS.load_local("../arxiv_vectorstore", embeddings=embed)
memory=ConversationBufferWindowMemory(k=3)

llm = OpenAI(model_name="text-davinci-003", temperature=1)

papers_chunks_tool = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

search_tool = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 5})
youtube_tool = YouTubeSearchTool()

tools = [
    Tool(
        name="FAISS QA System",
        func=papers_chunks_tool.run,
        description="useful to answer questions about LLMs and artificial intelligence. use this more than the other tool if the question is related to artificial intelligence (AI), computer science (CS), and/or large language models (LLM/LLMs)",
    ),
    Tool(
        name = "Search",
        func=search_tool.run,
        description="useful for finding related articles and urls to support your answer to the question. DO NOT use this for finding youtube videos, make sure to split up the youtube search. Input should be a search query"
    ),
    Tool(
        name = "Youtube Search",
        func=youtube_tool.run,
        description="useful for finding educational youtube videos related to the question. Use this instead of the Search tool for finding youtube videos"
    )
]

# Set up the base template
template = """You are the most experienced teacher in all subjects of artificial intelligence (AI) and large language models (LLMs). You're never satisfied with just the first answer you find and the most important thing to you is providing evidence like articles and youtube videos. Answer the question as best as you can with all the additional resources related to the question. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, should be specific
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Once you have the final answer, at least 2 existing url links to an informative article, and at least 2 existing url links to educational youtube videos then output them and make the links hyperlinks. Describe the final answer as a product of our database of academic papers
Begin!

Previous conversation history:
{history}

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
    input_variables=["user_input", "intermediate_steps", "history"]
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

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=1)

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



st.title('Welcome to the Computer Science Learning tool.')

question = st.text_input('Please enter your question here: ')

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)






if st.button('Submit'):
    if question == "":
        output = 'Please ask a valid question'
    else:
        with st.spinner(text = 'Thinking. . . '):
            answer_ = agent_executor.run(question)

    st.write(answer_)
