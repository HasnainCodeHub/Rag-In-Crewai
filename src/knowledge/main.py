# from crewai import Agent, Task, Crew, Process, LLM
# from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
# from dotenv import load_dotenv
# import os


# load_dotenv()

# # Get the GEMINI API key
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Create a knowledge source
# content = "Users name is John. He is 30 years old and lives in San Francisco."
# string_source = StringKnowledgeSource(
#     content=content,
# )

# # Create an LLM with a temperature of 0 to ensure deterministic outputs
# def main():
#     gemini_llm = LLM(
#         model="gemini/gemini-2.0-flash",
#         api_key=GEMINI_API_KEY,
#         temperature=0,
#     )

#     # Create an agent with the knowledge store
#     agent = Agent(
#         role="About User",
#         goal="You know everything about the user.",
#         backstory="""You are a master at understanding people and their preferences.""",
#         verbose=True,
#         allow_delegation=False,
#         llm=gemini_llm,
#         embedder={
#             "provider": "google",
#             "config": {
#                 "model": "models/text-embedding-004",
#                 "api_key": GEMINI_API_KEY,
#             }
#         }
#     )

#     task = Task(
#         description="Answer the following questions about the user: {question}",
#         expected_output="An answer to the question.",
#         agent=agent,
#     )

#     crew = Crew(
#         agents=[agent],
#         tasks=[task],
#         verbose=True,
#         process=Process.sequential,
#         knowledge_sources=[string_source],
#         embedder={
#             "provider": "google",
#             "config": {
#                 "model": "models/text-embedding-004",
#                 "api_key": GEMINI_API_KEY,
#             }
#         }
#     )

#     result = crew.kickoff(inputs={"question": "What city does John live in and how old is he?"})
#     print(result)


from crewai import Agent, Task, Crew, Process, LLM

def main():
    # Create the context as part of the agent's backstory instead of using knowledge source
    user_info = """
    Users name is Hasnain Ali. He is 10 years old and lives in San Francisco.
    This information is important and should be used to answer questions about Hasnain.
    """

    # Create an LLM with a temperature of 0 to ensure deterministic outputs
    llm = LLM(model="gemini/gemini-1.5-flash", temperature=0, api_key="AIzaSyAs1me_SiQUlZcKjW97s8o8MwhyXTja-DU")

    # Create an agent with the information in the backstory
    agent = Agent(
        role="Personal Information Assistant",
        goal="Provide accurate information about John based on the available knowledge.",
        backstory=f"""You are an assistant with access to the following information about Hasnain:
        {user_info}
        Your role is to accurately answer questions about him using this information.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    task = Task(
        description="Answer the following questions about John: {question}",
        expected_output="A precise answer based on the available information about Hasnain.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff(inputs={"question": "What city does Hasnain live in and how old is he?"})

