from agent_graph import ask_agent

question = "What is the population of Tokyo?"

print("Question:", question)
print("\nRunning Agentic RAG with web search...\n")

result = ask_agent(question)

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(
        f"Source: {source.get('source', 'Unknown')} | "
        f"Page: {source.get('page', 'Unknown')}"
    )