from web_search import search_web

question = "What is the population of Tokyo?"

results = search_web(question, max_results=5)

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Title:", result.get("title"))
    print("Snippet:", result.get("snippet"))
    print("URL:", result.get("url"))