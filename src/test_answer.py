
from answer_generator import answer_question

question = "Explain each deep learning model used in the project, its role, and the reported classification results."

print("Question:", question)
print("\nGenerating answer from the project report...\n")

result = answer_question(question)

print("\nANSWER:")
print(result["answer"])

print("\nSOURCE PASSAGES:")

for source in result["sources"]:
    print(f"\nFile: {source['source']}")
    print(f"Page: {source['page']}")
    print(source["text"][:250])
    print("-" * 50)
