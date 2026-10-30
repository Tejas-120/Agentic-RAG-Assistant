
import json
import time
from pathlib import Path

from agent_graph import ask_agent


TEST_CASES = [
    {
        "question": "Which deep learning models were used in the brain tumor classification project?",
        "expected_keywords": [
            "resnet18",
            "mobilenet",
            "efficientnet",
        ],
    },
    {
        "question": "What classification accuracy did the soft voting ensemble achieve?",
        "expected_keywords": [
            "accuracy",
            "ensemble",
        ],
    },
    {
        "question": "What is the population of Tokyo?",
        "expected_keywords": [],
    },
]


def evaluate_case(test_case):
    question = test_case["question"]
    expected_keywords = [
        keyword.lower()
        for keyword in test_case["expected_keywords"]
    ]

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    start_time = time.perf_counter()

    try:
        result = ask_agent(question)
        elapsed = time.perf_counter() - start_time

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        answer_lower = answer.lower()

        if expected_keywords:
            matched = [
                keyword
                for keyword in expected_keywords
                if keyword in answer_lower
            ]
            keyword_coverage = len(matched) / len(expected_keywords)
        else:
            matched = []
            keyword_coverage = None

        record = {
            "question": question,
            "answer": answer,
            "elapsed_seconds": round(elapsed, 2),
            "source_count": len(sources),
            "source_pages": [
                source.get("page", "Unknown")
                for source in sources
            ],
            "matched_keywords": matched,
            "keyword_coverage": keyword_coverage,
        }

        print("\nANSWER:")
        print(answer)
        print("\nSOURCE COUNT:", len(sources))
        print("SOURCE PAGES:", record["source_pages"])
        print("TIME:", record["elapsed_seconds"], "seconds")

        if expected_keywords:
            print(
                "KEYWORD COVERAGE:",
                f"{len(matched)}/{len(expected_keywords)}",
            )

        return record

    except Exception as exc:
        print("EVALUATION ERROR:", exc)

        return {
            "question": question,
            "error": str(exc),
        }


def main():
    results = []

    for test_case in TEST_CASES:
        results.append(evaluate_case(test_case))

    output_path = (
        Path(__file__).resolve().parent.parent
        / "evaluation_results.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    successful = [
        result for result in results
        if "error" not in result
    ]

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Successful runs: {len(successful)}")
    print(f"Failed runs: {len(TEST_CASES) - len(successful)}")
    print(f"Results saved to: {output_path}")

    if successful:
        average_time = sum(
            result["elapsed_seconds"]
            for result in successful
        ) / len(successful)

        print(f"Average response time: {average_time:.2f} seconds")


if __name__ == "__main__":
    main()
