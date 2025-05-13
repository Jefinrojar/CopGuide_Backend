import requests
import json

url_search = "http://localhost:5000/search"
url_feedback = "http://localhost:5000/feedback"

prompts = [
    {
        "question": "Can a government servant purchase second-hand articles at market rates?",
        "expected_answer": "Yes, a government servant can purchase second-hand articles at the normal or prevailing market rate, as this is explicitly allowed under the rules."
    },
    # Add other prompts here
]

for prompt in prompts:
    # Send question
    response = requests.post(url_search, json={"question": prompt["question"]})
    if response.status_code == 200:
        data = response.json()
        print(f"Question: {prompt['question']}")
        print(f"Answer: {data['answer']}")
        
        # Simulate feedback (1 if answer is correct, -1 if incorrect)
        feedback_score = 1 if prompt["expected_answer"] in data["answer"] else -1
        feedback_payload = {
            "feedback_id": data["feedback_id"],
            "feedback": feedback_score,
            "doc_index": data["doc_index"]
        }
        feedback_response = requests.post(url_feedback, json=feedback_payload)
        print(f"Feedback Status: {feedback_response.json()['status']}")
    else:
        print(f"Error: {response.json()['error']}")