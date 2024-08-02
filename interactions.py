import json
import time
import uuid

# Sample interaction data
sample_interactions = [
    {
        "timestamp": time.time(),
        "question": "How to plant tomatoes?",
        "response": "Tomatoes should be planted in well-drained soil with plenty of sunlight.",
        "collection_name": "agriculture_ar",
        "user": "user1"
    },
    {
        "timestamp": time.time(),
        "question": "Best fertilizers for wheat?",
        "response": "Nitrogen-rich fertilizers are best for wheat.",
        "collection_name": "agriculture_fr",
        "user": "user2"
    },
    {
        "timestamp": time.time(),
        "question": "How to control pests in rice fields?",
        "response": "Integrated pest management practices are effective for controlling pests in rice fields.",
        "collection_name": "agriculture_ar",
        "user": "user1"
    },
    {
        "timestamp": time.time(),
        "question": "Water requirements for corn?",
        "response": "Corn requires about 1 inch of water per week during the growing season.",
        "collection_name": "agriculture_fr",
        "user": "user3"
    }
]

# Save the interactions to a JSON file
with open("interaction_logs.json", "w", encoding='utf-8') as log_file:
    json.dump(sample_interactions, log_file, ensure_ascii=False, indent=4)

print("Sample interaction_logs.json file created successfully.")
