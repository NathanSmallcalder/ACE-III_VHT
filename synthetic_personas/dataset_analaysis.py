import json
import os

folder_path = "synthetic_transcripts"
all_data = []

# Loop through every file inside the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Process only JSON files
    if os.path.isfile(file_path) and filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle files containing either a single JSON object or a list of objects
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                all_data.append(data)


# Analysis: lexical difference in answers
def analyze_lexical_difference(data):
    """Analyze lexical differences in answers by class."""
    tokenized_by_class = {"healthy": [], "mci": [], "dementia": []}

    for item in data:
        if not isinstance(item, dict):
            continue

        label = str(item.get("cognitive_status") or "").lower()

        if label not in tokenized_by_class:
            continue
        
        for domain in item.values():
            if not isinstance(domain, list):
                continue
            for entry in domain:
                text = entry.get("Answered", "") if isinstance(entry, dict) else ""
                if not isinstance(text, str):
                    continue
                tokens = [token.lower().strip(".,?!") for token in text.replace("\n", " ").split()]
                tokenized_by_class[label].extend(token for token in tokens if token)

    return tokenized_by_class


tokens_by_class = analyze_lexical_difference(all_data)

for label, tokens in tokens_by_class.items():
    unique = len(set(tokens))
    print(f"{label:<10} {len(tokens):>7} tokens  {unique:>6} unique  / token type ratio {unique / len(tokens):.3f}")