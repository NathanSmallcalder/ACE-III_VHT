import json
import os
from datetime import datetime

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "json/session_config.json")
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def resolve_dynamic_answers(answers, session_config):
    now = datetime.now()
    resolved = []
    for answer in answers:
        if answer == "DYNAMIC:day_of_week":
            resolved.append(now.strftime("%A"))
        elif answer == "DYNAMIC:date":
            resolved.append(now.strftime("%d"))
        elif answer == "DYNAMIC:month":
            resolved.append(now.strftime("%B"))
        elif answer == "DYNAMIC:year":
            resolved.append(now.strftime("%Y"))
        elif answer == "DYNAMIC:season":
            resolved.append(get_season(now))
        elif answer == "DYNAMIC:number_or_floor":
            resolved.append(session_config["location"]["number"])
        elif answer == "DYNAMIC:street":
            resolved.append(session_config["location"]["street"])
        elif answer == "DYNAMIC:town":
            resolved.append(session_config["location"]["town"])
        elif answer == "DYNAMIC:county":
            resolved.append(session_config["location"]["county"])
        elif answer == "DYNAMIC:country":
            resolved.append(session_config["location"]["country"])
        elif answer == "DYNAMIC:uk_prime_minister":
            resolved.append(session_config["current_uk_pm"])
        elif answer == "DYNAMIC:us_president":
            resolved.append(session_config["current_us_president"])
        else:
            resolved.append(answer)
    return resolved

def get_session_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    # Fallback to interactive prompts if config file is missing
    return {
        "location": {
            "number": input("Building number/floor: "),
            "street": input("Street: "),
            "town": input("Town: "),
            "county": input("County: "),
            "country": input("Country: ")
        },
        "patient": {
            "name": input("Patient name: "),
            "dob": input("Date of birth: ")
        },
        "assessor": input("Assessor name: "),
        "current_uk_pm": input("Current UK Prime Minister: "),
        "current_us_president": input("Current US President: ")
    }
