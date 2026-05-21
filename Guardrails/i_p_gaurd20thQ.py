import re
from langdetect import detect

# 1. Prompt Injection Detection

# from langdetect import detect

# text = "Hello how are you"

# print(detect(text))

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "bypass security",
    "developer mode",
    "jailbreak",
    "forget previous instructions",
    "show confidential data"
]

def detect_prompt_injection(query):
    query = query.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in query:
            return True
    return False

# 2. PII Detection

def detect_pii(text):
    pii_patterns = [
        r"\b\d{12}\b",  #adhar card
        r"[A-Z]{5}[0-9]{4}[A-Z]{1}", #PAN
        r"\b\d{10}\b", #phone number
        r"\S+@\S+\.\S+" #Email
    ]

    for pattern in pii_patterns:
        if re.search(pattern,text):
            return True
    return False

# 3. Toxic / Unsafe Word Filtering

UNSAFE_WORDS=[
    "hack",
    "kill",
    "bomb",
    "steal"
]

def detect_unsafe_content(query):
    query = query.lower()
    for word in UNSAFE_WORDS:
        if word in query:
            return True
    return False

# 4. Query Length Validation

MAX_QUERY_LENGTH = 500

def validate_length(query):
    return len(query) <= MAX_QUERY_LENGTH


# 5. Language Validation

SUPPORTED_LANGUAGE ='en'

def validate_language(query):
    try:
        return detect(query)==SUPPORTED_LANGUAGE
    except:
        return False
    
# 6. Domain Validation
# Example:
# Your RAG project is related to finance/banking

ALLOWED_TOPICS = [
    "loan",
    "bank",
    "interest",
    "credit",
    "investment",
    "finance"
]

def validate_domain(query):
    query = query.lower()
    for topic in ALLOWED_TOPICS:
        if topic in query:
            return True
    return False

# FINAL INPUT GUARDRAIL FUNCTION

def validate_user_input(query):
    if detect_prompt_injection(query):
        return{
            "status":"BLOCKED",
            "reason":"Prompt Injection Detected"
        }
    if detect_pii(query):
        return {
            "status":"Blocked",
            "reason":"PII/Sensitive Data Detected"
        }
    if detect_unsafe_content(query):
        return {
            "status":"BLOCKED",
            "reason":"Unsafe Content Detected"
        }
    
    if not validate_length(query):
        return{
            "status":"BLOCKED",
            "reason":"Unsupported Language"
        }

    if not validate_domain(query):
        return{
            "status":"BLOCKED",
            "reason":"Out of Domain Query"
        }
    return {
        "status":"SAFE",
        "reason":"Validation Passed"
    }

# Example Usage Before Retrieval

user_query = input("Enter your query: ")

result = validate_user_input(user_query)

print(result)

if result["status"] == "SAFE":

    # Proceed to RAG Retrieval
    print("Sending query to Vector DB...")

    # embedding = embedding_model(user_query)
    # docs = vector_db.similarity_search(user_query)

else:
    print("Blocked Query")