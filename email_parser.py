import mailparser
import re
from email.utils import parseaddr
from email.utils import getaddresses


def clean_text(text):
    # Remove unwanted characters, extra spaces and special symbols
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?]', ' ', text)
    return text.strip()

def parse_email(raw_email):
    # Extracts structured data from an email
    parsed_mail = mailparser.parse_from_bytes(raw_email)

    email_content = "No Body Content"

    # Extract body content
    if parsed_mail.text_plain:
        email_content = parsed_mail.text_plain[0]  
        print("Extracted from text_plain")
    else:
        print("No body content found")


    email_content = clean_text(email_content) 

    # Extract subject
    if parsed_mail.subject:
        email_subject = parsed_mail.subject
    else:
        email_subject = "No Subject"

    # Extract email id
    if parsed_mail.from_:
        
        email_from = parsed_mail.from_[0][1]
    else:
        email_from = "Unkown Sender"

    return email_content, email_subject, email_from
