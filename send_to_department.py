import pandas as pd
import smtplib
from email.message import EmailMessage
import time
from dotenv import load_dotenv
import os

# Department emails
department_emails = {
    'Automobile': os.getenv('AUTOMOBILE_EMAIL'),
    'Medical': os.getenv('MEDICAL_EMAIL'),
    'Housing': os.getenv('HOUSING_EMAIL')
}

# Excel path
EXCEL_PATH = r'D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\Automate-Email-Classification\emails.xlsx'
SENT_LOG_PATH = r'D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\Automate-Email-Classification\sent_log.txt' # File to track sent email IDs

# Gmail credentials
load_dotenv()

EMAIL_ID = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("SMTP_SERVER")
#SMTP_PORT = os.getenv("SMTP_PORT")
SMTP_PORT = int(os.getenv("SMTP_PORT"))


# Load already sent email subjects
def load_sent_log():
    try:
        with open(SENT_LOG_PATH, 'r') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        return set()
    
def update_sent_log(subject):
    with open(SENT_LOG_PATH, 'a') as f:
        f.write(subject + '\n')

def send_email(subject, body, to_email):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ID
    msg['To'] = to_email
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.login(EMAIL_ID, EMAIL_PASSWORD)
        smtp.send_message(msg)
        print(f"📤 Email sent to {to_email}.")


def forward_new_emails():
    sent_subjects = load_sent_log()
    df = pd.read_excel(EXCEL_PATH)

    for _, row in df.iterrows():
        # subject = row['Subject']
        # body = row['Body']
        # category = row['Category']

        subject = str(row['Subject']).strip()
        body = str(row['Body']).strip()
        category = str(row['Category']).strip().capitalize()

        print(f"📋 Subject: {subject}")
        print(f"📂 Category: {category}")

        if subject not in sent_subjects and category in department_emails:
            to_email = department_emails[category]
            send_email(subject, body, to_email)
            update_sent_log(subject)
        else:
            print(f"⚠️ Skipped: Category '{category}' not found or already sent.")


# 🕒 Run every 5 minutes
while True:
    forward_new_emails()
    print("⏳ Waiting for next check...")
    time.sleep(300)
