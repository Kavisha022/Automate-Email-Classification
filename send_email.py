import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")

def send_email(to_email, subject, body):
    # Send an acknowledgment email

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
        server.ehlo()  # Identifies the client to the SMTP server
        server.starttls()  # Secure connection
        server.ehlo()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()


        print(f"Acknowledgment email sent to: {to_email}")

    except Exception as e:
        print(f"Error sending emails to {to_email}: {e}")


def process_and_send_acknowledgments(excel_path):
    # Reads emails from excel and send acknowledgment emails.

    try:
        df = pd.read_excel(excel_path, engine='openpyxl')

        for index, row in df.iterrows():
            to_email = row['Email ID']
            subject = "Acknowledgement: We received your email"
            body = f"Thank you for your email. We have received your message and will get back to you shortly."

            send_email(to_email, subject, body)


    except Exception as e:
        print(f"error processing acknowledgments: {e}")


if __name__ == "__main__":
    excel_path = "D:\\OneDrive - Lowcode Minds Technology Pvt Ltd\\Desktop\\Automate email classification\\emails.xlsx"
    process_and_send_acknowledgments(excel_path)
