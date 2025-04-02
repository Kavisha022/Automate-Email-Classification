import imaplib
import pandas as pd
from email_parser import parse_email
from email.utils import parseaddr
from dotenv import load_dotenv
import os
# import smtplib
# from email.mime.text import MIMEText
# from send_email import process_and_send_acknowledgments


#Excel_PATH = 'D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\Automate email classification\emails.xlsx'

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("IMAP_SERVER")

email_data = []

def fetch_emails():
    # Fetches an processes incoming emails

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")

        result, data = mail.search(None, "ALL")
        if result != "OK":
            print("No messages found!")
            return
        
        email_ids = data[0].split()
        print(f"Found {len(email_ids)} emails.")

        for email_id in email_ids:
            try:
                result, msg_data = mail.fetch(email_id, "(RFC822)")
                print(f"Fetching email ID: {email_id}")

                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        email_content, email_subject, email_from = parse_email(response_part[1])
                        #category = classify_email(email_content)

                        print(f"Fetching email from: {email_from}")

                        email_data.append({
                            'Email ID': email_from,
                            'Subject': email_subject,
                            'Body': email_content, 
                            #'Category': category
                        })

                        #send_email(EMAIL_USER, Catogory)

                        # # Send acknowledgement email
                        # send_acknowledgment(email_from, email_subject)

            except Exception as e:
                print(f"Error processing email {email_id}: {e}")
            
        
        print("Extracted Email Data: ",email_data)    
        save_to_excel()
        mail.logout() 

    except Exception as e:
        print(f"Error fetching emails: {e}")

# def send_acknowledgment(to_email, subject):
#     # Send an acknowledgment email

#     try:
#         msg = MIMEText("Thank you for your email. We have received your message and will get back to you shortly.")
#         msg['Subject'] = f"Acknowledgment: {subject}"
#         msg['From'] = EMAIL_USER
#         msg['To'] = to_email

#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls() # Upgrade the connection to a secure encrypted SSL/TLS connection
#             server.login(EMAIL_USER, EMAIL_PASS)
#             server.sendmail(EMAIL_USER, to_email, msg.as_string())
#             print(f"Acknowledgment sent to {to_email}")

#     except Exception as e:
#         print(f"Error sending acknowledgment to {to_email}: {e}")


def save_to_excel():
    # Save the extracted email data to Excel sheet

    try:
        df = pd.DataFrame(email_data)
        df["Email ID"] = df["Email ID"].str.replace(" ", "", regex=False)
        df.to_excel(r'D:\OneDrive - Lowcode Minds Technology Pvt Ltd\Desktop\Automate email classification\emails.xlsx', index=False, engine='openpyxl')
        print("Emalis saving to Excel.")

    except Exception as e:
        print(f"Error saving to Excel: {e}")


if __name__ == "__main__":
    fetch_emails()   
    #process_and_send_acknowledgments(Excel_PATH)         

