from imapclient import IMAPClient
import pyzmail
from bs4 import BeautifulSoup
import os

from dotenv import load_dotenv


load_dotenv()
# ============================================================
# Gmail configuration
# ============================================================

HOST = "imap.gmail.com"

USERNAME = os.getenv("EMAIL_USERNAME")
PASSWORD = os.getenv("EMAIL_PASSWORD")


# ============================================================
# Decode email content safely
# ============================================================

def decode_part(part):

    payload = part.get_payload()

    if isinstance(payload, bytes):

        charset = part.charset or "utf-8"

        return payload.decode(
            charset,
            errors="replace"
        )

    return str(payload)


# ============================================================
# Read latest email
# ============================================================

def read_latest_mail():

    if not USERNAME or not PASSWORD:

        print("ERROR: Email credentials not found.")
        print("Please set EMAIL_USERNAME and EMAIL_PASSWORD.")

        return None, None, None


    server = IMAPClient(
        HOST,
        ssl=True
    )


    try:

        # ----------------------------------------------------
        # Login
        # ----------------------------------------------------

        server.login(
            USERNAME,
            PASSWORD
        )


        # ----------------------------------------------------
        # Open Inbox
        # ----------------------------------------------------

        server.select_folder("INBOX")


        # ----------------------------------------------------
        # Get all email UIDs
        # ----------------------------------------------------

        messages = server.search(["ALL"])


        if not messages:

            return None, None, None


        # ----------------------------------------------------
        # Get latest email UID
        # ----------------------------------------------------

        latest_uid = messages[-1]


        # ----------------------------------------------------
        # Fetch latest email
        # ----------------------------------------------------

        raw_data = server.fetch(
            [latest_uid],
            ["BODY[]"]
        )


        # ----------------------------------------------------
        # Convert raw email into PyzMessage
        # ----------------------------------------------------

        msg = pyzmail.PyzMessage.factory(
            raw_data[latest_uid][b"BODY[]"]
        )


        # ----------------------------------------------------
        # Get subject
        # ----------------------------------------------------

        subject = msg.get_subject()


        # ----------------------------------------------------
        # Get email body
        # ----------------------------------------------------

        body = ""


        # ----------------------------------------------------
        # First try plain text
        # ----------------------------------------------------

        if msg.text_part is not None:

            body = decode_part(
                msg.text_part
            )


        # ----------------------------------------------------
        # If plain text doesn't exist,
        # try HTML email
        # ----------------------------------------------------

        elif msg.html_part is not None:

            html = decode_part(
                msg.html_part
            )


            # Convert HTML to normal text
            body = BeautifulSoup(
                html,
                "html.parser"
            ).get_text(
                separator=" ",
                strip=True
            )


        # ----------------------------------------------------
        # No usable content
        # ----------------------------------------------------

        else:

            body = ""


        return (
            latest_uid,
            subject,
            body
        )


    except Exception as e:

        print(
            "Error reading email:",
            e
        )

        return (
            None,
            None,
            None
        )


    finally:

        try:
            server.logout()

        except Exception:
            pass


# ============================================================
# Test reader.py directly
# ============================================================

if __name__ == "__main__":

    uid, subject, body = read_latest_mail()


    print()
    print("=" * 60)

    print("UID:")
    print(uid)

    print()

    print("SUBJECT:")
    print(subject)

    print()

    print("BODY:")
    print(body)

    print("=" * 60)