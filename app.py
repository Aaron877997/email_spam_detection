from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

import spacy
import os
import threading
import time

from pickle import load

from reader import read_latest_mail


# ============================================================
# Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# Load spaCy model
# ============================================================

print("Loading spaCy model...")

nlp = spacy.load(
    "en_core_web_lg"
)

print("spaCy model loaded.")


# ============================================================
# Load trained ML model and TF-IDF vectorizer
# ============================================================

model = None
tv = None


if not os.path.exists("model.pkl"):

    print(
        "ERROR: model.pkl not found."
    )

elif not os.path.exists("vector.pkl"):

    print(
        "ERROR: vector.pkl not found."
    )

else:

    try:

        with open(
            "model.pkl",
            "rb"
        ) as f:

            model = load(f)


        with open(
            "vector.pkl",
            "rb"
        ) as f:

            tv = load(f)


        print(
            "ML model and vectorizer loaded successfully."
        )

    except Exception as e:

        print(
            "Error loading ML model:",
            e
        )

        model = None
        tv = None


# ============================================================
# Global variables for Live Analysis
# ============================================================

last_uid = None

last_body = None

last_subject = None

live_result = None

live_running = False


# Lock prevents multiple live threads
live_lock = threading.Lock()


# ============================================================
# Text preprocessing
# ============================================================

def clean_function(text):

    if not text:

        return ""


    # Convert to lowercase

    text = text.lower()


    # spaCy processing

    doc = nlp(text)


    cleaned_tokens = []


    for token in doc:

        # Remove punctuation

        if token.is_punct:
            continue


        # Remove stop words

        if token.is_stop:
            continue


        # Lemmatization

        lemma = token.lemma_


        # Add token

        cleaned_tokens.append(
            lemma
        )


    # Convert list to string

    cleaned_text = " ".join(
        cleaned_tokens
    )


    return cleaned_text


# ============================================================
# Predict Spam / Not Spam
# ============================================================

def predict_spam(text):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None or tv is None:

        return "Model not loaded"


    # --------------------------------------------------------
    # Check text
    # --------------------------------------------------------

    if not text or not text.strip():

        return "No email content"


    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    cleaned_text = clean_function(
        text
    )


    # --------------------------------------------------------
    # Convert text to TF-IDF vector
    # --------------------------------------------------------

    vector_text = tv.transform(
        [cleaned_text]
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = model.predict(
        vector_text
    )[0]


    # --------------------------------------------------------
    # Convert model result to UI result
    # --------------------------------------------------------

    if str(prediction).lower() == "ham":

        return "Not Spam"


    elif str(prediction).lower() == "spam":

        return "Spam"


    return str(prediction)


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    # --------------------------------------------------------
    # POST = manually entered text
    # --------------------------------------------------------

    if request.method == "POST":

        text = request.form.get(
            "text",
            ""
        ).strip()


        if not text:

            return render_template(
                "home.html",
                result="Please enter text",
                body=""
            )


        # Predict

        result = predict_spam(
            text
        )


        return render_template(
            "home.html",
            result=result,
            body=text
        )


    # --------------------------------------------------------
    # GET = normal home page
    # --------------------------------------------------------

    return render_template(
        "home.html"
    )


# ============================================================
# CHECK RECENT EMAIL
# ============================================================

@app.route(
    "/check-recent"
)
def check_recent():

    global last_uid
    global last_body
    global last_subject


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None or tv is None:

        return render_template(
            "home.html",
            result="Model not loaded"
        )


    # --------------------------------------------------------
    # Read latest Gmail email
    # --------------------------------------------------------

    uid, subject, body = read_latest_mail()


    # --------------------------------------------------------
    # No email
    # --------------------------------------------------------

    if uid is None:

        return render_template(
            "home.html",
            result="No emails found"
        )


    # --------------------------------------------------------
    # Save latest email information
    # --------------------------------------------------------

    last_uid = uid

    last_subject = subject

    last_body = body


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    result = predict_spam(
        body
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    return render_template(
        "home.html",
        result=result,
        body=body,
        subject=subject
    )


# ============================================================
# LIVE EMAIL CHECKER
# ============================================================

def live_checker():

    global last_uid
    global last_body
    global last_subject
    global live_result


    print()
    print("=" * 60)
    print("LIVE EMAIL CHECKER STARTED")
    print("=" * 60)


    # --------------------------------------------------------
    # Get the current latest email
    #
    # We do NOT classify it.
    #
    # This establishes the starting point.
    # Only emails arriving AFTER this point are considered new.
    # --------------------------------------------------------

    uid, subject, body = read_latest_mail()


    if uid is not None:

        last_uid = uid

        print(
            "Current latest email UID:",
            uid
        )


    # --------------------------------------------------------
    # Continuously monitor inbox
    # --------------------------------------------------------

    while True:

        try:

            # Read latest email

            uid, subject, body = read_latest_mail()


            # ------------------------------------------------
            # No email
            # ------------------------------------------------

            if uid is None:

                print(
                    "Could not find an email."
                )

                time.sleep(5)

                continue


            # ------------------------------------------------
            # NEW EMAIL DETECTED
            # ------------------------------------------------

            if uid != last_uid:

                print()
                print(
                    "NEW EMAIL DETECTED!"
                )


                # Save email

                last_uid = uid

                last_subject = subject

                last_body = body


                # ------------------------------------------------
                # Predict
                # ------------------------------------------------

                live_result = predict_spam(
                    body
                )


                # ------------------------------------------------
                # Print result
                # ------------------------------------------------

                print(
                    "Subject:",
                    subject
                )

                print(
                    "Result:",
                    live_result
                )

                print(
                    "UID:",
                    uid
                )

                print(
                    "-" * 60
                )


            # ------------------------------------------------
            # Wait before checking again
            # ------------------------------------------------

            time.sleep(5)


        except Exception as e:

            print(
                "Live checker error:",
                e
            )

            time.sleep(5)


# ============================================================
# START LIVE ANALYSIS
# ============================================================

@app.route(
    "/live"
)
def live():

    global live_running


    # --------------------------------------------------------
    # Start only ONE background thread
    # --------------------------------------------------------

    with live_lock:

        if not live_running:

            thread = threading.Thread(
                target=live_checker,
                daemon=True
            )


            thread.start()


            live_running = True


            print(
                "Live analysis enabled."
            )


    # --------------------------------------------------------
    # Show page
    # --------------------------------------------------------

    return render_template(
        "home.html",
        result=live_result,
        body=last_body,
        subject=last_subject,
        live=True
    )


# ============================================================
# LIVE STATUS API
#
# Browser calls this every few seconds.
# ============================================================

@app.route(
    "/live-status"
)
def live_status():

    return jsonify({

        "result": live_result,

        "body": last_body,

        "subject": last_subject

    })


# ============================================================
# Run Flask
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("STARTING SPAM DETECTION APPLICATION")
    print("=" * 60)
    print()


    app.run(

        debug=True,

        # IMPORTANT:
        # prevents Flask reloader from
        # starting background thread twice

        use_reloader=False
    )