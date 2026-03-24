"""
tbt_demo.py
===========
Demo dataset generators for the TbT Sentiment Analytics Streamlit app.

Each function returns a ``pd.DataFrame`` in the exact column format expected
by the corresponding domain parser in ``tbt_engine.ConversationProcessor``.

These are kept in a separate module so that:
- ``tbt_app.py`` stays focused on UI logic.
- Tests can import generators without importing Streamlit.
- Adding a new demo format is a single, isolated change here.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Public registry — imported by tbt_app.py
# ---------------------------------------------------------------------------

#: Maps sidebar demo labels → (generator_fn, dataset_type_key)
DEMO_REGISTRY: dict = {}  # populated by @_register decorator below


def _register(label: str, dtype: str):
    """Decorator that registers a demo generator under a sidebar-display label."""
    def decorator(fn):
        DEMO_REGISTRY[label] = (fn, dtype)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Individual demo generators
# ---------------------------------------------------------------------------

@_register("Media / Entertainment B (Bracket)", "netflix")
def _demo_netflix() -> pd.DataFrame:
    s = (
        "[12:30:08 AGENT]:\n These articles might answer your questions.\n"
        "[12:30:08 CUSTOMER]:\n Chat with an agent\n"
        "[12:34:52 AGENT]:\n You are now chatting with: Support\n"
        "[12:35:26 AGENT]:\n Hello! How can I help you today?\n"
        "[12:36:00 CUSTOMER]:\n I cannot access my account at all. Very frustrated.\n"
        "[12:36:30 AGENT]:\n I apologise for the trouble. Let me reset your password right away.\n"
        "[12:37:00 CUSTOMER]:\n That would be great, thank you!\n"
        "[12:37:30 AGENT]:\n Reset email has been sent to your inbox.\n"
        "[12:38:00 CUSTOMER]:\n Works now. Thank you so much!\n"
        "[12:39:00 CUSTOMER]:\n Actually there is another issue — my watchlist disappeared.\n"
        "[12:39:30 AGENT]:\n I am sorry to hear that. Let me investigate further.\n"
        "[12:40:10 AGENT]:\n I can see the watchlist is still on our servers. Refreshing now.\n"
        "[12:40:45 CUSTOMER]:\n It is back! You have been incredibly helpful.\n"
        "[12:41:00 AGENT]:\n Wonderful! Is there anything else I can assist you with today?\n"
        "[12:41:30 CUSTOMER]:\n No, that is all. Have a great day!\n"
    )
    # Three conversations with slight variation
    s2 = (
        "[09:00:01 AGENT]:\n Welcome to support. How can I help?\n"
        "[09:00:30 CUSTOMER]:\n My subscription was charged twice this month!\n"
        "[09:01:05 AGENT]:\n I sincerely apologise. Let me pull up your billing details.\n"
        "[09:02:00 AGENT]:\n I can see two charges. I will raise a refund immediately.\n"
        "[09:02:45 CUSTOMER]:\n How long will the refund take?\n"
        "[09:03:10 AGENT]:\n Up to 5 business days, depending on your bank.\n"
        "[09:04:00 CUSTOMER]:\n That is very disappointing. I needed that money today.\n"
        "[09:04:30 AGENT]:\n I understand your frustration. I have escalated this for priority processing.\n"
        "[09:05:00 CUSTOMER]:\n Okay, I appreciate you escalating it. Thank you.\n"
    )
    s3 = (
        "[15:10:00 AGENT]:\n Hi there! How can I help you today?\n"
        "[15:10:30 CUSTOMER]:\n The video quality on my device is terrible — constant buffering.\n"
        "[15:11:00 AGENT]:\n I am sorry to hear that. Are you on Wi-Fi or mobile data?\n"
        "[15:11:30 CUSTOMER]:\n Wi-Fi, and it is fast. Other apps work fine.\n"
        "[15:12:00 AGENT]:\n Let me run a diagnostic on your account settings.\n"
        "[15:12:45 AGENT]:\n Found it — your streaming quality was capped. I have removed the cap.\n"
        "[15:13:15 CUSTOMER]:\n It is working perfectly now! That was so simple. Thank you!\n"
    )
    return pd.DataFrame({"Transcripts": [s, s2, s3]})


@_register("Healthcare A (Call Transcript)", "humana")
def _demo_humana() -> pd.DataFrame:
    t1 = (
        "[00:09] System: [Automated] Thank you for calling. "
        "[03:02] Agent: Thank you for calling, may I have your member ID? "
        "[03:06] Member: Yes, let me find my card. "
        "[03:20] Member: It is 603345. "
        "[03:24] Agent: Can I have your date of birth? "
        "[03:27] Member: January 9, 1958. "
        "[03:46] Agent: How can I help you today? "
        "[03:52] Member: I had X-rays last month and I am checking on the claim status. "
        "[04:36] Agent: The claim was processed on December 31. "
        "[04:44] Member: Do I owe anything? "
        "[04:48] Agent: Your portion is $30. "
        "[05:22] Member: Thank you so much, that is great! "
        "[05:42] Agent: You are welcome! Have a great day."
    )
    t2 = (
        "[00:05] System: [Automated] Please hold while we connect you. "
        "[02:00] Agent: Hello, thank you for your patience. How can I assist? "
        "[02:10] Member: I was billed for a service my doctor said should be covered. "
        "[02:30] Agent: I understand your concern. Let me look up your policy benefits. "
        "[03:15] Agent: I see the service should indeed be covered. This looks like a billing error. "
        "[03:45] Member: That is incredibly frustrating. This is the second error this year. "
        "[04:00] Agent: I sincerely apologise. I am raising a correction request now with highest priority. "
        "[04:30] Agent: You should receive a corrected explanation of benefits within 10 days. "
        "[05:00] Member: I hope so. This should not keep happening. "
        "[05:20] Agent: You are absolutely right, and I am making a note to flag your account for review. "
        "[06:00] Member: Okay, thank you for your help."
    )
    t3 = (
        "[01:00] Agent: Good morning! How can I help you today? "
        "[01:10] Member: I need to find an in-network specialist near me. "
        "[01:20] Agent: Of course. What is your zip code? "
        "[01:25] Member: 30301. "
        "[01:45] Agent: I have found three rheumatologists within 10 miles who are accepting new patients. "
        "[02:00] Member: Wonderful! Can you send me their contact details? "
        "[02:10] Agent: Absolutely — I will send them to your email on file right now. "
        "[02:30] Member: Perfect. This is exactly what I needed. Thank you so much!"
    )
    return pd.DataFrame({"Conversation": [t1, t2, t3]})


@_register("Healthcare B (Chat / SMS)", "ppt")
def _demo_ppt() -> pd.DataFrame:
    pt1 = (
        "<b>00:00:04 system : </b>Thank you for contacting us. Please wait.<br />"
        "<b>00:00:34 customer1 : </b>Hi, can I cancel my appointment for tomorrow?<br />"
        "<b>00:01:31 agent1 : </b>Hello! Thank you for contacting us.<br />"
        "<b>00:01:54 customer1 : </b>Jane Doe, appointment on Monday.<br />"
        "<b>00:02:35 agent1 : </b>Thank you! Let me pull up the account.<br />"
        "<b>00:03:10 agent1 : </b>I see the appointment. Would you like to reschedule?<br />"
        "<b>00:04:22 customer1 : </b>No, just cancel please. I am not feeling well.<br />"
        "<b>00:05:18 agent1 : </b>Done! Appointment cancelled.<br />"
        "<b>00:08:34 customer1 : </b>Thank you so much!<br />"
        "<b>00:10:01 agent1 : </b>You are welcome! Have a nice day!<br />"
    )
    pt2 = (
        "<b>10:00:00 system : </b>You are now connected.<br />"
        "<b>10:00:30 customer2 : </b>I need to get my prescription refilled urgently.<br />"
        "<b>10:01:00 agent2 : </b>I am here to help. What is the medication name?<br />"
        "<b>10:01:30 customer2 : </b>Metformin 500mg. I have run out and I feel terrible.<br />"
        "<b>10:02:00 agent2 : </b>I am so sorry you are going through this. Let me check your file.<br />"
        "<b>10:02:45 agent2 : </b>I see an active prescription. I can send a 30-day supply to your pharmacy.<br />"
        "<b>10:03:10 customer2 : </b>That would be such a relief. Thank you!<br />"
        "<b>10:03:45 agent2 : </b>It has been sent. You can pick it up in 2 hours.<br />"
        "<b>10:04:00 customer2 : </b>Amazing. You are a lifesaver!<br />"
    )
    return pd.DataFrame({"Comments": [pt1, pt2, pt1]})


@_register("Transportation (Verbatim)", "lyft")
def _demo_lyft() -> pd.DataFrame:
    verbatims = [
        "I waited to be connected and never was. Wanted a partial refund or credit.",
        "Driver was amazing. Clean car, smooth ride! Will definitely use again.",
        "App kept crashing every time I tried to book. Very frustrating experience.",
        "Waited 20 minutes and the driver cancelled twice. Completely unacceptable.",
        "Great service, on time, and fair pricing. Highly recommend to everyone.",
        "Charged a cleaning fee but the ride was perfectly clean. Disputing this charge.",
        "Driver went above and beyond. Five stars all the way! Excellent experience.",
        "Ride was fine but nothing special for the premium price paid.",
        "The GPS routed us through an awful traffic jam. Added 30 minutes to my trip.",
        "Second time this month with a no-show driver. Need to rethink using this service.",
        "Quick pickup and the driver was very professional. Exactly what I needed.",
        "Surge pricing at peak time seems excessive — nearly three times the normal rate.",
    ]
    return pd.DataFrame({"verbatim": verbatims})


@_register("Travel (Guest Feedback)", "hilton")
def _demo_hilton() -> pd.DataFrame:
    feedback = [
        "Room was spotless and the bed was incredibly comfortable! Best sleep in years.",
        "Check-in took over 30 minutes despite having a reservation. Very frustrating.",
        "Pool area was beautiful, but the restaurant service was painfully slow.",
        "Room was not cleaned properly — stains on the sheets. Completely unacceptable.",
        "The concierge was extremely helpful and went out of their way for us.",
        "WiFi kept dropping throughout the stay — unacceptable for a business trip.",
        "Loved the complimentary breakfast, great variety and everything was fresh!",
        "Room was fine but nothing exceptional for the premium price paid.",
        "The spa treatment was wonderful and the staff made us feel very welcome.",
        "Air conditioning in the room was noisy all night. Could not sleep properly.",
        "Upgraded to a suite as a surprise! The view was breathtaking. Exceptional stay.",
        "Parking fee was never disclosed at booking. Hidden charges feel dishonest.",
    ]
    return pd.DataFrame({"Additional Feedback": feedback})


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def make_demo_df(label: str) -> tuple[pd.DataFrame, str]:
    """
    Return (DataFrame, dataset_type) for a given demo label.

    Parameters
    ----------
    label : str
        One of the keys in ``DEMO_REGISTRY``.

    Returns
    -------
    df : pd.DataFrame
    dataset_type : str
        Matching key for ``ConversationProcessor``.

    Raises
    ------
    KeyError
        If the label is not in DEMO_REGISTRY.
    """
    fn, dtype = DEMO_REGISTRY[label]
    return fn(), dtype
