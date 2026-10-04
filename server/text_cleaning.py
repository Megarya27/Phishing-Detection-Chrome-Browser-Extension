#This script preprocesses raw email text to match distilBERTs training distribution
#removes html, strips invisible padding characters and replaces pii with standard tokens
# maintained as an separate module so backend server can deploy without external source dependencies

import re
from html import unescape
# regex patterns for replacing personal identifying information

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{3,4}")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

#matches plain text reply markers and quote headers
QUOTED_REPLY_PATTERN = re.compile(r"^(>.*|On .* wrote:)$", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")

#matches person names following common email greetings
GREETING_NAME_PATTERN = re.compile(
    r"\b(Hi|Hello|Hey|Dear|Greetings)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")

#matches person names following email endings
SIGNOFF_NAME_PATTERN = re.compile(
    r"\b(Regards|Kind regards|Best regards|Best wishes|Sincerely|Yours sincerely|"
    r"Yours truly|Cheers|Thanks|Many thanks|Thank you)\s*,?\s*\n+\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

#invisible zero width characters used by some senders to manipulate preview snippets
ZERO_WIDTH_PATTERN = re.compile("[​‌‍‎‏͏⁠]")


#unescapes html entities first then replaces tags with spaces to prevent words running together
def strip_html(raw_html_text: str) -> str:
    unescaped_text = unescape(raw_html_text)
    plain_text = HTML_TAG_PATTERN.sub(" ", unescaped_text)
    return plain_text


#replaces personal details with token placeholders in fixed sequence
def scrub_pii(input_text: str) -> str:
    text_without_emails = EMAIL_PATTERN.sub("[EMAIL]", input_text) #replace emails first so phone regex doesn't match numbers inside addressesc
    text_without_urls = URL_PATTERN.sub("[URL]", text_without_emails) #replace urls next so phone regex doesn't match digits in url parameters
    text_without_phones = PHONE_PATTERN.sub("[PHONE]", text_without_urls)  #replace phone numbers last once other numeric patterns are gone
    text_with_clean_greetings = GREETING_NAME_PATTERN.sub(r"\1 [NAME]", text_without_phones) #replace recipient names in greetings while keeping greeting keywords
    #replace sender names in sign offs while keeping closing keywords
    fully_scrubbed_text = SIGNOFF_NAME_PATTERN.sub(r"\1,\n[NAME]", text_with_clean_greetings)

    return fully_scrubbed_text


#clean + normalise full email content, subject line and message body
def clean_email_text(subject_text: str, body_text: str) -> str:
    #ensure inputs are strings rather than None
    safe_subject = subject_text or ""
    safe_body = body_text or ""
    combined_text = f"{safe_subject}\n{safe_body}"

    #remove zero width characters so word boundaries correctly match
    clean_chars_text = ZERO_WIDTH_PATTERN.sub("", combined_text)
    no_html_text = strip_html(clean_chars_text) #decode and strip html markup

    no_quotes_text = QUOTED_REPLY_PATTERN.sub(" ", no_html_text) #remove quoted message history

    #replace sensitive pii with placeholder tokens
    scrubbed_text = scrub_pii(no_quotes_text)
    valid_utf8_text = scrubbed_text.encode("utf-8", errors="ignore").decode("utf-8") #drop undecodable utf-8 bytes to avoid tokeniser errors
    normalised_text = WHITESPACE_PATTERN.sub(" ", valid_utf8_text).strip() #turn repeated spaces, tabs and newlines into single spaces

    return normalised_text