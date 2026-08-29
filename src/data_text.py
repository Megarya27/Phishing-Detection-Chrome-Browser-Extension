# This program builds teh NLP training corpus for DistilBERT
# using the Nazario Phishing Email Corpus which contains only phishing Emails
# and the SpamAssassin Public Corpus which contains mostly legitamate emails.
# Additionally, datasets like Enron / Ling / Nigerian_Fraud / CEAS_08 will  be used
# if the combined pool is less than 15,000, which is the sample minumum. 