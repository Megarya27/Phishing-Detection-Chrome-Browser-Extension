# Feature extraction functions for URL analysis
#this module is for features that require network lookups (WHOIS, DNS, blacklist) and 
# are too slow to compute on the client side. 
# The client can supply the other features (DOM/URL) in a dict, 
# and this module will merge them with the server-resolved 
# features to produce a complete feature vector for the XGBoost model.

from ast import If
from datetime import datetime, timezone
import whois

import csv
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import dns.resolver
import requests
import whois

# web_traffic, Page_Rank, Google_Index and Links_pointing_to_page
# are not included here because they require external API calls and are not 
# feasible to compute in real-time for every URL.
# 10 destination-DOM features (Favicon, Request_URL, URL_of_Anchor,
# Links_in_tags, SFH, Submitting_to_email, on_mouseover, RightClick,
#popUpWidnow, Iframe) that require rendering the page a link points to are also not 
#included here because they require a headless browser 
# and are too slow to compute in real-time for every URL.

STRUCTURAL_FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "port", "HTTPS_token", "Abnormal_URL",
    "Redirect", "age_of_domain", "DNSRecord", "Statistical_report",
]

DEFAULT = 0 #default value for missing features

PHISHTANK_CACHE_PATH = Path(__file__).resolve().parent.parent / "phishtank dataset.csv"
phishtank_hosts = None
# load the PhishTank dataset once and cache it in memory to avoid repeated disk reads.
def load_phishtank_hosts():
    global phishtank_hosts
    if phishtank_hosts is not None:
        return phishtank_hosts
    hosts = set() #not full URLs, just the hostnames, to avoid false negatives from query string variations
    #missing or malformed PhishTank dataset should not crash the program so we ignore errors and return an empty 
    #set if the file is not found or cannot be read.
    if PHISHTANK_CACHE_PATH.exists():
        with open(PHISHTANK_CACHE_PATH, encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                url = row.get("url", "")
                if url:
                    hosts.add(urlparse(url).hostname or "")
    phishtank_hosts = hosts
    return hosts

#SSL final state feature: 1 if HTTPS, -1 if HTTP.
def compute_ssl_final_state(url: str) -> int:
    parsed_url = urlparse(url)
    if parsed_url.scheme == "https":
        return 1
    else:
        return -1
    
from datetime import datetime, timezone
import whois


def first(value):
    # Some WHOIS records return list of dates/domains instead of just one.
    # If it's a list, grab the first item else, keep value as it is.
    if isinstance(value, list):
        return value[0]
    else:
        return value


def to_utc(date_value):
    # If date exists and doesn't have a timezone set, label it as UTC.
    if date_value is not None:
        if date_value.tzinfo is None:
            return date_value.replace(tzinfo=timezone.utc)

    return date_value

# if WHOIS lookup fails, return -1 for all three features. Otherwise, compute:
# Age of domain (is it at least 6 months?)
#Registration length (is it registered for at least 1 year?)
#Abnormal URL (does registered domain match requested hostname?)

def compute_domain_features(hostname):
    #get WHOIS domain registration details
    try:
        domain_info = whois.whois(hostname)
    except Exception:
        #return fallback values if lookup fails
        return -1, -1, -1

    #Extract and standardise the dates
    creation_date = first(domain_info.creation_date)
    creation_date = to_utc(creation_date)

    expiration_date = first(domain_info.expiration_date)
    expiration_date = to_utc(expiration_date)

    current_time = datetime.now(timezone.utc)

    #age of domain (is it at least 6 months / 180 days old?)
    try:
        if creation_date is not None:
            age_in_days = (current_time - creation_date).days
            if age_in_days >= 180:
                age_of_domain = 1
            else:
                age_of_domain = -1
        else:
            age_of_domain = -1
    except Exception:
        age_of_domain = -1

    #registration length (is it registered for at least 365 days?)
    try:
        if creation_date is not None and expiration_date is not None:
            duration_in_days = (expiration_date - creation_date).days
            if duration_in_days >= 365:
                registration_length = 1
            else:
                registration_length = -1
        else:
            registration_length = -1
    except Exception:
        registration_length = -1

    #abnormal URL (does the registered domain match the requested hostname?)
    try:
        registered_domain = first(domain_info.domain_name)

        if registered_domain is None:
            registered_domain = ""

        #formatting: lowercase and remove trailing dots
        clean_registered = registered_domain.lower().strip(".")
        clean_hostname = hostname.lower()

        if clean_registered in clean_hostname and clean_registered != "":
            abnormal_url = 1
        else:
            abnormal_url = -1
    except Exception:
        abnormal_url = -1

    return registration_length, age_of_domain, abnormal_url

#dns record feature: 1 if DNS record exists, -1 if not. 
#this is proxy for if domain is active and reachable.
def compute_dns_record(hostname: str) -> int:
    try:
        dns.resolver.resolve(hostname, "A", lifetime=3.0) #3 second timeout for DNS query. 
                                                        #If no A record found, it will raise an exception.
        return 1
    #fallback through OS resolver if DNS query fails. This is slower but more robust for some domains.
    except Exception: 
        try:
            socket.gethostbyname(hostname)
            return 1
        except Exception:
            return -1

#redirect feature: 0 if no redirects, 1 if at least one redirect.
def compute_redirect(url: str) -> int:
    #make a HEAD request to the URL and follow redirects. Count the number of redirects.
    #Head request is used instead of GET to avoid downloading the full page content, which is unnecessary for this feature.
    try: 
        resp = requests.head(url, allow_redirects=True, timeout=4.0)
        n_redirects = len(resp.history)
        # at most one redirect is allowed for legitimate sites, so we return 0 if there are 0 or 1 redirects, and 1 if there are more than 1.
        return 0 if n_redirects <= 1 else 1
    #If the request fails (timeout, connection error, etc.), return DEFAULT value.
    except Exception:
        return DEFAULT

#crosschecks the given hostname against the PhishTank dataset. 
# Returns 1 if not found (legitimate), -1 if found (phishing).
#used as cheap proxy for if domain is known to be malicious.
def compute_statistical_report(hostname: str) -> int:
    hosts = load_phishtank_hosts()
    return -1 if hostname in hosts else 1

#merges client supplied features with server resolved features to output 
#complete feature vector for the XGBoost model.
def complete_feature_vector(url: str, client_features: dict) -> list:
    hostname = urlparse(url).hostname or ""
    #1 whois lookup to get all three domain features
    registration_length, age_of_domain, abnormal_url = compute_domain_features(hostname)
    resolved = dict(client_features or {}) #copy client supplied features to avoid mutating the input dict
    #setdefault ensures that if the feature is already present in client_features, it is not overwritten.
    resolved.setdefault("SSLfinal_State", compute_ssl_final_state(url)) 
    resolved["Domain_registeration_length"] = registration_length
    resolved["age_of_domain"] = age_of_domain
    resolved["Abnormal_URL"] = abnormal_url
    resolved["DNSRecord"] = compute_dns_record(hostname)
    resolved["Redirect"] = compute_redirect(url)
    resolved["Statistical_report"] = compute_statistical_report(hostname)

#itrates through the list of STRUCTURAL_FEATURES and retrieves corresponding 
#value from resolved dict. dict iteration order would not be guaranteed to 
#match the order of STRUCTURAL_FEATURES, so it explicitly iterates
#through list to ensure output vector is in correct order.
    return [float(resolved.get(name, DEFAULT) or DEFAULT)
            for name in STRUCTURAL_FEATURES]
