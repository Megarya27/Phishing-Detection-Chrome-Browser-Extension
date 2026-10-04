//This is the content script for Gmail and Outlook Web
//finds which email is currently open, extracts its body text and links, and communicates with
//background worker to score each link against the hybrid model

const MAX_TEXT_CHARS = 4000;
const MAX_LINKS_PER_EMAIL = 5;
const SCAN_INTERVAL_MS = 1500;
const GMAIL_SELECTORS = ["div.a3s.aiL", "div.a3s", "div[role='listitem'] div.ii.gt div.a3s"];
const OUTLOOK_SELECTORS = ['div[aria-label="Message body"]',"div.allowTextSelection",'div[role="document"]'];

//quoted history/forwarded blocks get removed so the model only sees the text the user is currently reading not old content
const QUOTE_SELECTORS = [".gmail_quote", "blockquote", "#divRplyFwdMsg", "div[id^='appendonsend']"];

//returns true if element is currently visible on screen
function isElementVisible(element) {
  if (element === undefined || element === null) {
    return false;}
  if (element.offsetParent === null) {
    return false;}
  const rect = element.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    return true;} 
    else {
    return false;}
}

// finds the dom container of the currently active email in gmail or outlook
function getEmailContainer() {
  let selectors;
  if (location.hostname === "mail.google.com") {
    selectors = GMAIL_SELECTORS;} 
    else {
    selectors = OUTLOOK_SELECTORS;}

  for (let i = 0; i < selectors.length; i++) {
    const selector = selectors[i];
    const elements = document.querySelectorAll(selector);
    const elementsArray = Array.from(elements);
    const matches = [];
    for (let j = 0; j < elementsArray.length; j++) {
      const el = elementsArray[j];
      let text = el.innerText;
      if (text === undefined || text === null) {
        text = "";}
      text = text.trim();
      if (isElementVisible(el) === true && text.length > 0) {
        matches.push(el);}
    }

    //in a thread, last match is most recently opened email
    if (matches.length > 0) {
      const lastMatchIndex = matches.length - 1;
      return matches[lastMatchIndex];}
  }
  return null;
}

//extracts email text, stripping out quoted replies and forwarded messages
function extractEmailText(emailContainer) {
  let bodyText = emailContainer.innerText;
  if (bodyText === undefined || bodyText === null) {
    bodyText = "";}

  for (let i = 0; i < QUOTE_SELECTORS.length; i++) {
    const selector = QUOTE_SELECTORS[i];
    const quoteBlocks = emailContainer.querySelectorAll(selector);
    for (let j = 0; j < quoteBlocks.length; j++) {
      const quoteBlock = quoteBlocks[j];
      const quotedText = quoteBlock.innerText;
      if (quotedText !== undefined && quotedText !== null && quotedText !== "") {
        const textParts = bodyText.split(quotedText);
        bodyText = textParts.join(" ");}
    }
  }

  //compresses runs of spaces/tabs/nbsp down to one space, then 3+ blank lines down to 2
  bodyText = bodyText.replace(/[ \t ]+/g, " ");
  bodyText = bodyText.replace(/\n{3,}/g, "\n\n");
  return bodyText.trim();}

//extracts base domain from full hostname
function getRegistrableDomain(hostname) {
  const parts = hostname.split(".");
  if (parts.length >= 2) {
    const lastTwoParts = parts.slice(-2);
    return lastTwoParts.join(".");} 
    else {
    return hostname;}}

//finds all unique external links in email, prioritising first link per domain
function extractEmailLinks(emailContainer) {
  const seenUrls = new Set();
  const seenDomains = new Set();
  const primaryLinks = [];
  const secondaryLinks = [];
  const linkElements = emailContainer.querySelectorAll("a[href]");
  for (let i = 0; i < linkElements.length; i++) {
    const link = linkElements[i];
    let rawUrl = link.getAttribute("href");
    
    if (rawUrl === null || rawUrl === undefined) {
      rawUrl = "";}
    
    const isHttpUrl = /^https?:\/\//i.test(rawUrl);
    if (isHttpUrl === false) {
      continue;} // skip mailto:, javascript:, etc.
    if (seenUrls.has(rawUrl) === true) {
      continue;} // skip duplicates
    seenUrls.add(rawUrl);
    let domain;
    try {
      const urlObject = new URL(rawUrl);
      domain = getRegistrableDomain(urlObject.hostname);} 
      catch (error) {
      continue;
    }

    //first link on domain goes to front of queue, subsequent links on same domain go to back of queue
    if (seenDomains.has(domain) === true) {
      secondaryLinks.push(rawUrl);} 
    else {
      seenDomains.add(domain);
      primaryLinks.push(rawUrl);}
  }
  const allLinks = primaryLinks.concat(secondaryLinks);
  return allLinks.slice(0, MAX_LINKS_PER_EMAIL);
}

//sends a single link and its features to background script for inference
function checkLink(targetUrl, emailBodyText) {
  const timings = {};
  const featureStart = performance.now();
  const features = extractEmailLinkFeatures(targetUrl);
  timings.dom_scrape_and_feature_extraction_ms = performance.now() - featureStart;
  const dispatchStart = performance.now();
  return new Promise(function(resolve) {
    const requestMessage = {
      type: "PHISHING_CHECK_REQUEST",
      payload: { 
        url: targetUrl, 
        text: emailBodyText, 
        structural_features: features},
      client_timings: timings,
    };

    chrome.runtime.sendMessage(requestMessage, function(response) {
      if (chrome.runtime.lastError !== undefined) {
        console.warn("[Phishing Guard] sendMessage failed:", chrome.runtime.lastError.message, targetUrl);
        resolve(null);
        return;}
      if (response === undefined || response === null) {
        console.warn("[Phishing Guard] empty response from background", targetUrl);
        resolve(null);
        return;}
      if (response.error !== undefined && response.error !== null) {
        console.warn("[Phishing Guard] API error:", response.error, targetUrl);
        resolve(null);
        return;}
      let apiRoundTripMs = response.timings.api_round_trip_ms;
      if (apiRoundTripMs === undefined || apiRoundTripMs === null) {
        apiRoundTripMs = 0;}

      response.timings.message_passing_ms = performance.now() - dispatchStart - apiRoundTripMs;
      console.log(`[Phishing Guard] checked ${targetUrl}: p=${response.phishing_probability.toFixed(3)}`);
      const resultObject = { url: targetUrl, response: response };
      resolve(resultObject);
    });
  });
}
let lastScannedSignature = null;
let noContainerWarned = false;
let isScanInProgress = false;
//creates a fingerprint to detect when user switches emails
function createEmailSignature(emailText, linkList) {
  return `${emailText.length}:${emailText.slice(0, 200)}:${linkList.join("|")}`;}

//finds the active email, extracts features, and checks all links
async function scanCurrentEmail() {
  if (isScanInProgress === true) {
    return;}
  const container = getEmailContainer();
  if (container === null || container === undefined) {
    if (noContainerWarned === false) {
      console.log("[Phishing Guard] no open-email container found yet (waiting -- open an email to scan it)");
      noContainerWarned = true;}
    return;
  }
  noContainerWarned = false;
  let emailText = extractEmailText(container);
  emailText = emailText.slice(0, MAX_TEXT_CHARS);
  const links = extractEmailLinks(container);
  const signature = createEmailSignature(emailText, links);
  if (signature === lastScannedSignature) {
    return;}
  lastScannedSignature = signature;

  console.log(`[Phishing Guard] scanning open email: ${emailText.length} chars, ${links.length} links found`, links);
  if (links.length === 0) {
    return;}

  chrome.runtime.sendMessage({ type: "SCAN_STARTED" });

  isScanInProgress = true;
  const results = [];
  try {
    for (let i = 0; i < links.length; i++) {
      const url = links[i];
      const result = await checkLink(url, emailText);
      if (result !== null && result !== undefined) {
        results.push(result);}
    }} 
  finally {
    isScanInProgress = false;}

  if (results.length === 0) {
    lastScannedSignature = null; // let next poll cycle retry
    console.warn("[Phishing Guard] all link checks failed: is API reachable at base URL set in popup?");
    chrome.runtime.sendMessage({
      type: "SCAN_RESULT",
      emailText: emailText.slice(0, 300),
      links: [],
      scanFailed: true});
    return;
  }

  const linkSummaries = [];
  for (let i = 0; i < results.length; i++) {
    const resultItem = results[i];
    const url = resultItem.url;
    const response = resultItem.response;
    let isWarningShown = false;
    if (response.phishing_probability >= response.threshold) {
      isWarningShown = true;}
    const summaryObject = {
      url: url,
      warning_shown: isWarningShown,
      phishing_probability: response.phishing_probability,
      semantic_probability: response.semantic_probability,
      structural_probability: response.structural_probability,
      timings: response.timings};
    linkSummaries.push(summaryObject);
  }

  console.log(`[Phishing Guard] sending ${linkSummaries.length} result(s) to background/popup`, linkSummaries);

  const badgeStart = performance.now();
  chrome.runtime.sendMessage({
    type: "SCAN_RESULT",
    emailText: emailText.slice(0, 300),
    links: linkSummaries});

  const badgeUpdateMs = performance.now() - badgeStart;

  //build one latency record per link, print it, and send
  // batch to background worker to append to stored log
  const latencyRecords = [];
  for (let i = 0; i < linkSummaries.length; i++) {
    const summary = linkSummaries[i];
    const t = summary.timings;
    
    let domScrapeMs = t.dom_scrape_and_feature_extraction_ms;
    if (domScrapeMs === undefined || domScrapeMs === null) {
      domScrapeMs = 0;}
    
    let messagePassingMs = t.message_passing_ms;
    if (messagePassingMs === undefined || messagePassingMs === null) {
      messagePassingMs = 0;}
    
    let apiRoundTripMs = t.api_round_trip_ms;
    if (apiRoundTripMs === undefined || apiRoundTripMs === null) {
      apiRoundTripMs = 0;}
    
    const detectionMs = domScrapeMs + messagePassingMs + apiRoundTripMs;
    const endToEndMs = detectionMs + badgeUpdateMs;
    
    const newTimings = Object.assign({}, t);
    newTimings.badge_update_ms = badgeUpdateMs;

    const recordObject = {
      url: summary.url,
      warning_shown: summary.warning_shown,
      phishing_probability: summary.phishing_probability,
      timings: newTimings,
      detection_ms: detectionMs,
      end_to_end_ms: endToEndMs,
      logged_at: Date.now()};
    latencyRecords.push(recordObject);
  }

  for (let i = 0; i < latencyRecords.length; i++) {
    console.debug("[Phishing Guard] latency", latencyRecords[i]);}
  
  chrome.runtime.sendMessage({ type: "LOG_LATENCY_RESULTS", records: latencyRecords });
}
setInterval(scanCurrentEmail, SCAN_INTERVAL_MS); // poll for single page application route changes in webmail
