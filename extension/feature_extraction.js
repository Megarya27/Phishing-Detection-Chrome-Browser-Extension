//This script performs client side extraction of structural features
//1 = legitimate, 0 = suspicious, -1 = phishing
// network dependent features are set to null and filled on server side.

//list of common url shortening domain patterns
const SHORTENER_PATTERNS = /\b(bit\.ly|tinyurl\.com|goo\.gl|t\.co|is\.gd|ow\.ly|buff\.ly|adf\.ly|shorte\.st)\b/i;

//checks if hostname is ipv4 or ipv6 address instead of domain name
function isIpAddress(hostname) {
  const isIpv4 = /^(\d{1,3}\.){3}\d{1,3}$/.test(hostname);
  const isIpv6 = hostname.includes(":");
  return isIpv4 || isIpv6;}

//extracts preclick structural features directly from url text
function extractUrlFeatures(urlString) {
  const parsedUrl = new URL(urlString);
  const hostname = parsedUrl.hostname;
  const features = {};

  //raw ip addresses instead of domain names suggest suspicious hosting
  if (isIpAddress(hostname)) {
    features.having_IP_Address = -1;}
    else {
    features.having_IP_Address = 1;}

  //score url length based on standard cutoff lengths, short is clean, long is suspicious
  //these cutoffs are based on the statistical analysis of url lengths in the UCI phishing dataset
  if (urlString.length < 54) {
    features.URL_Length = 1;} 
  else if (urlString.length <= 75) {
    features.URL_Length = 0;} 
  else {
    features.URL_Length = -1;}

  //url shorteners hide real destination address
  if (SHORTENER_PATTERNS.test(urlString)) {
    features.Shortining_Service = -1;} 
    else {
    features.Shortining_Service = 1;}

  //using @ symbol tricks browsers into ignoring preceding text, commonly used to disguise destinations
  if (urlString.includes("@")) {
    features.having_At_Symbol = -1;} 
  else {
    features.having_At_Symbol = 1;}

  //look for double slash after index 8 to detect redirect decoys
  const doubleSlashIndex = urlString.indexOf("//", 8);
  if (doubleSlashIndex > 7) {
    features.double_slash_redirecting = -1;} 
  else {
    features.double_slash_redirecting = 1;}

  // hyphens in domain names are often used to fake brand names
  if (hostname.includes("-")) {
    features.Prefix_Suffix = -1;} 
  else {
    features.Prefix_Suffix = 1;}

  //count subdomains, subtract 2 to account for the primary domain and tld
  const domainParts = hostname.split(".");
  const subdomainCount = domainParts.length - 2;
  if (subdomainCount <= 1) {
    features.having_Sub_Domain = 1;} 
  else if (subdomainCount === 2) {
    features.having_Sub_Domain = 0;} 
  else {
    features.having_Sub_Domain = -1;}

  //check for http/https tokens used inside domain name itself to fake security
  if (hostname.includes("https") || hostname.includes("http")) {
    features.HTTPS_token = -1;} 
  else {
    features.HTTPS_token = 1;}

  //standard ports (empty, 80, or 443) are expected and unusual ports are suspicious
  const portNumber = parsedUrl.port;
  if (portNumber === "" || portNumber === "80" || portNumber === "443") {
    features.port = 1;} 
  else {
    features.port = -1;}
  return features;
}

//extracts page level dom features which is used by offline dataset builders on rendered pages
function extractDomFeatures(pageDocument, pageOrigin) {
  const features = {};
  //check if  pages favicon is borrowed from external domain
  const faviconLink = pageDocument.querySelector("link[rel~='icon']");
  if (faviconLink) {
    try {
      const faviconOrigin = new URL(faviconLink.href, pageOrigin).origin;
      if (faviconOrigin === pageOrigin) {
        features.Favicon = 1;} 
      else {
        features.Favicon = -1;}} 
    catch (urlError) {
      features.Favicon = 0;}} 
  else {
    features.Favicon = 0;}

  //measure fraction of embedded resources load from outside origins
  const resourceElements = Array.from(pageDocument.querySelectorAll("img[src], script[src], link[href]"));
  let externalResourceCount = 0;
  resourceElements.forEach((element) => {
    const resourceUrl = element.getAttribute("src") || element.getAttribute("href");
    try {
      if (resourceUrl && new URL(resourceUrl, pageOrigin).origin !== pageOrigin) {
        externalResourceCount = externalResourceCount + 1;}} 
    catch (parseError) { }
  });

  const totalResources = resourceElements.length;
  const externalResourceRatio = totalResources > 0 ? externalResourceCount / totalResources : 0;

  if (externalResourceRatio < 0.22) {
    features.Request_URL = 1;} 
  else if (externalResourceRatio <= 0.61) {
    features.Request_URL = 0;} 
  else {
    features.Request_URL = -1;}
  features.Links_in_tags = features.Request_URL; // links_in_tags shares exact same metric base

  //inspect link anchors as blank, javascript or external links suggest inactive template clone
  const anchorElements = Array.from(pageDocument.querySelectorAll("a[href]"));
  let unsafeAnchorCount = 0;
  anchorElements.forEach((anchor) => {
    const linkDestination = anchor.getAttribute("href") || "";
    if (linkDestination === "#" || linkDestination.startsWith("javascript:") || linkDestination === "") {
      unsafeAnchorCount = unsafeAnchorCount + 1;} 
      else {
      try {
        if (new URL(linkDestination, pageOrigin).origin !== pageOrigin) {
          unsafeAnchorCount = unsafeAnchorCount + 1;}} 
      catch (parseError) {
        unsafeAnchorCount = unsafeAnchorCount + 1;}
    }
  });
  const totalAnchors = anchorElements.length;
  const unsafeAnchorRatio = totalAnchors > 0 ? unsafeAnchorCount / totalAnchors : 0;
  if (unsafeAnchorRatio < 0.31) {
    features.URL_of_Anchor = 1;} 
  else if (unsafeAnchorRatio <= 0.67) {
    features.URL_of_Anchor = 0;} 
  else {
    features.URL_of_Anchor = -1;}

  //check destination of forms, cross origin or mailto submissions are suspicious
  const formElements = Array.from(pageDocument.querySelectorAll("form"));
  let formHandlerScore = 1;
  let emailSubmissionScore = 1;

  formElements.forEach((form) => {
    const formAction = (form.getAttribute("action") || "").trim();

    //blank or about:blank actions are immediate suspicious signs
    if (formAction === "" || formAction.toLowerCase() === "about:blank") {
      formHandlerScore = -1;} 
    else {
      try {
        const formOrigin = new URL(formAction, pageOrigin).origin;
        // flag cross-origin form destinations unless already scored as worst case (-1).
        if (formOrigin !== pageOrigin && formHandlerScore !== -1) {
          formHandlerScore = 0;
        }} 
        catch (parseError) {
        formHandlerScore = -1;}
    }

    // submitting sensitive data directly to a mailto address is suspicious.
    if (/mailto:/i.test(formAction)) {
      emailSubmissionScore = -1;}
  });

  features.SFH = formHandlerScore;
  features.Submitting_to_email = emailSubmissionScore;

  //detects mouseover tricks that alter links or spoof status bar information
  const mouseoverElements = pageDocument.querySelectorAll("[onmouseover]");
  if (mouseoverElements.length > 0) {
    features.on_mouseover = -1;} 
  else {
    features.on_mouseover = 1;}

  //inspect whether right clicking is disabled to stop users inspecting source code
  const scriptTextSnippet = Array.from(pageDocument.scripts).map((scriptElement) => scriptElement.textContent || "").join("\n").slice(0, 20000);
  const isContextMenuDisabled = pageDocument.oncontextmenu !== null;
  const hasRightClickScript = /event\.button\s*==\s*2/i.test(scriptTextSnippet);

  if (isContextMenuDisabled || hasRightClickScript) {
    features.RightClick = -1;} 
  else {
    features.RightClick = 1;}

  //detect popup windows launched via script
  const hasWindowOpenCall = Array.from(pageDocument.scripts).some((scriptElement) => {
    return /window\.open\s*\(/.test(scriptElement.textContent || "");});

  if (hasWindowOpenCall) {
    features.popUpWidnow = -1;} 
  else {
    features.popUpWidnow = 1;}

  //detect hidden or zero size iframes used for clickjacking or silent overlays
  const iframeElements = pageDocument.querySelectorAll("iframe, frame");
  let hasHiddenIframe = false;
  iframeElements.forEach((iframe) => {
    const isDisplayNone = iframe.style.display === "none";
    const isVisibilityHidden = iframe.style.visibility === "hidden";
    const isZeroWidth = iframe.getAttribute("width") === "0";
    const isZeroHeight = iframe.getAttribute("height") === "0";

    if (isDisplayNone || isVisibilityHidden || isZeroWidth || isZeroHeight) {
      hasHiddenIframe = true;}
  });

  if (hasHiddenIframe) {
    features.Iframe = -1;} 
  else {
    features.Iframe = 1;}
  return features;
}

//features that require external network lookups and must be resolved by backend server
const SERVER_RESOLVED_FEATURES = [
  "Domain_registeration_length",
  "Abnormal_URL",
  "Redirect",
  "age_of_domain",
  "DNSRecord",
  "web_traffic",
  "Page_Rank",
  "Google_Index",
  "Links_pointing_to_page"];

//extracts all observable features for full page offline crawls
function extractClientFeatures() {
  const urlFeatures = extractUrlFeatures(window.location.href);
  const domFeatures = extractDomFeatures(document, window.location.origin);
  const combinedFeatures = Object.assign({}, urlFeatures, domFeatures);

  //mark backend dependent features as null so server knows to look them up
  SERVER_RESOLVED_FEATURES.forEach((featureName) => {
    combinedFeatures[featureName] = null;});
  return combinedFeatures;
}

//extracts preclick features for links inside webmail, no dom features are included
function extractEmailLinkFeatures(urlString) {
  const urlFeatures = extractUrlFeatures(urlString);

  // mark server resolved features as null
  SERVER_RESOLVED_FEATURES.forEach((featureName) => {
    urlFeatures[featureName] = null;});

  return urlFeatures;
}

//export functions for unit tests in node environments while remaining clean in browser scripts
if (typeof module !== "undefined") {
  module.exports = {
    extractClientFeatures,
    extractEmailLinkFeatures,
    extractUrlFeatures,
    extractDomFeatures};}