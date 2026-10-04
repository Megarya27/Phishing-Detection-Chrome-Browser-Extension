//This script controls popup panel: shows verdicts, probability breakdowns and latency for email open
//builds DOM nodes with textContent rather than innerHTML so an attacker controlled URL
//in results cant inject markup into the popup itself

const backendApiUrlInput = document.getElementById("backendApiUrlInput");
const decisionThresholdDisplay = document.getElementById("decisionThresholdDisplay");
const statusMessage = document.getElementById("statusMessage");
const resultsList = document.getElementById("resultsList");
const emptyStateNotice = document.getElementById("emptyStateNotice");
const scanMetadata = document.getElementById("scanMetadata");
const saveSettingsButton = document.getElementById("saveSettingsButton");
const latencyLogCount = document.getElementById("latencyLogCount");
const exportLatencyButton = document.getElementById("exportLatencyButton");
const clearLatencyButton = document.getElementById("clearLatencyButton");

//load saved api url from storage on startup
chrome.storage.local.get(["apiBaseUrl"], (stored) => {
  backendApiUrlInput.value = (stored && stored.apiBaseUrl) || "http://localhost:5000";});

//gets fixed decision threshold from background script
chrome.runtime.sendMessage({ type: "GET_THRESHOLD" }, (response) => {
  if (!chrome.runtime.lastError && response && typeof response.threshold === "number") {
    decisionThresholdDisplay.textContent = response.threshold.toFixed(2);
  }});

//save custom api url to local storage
saveSettingsButton.addEventListener("click", () => {
  const updatedUrl = backendApiUrlInput.value.trim();
  chrome.storage.local.set({ apiBaseUrl: updatedUrl }, () => {
    statusMessage.textContent = "Saved.";
    setTimeout(() => { statusMessage.textContent = ""; }, 1500);});
});

//update display showing number of latency records collected
function refreshLatencyLogCount() {
  chrome.storage.local.get(["latencyLog"], (stored) => {
    const n = (stored.latencyLog || []).length;
    latencyLogCount.textContent = `${n} link check(s) recorded`;});
}

//start download of recorded latency benchmark data
exportLatencyButton.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "EXPORT_LATENCY_LOG" }, (response) => {
    if (response && response.error) {
      latencyLogCount.textContent = response.error;
    } else if (response) {
      latencyLogCount.textContent = `Exported ${response.n_records} record(s).`;
    }});
});

//clear recorded latency benchmark data from local storage
clearLatencyButton.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "CLEAR_LATENCY_LOG" }, () => refreshLatencyLogCount());});
refreshLatencyLogCount();

function formatMilliseconds(ms) {
  if (ms === undefined || ms === null) {
    return "--";} 
  else {
    return ms.toFixed(1) + "ms";}}

function formatPercentage(probability) {
  const value = Number(probability);
  if (Number.isFinite(value)) {
    return (value * 100).toFixed(1) + "%";} 
  else {
    return "--";}}

function createElementWithText(tagName, className, text) {
  const el = document.createElement(tagName);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;}

function createLabelledRow(rowClass, labelText, labelClass, valueText, valueClass) {
  const row = createElementWithText("div", rowClass);
  row.appendChild(createElementWithText("span", labelClass, labelText));
  row.appendChild(createElementWithText("span", valueClass, valueText));
  return row;}

//builds dom elements for single scanned link result card
function renderLinkCard(linkData) {
  const card = document.createElement("div");
  if (linkData.warning_shown) {
    card.className = "link-result-card is-flagged";} 
  else {
    card.className = "link-result-card";}

  const timings = linkData.timings || {};
  const totalDetectionMs =
    (timings.dom_scrape_and_feature_extraction_ms || 0) +
    (timings.message_passing_ms || 0) +
    (timings.api_round_trip_ms || 0);

  let verdictText;
  let verdictClass;
  if (linkData.warning_shown) {
    verdictText = "Phishing suspected";
    verdictClass = "verdict-header is-danger";} 
  else {
    verdictText = "Not phishing";
    verdictClass = "verdict-header is-safe";}

  //built with createTextNode rather than reading the URL directly, so URL can't break out as markup
  const urlLine = createElementWithText("div", "link-url-text");
  let urlString;
  if (linkData.url == null) {
    urlString = "";} 
  else {
    urlString = String(linkData.url);}
  urlLine.appendChild(document.createTextNode(urlString));
  card.appendChild(urlLine);
  card.appendChild(createElementWithText("div", verdictClass, verdictText));
  card.appendChild(createLabelledRow("risk-data-row", "Phishing probability", "risk-label",
    formatPercentage(linkData.phishing_probability), "risk-score phishing-probability-value"));
  card.appendChild(createLabelledRow("risk-data-row", "Semantic (email text)", "risk-label",
    formatPercentage(linkData.semantic_probability), "risk-score"));
  card.appendChild(createLabelledRow("risk-data-row", "Structural (URL)", "risk-label",
    formatPercentage(linkData.structural_probability), "risk-score"));

  const timingsSection = createElementWithText("div", "timing-breakdown-section");
  timingsSection.appendChild(createLabelledRow("", "DOM scrape + features", "", formatMilliseconds(timings.dom_scrape_and_feature_extraction_ms), ""));
  timingsSection.appendChild(createLabelledRow("", "Message passing", "", formatMilliseconds(timings.message_passing_ms), ""));
  timingsSection.appendChild(createLabelledRow("", "API round-trip", "", formatMilliseconds(timings.api_round_trip_ms), ""));
  timingsSection.appendChild(createLabelledRow("", " of which backend", "", formatMilliseconds(timings.backend_processing_ms), ""));
  timingsSection.appendChild(createLabelledRow("", "Badge update", "", formatMilliseconds(timings.badge_update_ms), ""));

  const totalRow = createElementWithText("div");
  const bold = createElementWithText("strong");
  bold.appendChild(createElementWithText("span", "", "Detection total"));
  bold.appendChild(createElementWithText("span", "", formatMilliseconds(totalDetectionMs)));
  totalRow.appendChild(bold);
  timingsSection.appendChild(totalRow);
  card.appendChild(timingsSection);
  return card;}

function renderLoadingState() {
  resultsList.innerHTML = "";
  emptyStateNotice.style.display = "none";
  scanMetadata.textContent = "";
  resultsList.appendChild(createElementWithText("div", "", "Checking email..."));}

// updates the dom with the latest scan results for the active tab
function renderResults(scanPayload) {
  resultsList.innerHTML = "";

  if (!scanPayload || !scanPayload.links || scanPayload.links.length === 0) {
    emptyStateNotice.style.display = "block";
    if (scanPayload && scanPayload.scanFailed) {
      emptyStateNotice.textContent = "Scan failed -- check the browser console (is the API reachable at the configured base URL?).";} 
    else {
      emptyStateNotice.textContent = "No scan results for this tab yet. Open an email in Gmail or Outlook Web.";}
    scanMetadata.textContent = "";
    return;
  }

  emptyStateNotice.style.display = "none";

  const secondsAgo = Math.round((Date.now() - scanPayload.scannedAt) / 1000);
  const flaggedCount = scanPayload.links.filter((link) => link.warning_shown).length;
  scanMetadata.textContent = `${scanPayload.links.length} link(s) checked, ${flaggedCount} flagged -- scanned ${secondsAgo}s ago`;
  scanPayload.links.forEach((link) => resultsList.appendChild(renderLinkCard(link)));
}

let currentActiveTabId = null;
//fetches latest scan state from background script and updates popup
function refreshPopupData() {
  if (currentActiveTabId == null) return;

  chrome.runtime.sendMessage({ type: "GET_SCAN_RESULT", tabId: currentActiveTabId }, (response) => {
    if (response && response.scanning) {
      renderLoadingState();} 
    else if (response && response.data) {
      renderResults(response.data);} 
    else {
      renderResults(null);
    }});
}

//find active tab and trigger an initial data refresh
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const activeTab = tabs[0];
  if (!activeTab) return;
  currentActiveTabId = activeTab.id;
  refreshPopupData();
});

//listen for state changes from background script to keep popup live
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "SCAN_STATUS_CHANGED" && message.tabId === currentActiveTabId) {
    refreshPopupData();}
});
