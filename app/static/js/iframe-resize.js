/**
 * WordPress iframe height communication
 * Sends height updates to parent WordPress page for responsive embedding
 */

function sendHeight() {
    const height = document.body.scrollHeight;
    parent.postMessage({iframeHeight: height}, "*");
}

// Initialize on page load
window.onload = sendHeight;
window.onresize = sendHeight;

// Also send height after any DOM changes (e.g., plot updates)
if (window.MutationObserver) {
    const observer = new MutationObserver(sendHeight);
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true
    });
}

