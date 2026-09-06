// The backend has no static-file server yet, so this page is opened
    // directly (file://) or from a separate static server, hence an
    // absolute API base rather than a relative path. Matches the README's
    // `uvicorn main:app --reload` default.
    const API_BASE =window.location.hostname === "127.0.0.1" ||
      window.location.hostname === "localhost" ? "http://127.0.0.1:8000": "";

    const STEPS = [
      { stage: "plan", nodes: ["analyze", "plan"], label: "Plan" },
      { stage: "search", nodes: ["search"], label: "Search" },
      { stage: "review", nodes: ["process"], label: "Review" },
      { stage: "synthesize", nodes: ["summarize"], label: "Synthesize" },
      { stage: "validate", nodes: ["validate", "quality_check"], label: "Validate" },
      { stage: "report", nodes: ["report"], label: "Report" },
    ];

    function stageForNode(node) {
      return STEPS.findIndex((step) => step.nodes.includes(node));
    }

    const promptInput = document.getElementById("prompt-input");
    const sendBtn = document.getElementById("send-btn");
    const errorText = document.getElementById("error-text");

    const progressSection = document.getElementById("progress-section");
    const progressCaption = document.getElementById("progress-caption");
    const progressList = document.getElementById("progress-list");

    const reportDivider = document.getElementById("report-divider");
    const reportSection = document.getElementById("report-section");
    const reportTitle = document.getElementById("report-title");
    const reportSummary = document.getElementById("report-summary");
    const keyFindingsEl = document.getElementById("key-findings");
    const findingsContainer = document.getElementById("findings-container");
    const sourcesList = document.getElementById("sources-list");
    const copyReportBtn = document.getElementById("copy-report-btn");
    let reportTextForClipboard = "";
    let copyFeedbackTimer = null;

    // --- Settings (Step: Model Selector + Settings panel) ---------------
    // Everything below is additive UI chrome: persisted locally via
    // localStorage. It does not alter the research pipeline or streaming
    // logic below (buildProgressList / setProgressState / renderReport /
    // parseSseBlock / runResearch's SSE handling are all unchanged).

    const SETTINGS_KEY = "meridian_settings_v1";
    const KEY_PROVIDERS = ["gemini", "nvidia", "groq"];
    const DEFAULT_SETTINGS = {
      provider: "gemini",
      theme: "dark",
      useInbuiltKey: { gemini: true, nvidia: true, groq: true },
      keys: { gemini: "", nvidia: "", groq: "" },
      models: {
        gemini: "gemini-2.5-flash",
        nvidia: "nvidia/nemotron-3-super-120b-a12b",
        groq: "openai/gpt-oss-20b",
      },
    };

    function loadSettings() {
      try {
        const raw = localStorage.getItem(SETTINGS_KEY);
        if (!raw) return { ...DEFAULT_SETTINGS };
        const parsed = JSON.parse(raw);
        return {
          ...DEFAULT_SETTINGS,
          ...parsed,
          useInbuiltKey: { ...DEFAULT_SETTINGS.useInbuiltKey, ...(parsed.useInbuiltKey || {}) },
          keys: { ...DEFAULT_SETTINGS.keys, ...(parsed.keys || {}) },
          models: { ...DEFAULT_SETTINGS.models, ...(parsed.models || {}) },
        };
      } catch {
        return { ...DEFAULT_SETTINGS };
      }
    }

    function saveSettings(settings) {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    }

    let currentSettings = loadSettings();

    function applyTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme === "light" ? "light" : "dark");
    }

    function setActiveProviderButton(provider) {
      const buttons = document.querySelectorAll("#model-select button");
      buttons.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.provider === provider);
      });
    }

    function setActiveThemeButton(theme) {
      const buttons = document.querySelectorAll("#theme-toggle button");
      buttons.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.theme === theme);
      });
    }

    // Reflects a provider's On/Off state in its toggle buttons and shows
    // or hides that provider's "Your API Key" field accordingly.
    function setInbuiltToggleState(provider, useInbuilt) {
      const toggle = document.querySelector(`.inbuilt-toggle[data-provider="${provider}"]`);
      if (toggle) {
        toggle.querySelectorAll("button[data-value]").forEach((btn) => {
          const isOn = btn.dataset.value === "on";
          btn.classList.toggle("active", isOn === useInbuilt);
        });
      }
      const field = document.querySelector(`.custom-key-field[data-provider="${provider}"]`);
      if (field) field.classList.toggle("open", !useInbuilt);
    }

    function populateSettingsForm() {
      for (const provider of KEY_PROVIDERS) {
        document.getElementById(`key-${provider}`).value = currentSettings.keys[provider];
        setInbuiltToggleState(provider, currentSettings.useInbuiltKey[provider]);
        const confirmEl = document.getElementById(`save-confirm-${provider}`);
        if (confirmEl) confirmEl.textContent = "";
      }
      document.getElementById("model-gemini").value = currentSettings.models.gemini;
      document.getElementById("model-nvidia").value = currentSettings.models.nvidia;
      document.getElementById("model-groq").value = currentSettings.models.groq;
      setActiveThemeButton(currentSettings.theme);
    }

    // Apply persisted theme/provider immediately on load.
    applyTheme(currentSettings.theme);
    setActiveProviderButton(currentSettings.provider);

    document.getElementById("model-select").addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-provider]");
      if (!btn) return;
      currentSettings.provider = btn.dataset.provider;
      setActiveProviderButton(currentSettings.provider);
      saveSettings(currentSettings);
    });

    const settingsOverlay = document.getElementById("settings-overlay");
    const settingsBtn = document.getElementById("settings-btn");

    settingsBtn.addEventListener("click", () => {
      populateSettingsForm();
      settingsOverlay.classList.add("open");
    });

    document.getElementById("settings-close").addEventListener("click", () => {
      settingsOverlay.classList.remove("open");
    });

    settingsOverlay.addEventListener("click", (e) => {
      if (e.target === settingsOverlay) settingsOverlay.classList.remove("open");
    });

    document.getElementById("top-theme-btn").addEventListener("click", () => {
      const nextTheme = currentSettings.theme === "dark" ? "light" : "dark";
      currentSettings.theme = nextTheme;
      applyTheme(nextTheme);
      setActiveThemeButton(nextTheme);
      saveSettings(currentSettings);
    });

    document.getElementById("theme-toggle").addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-theme]");
      if (!btn) return;
      setActiveThemeButton(btn.dataset.theme);
      applyTheme(btn.dataset.theme);
    });

    // Each provider's On/Off toggle switches immediately (no Save needed
    // to reveal/hide the custom key field), matching the requested
    // "Off: show ... input" behavior.
    document.querySelectorAll(".inbuilt-toggle").forEach((toggle) => {
      toggle.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-value]");
        if (!btn) return;
        const provider = toggle.dataset.provider;
        const useInbuilt = btn.dataset.value === "on";
        currentSettings.useInbuiltKey[provider] = useInbuilt;
        setInbuiltToggleState(provider, useInbuilt);
      });
    });

    // Pulls every current form control's value into currentSettings.
    // Shared by the per-provider "Save" buttons and the main Save button
    // so both write the same complete, consistent settings object.
    function readFormIntoSettings() {
      for (const provider of KEY_PROVIDERS) {
        currentSettings.keys[provider] = document.getElementById(`key-${provider}`).value.trim();
        // useInbuiltKey[provider] is already kept current by the toggle handler above.
      }
      currentSettings.models.gemini =
        document.getElementById("model-gemini").value.trim() || DEFAULT_SETTINGS.models.gemini;
      currentSettings.models.nvidia =
        document.getElementById("model-nvidia").value.trim() || DEFAULT_SETTINGS.models.nvidia;
      currentSettings.models.groq =
        document.getElementById("model-groq").value.trim() || DEFAULT_SETTINGS.models.groq;
      currentSettings.theme =
        document.querySelector("#theme-toggle button.active")?.dataset.theme || "dark";
    }

    // Builds the request payload's custom_api_keys: only providers whose
    // Inbuilt toggle is Off AND that have a saved key are included. A
    // provider left On (or Off with no key saved yet) is simply omitted,
    // so the backend falls back to its own configured (inbuilt) key for
    // it. Tavily is never included — it has no override, by design.
    function buildCustomApiKeysPayload() {
      const payload = {};
      for (const provider of KEY_PROVIDERS) {
        if (!currentSettings.useInbuiltKey[provider] && currentSettings.keys[provider]) {
          payload[provider] = currentSettings.keys[provider];
        }
      }
      return payload;
    }

    document.querySelectorAll(".save-key-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const provider = btn.dataset.provider;
        readFormIntoSettings();
        saveSettings(currentSettings);
        const confirmEl = document.getElementById(`save-confirm-${provider}`);
        if (confirmEl) {
          confirmEl.textContent = "Saved.";
          setTimeout(() => {
            confirmEl.textContent = "";
          }, 2000);
        }
      });
    });

    document.getElementById("settings-save").addEventListener("click", () => {
      readFormIntoSettings();
      applyTheme(currentSettings.theme);
      saveSettings(currentSettings);
      settingsOverlay.classList.remove("open");
    });

    // --- Speech-to-text (mic button) --------------------------------------
    // Uses the browser's native Web Speech API. Purely client-side: no
    // backend involvement, no API keys. Fills the input with the
    // recognized transcript and never submits automatically — the person
    // still has to review the text and press Send themselves.

    const micBtn = document.getElementById("mic-btn");
    const micBtnLabel = document.getElementById("mic-btn-label");
    const speechLangSelect = document.getElementById("speech-lang-select");

    const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isListening = false;
    let isStartingRecognition = false;
    let currentSpeechLang = "en-US";
    let speechPrefix = "";

    function setActiveSpeechLangButton(lang) {
      speechLangSelect.querySelectorAll("button[data-lang]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.lang === lang);
      });
    }

    function closeSpeechLanguageMenu() {
      speechLangSelect.classList.remove("open");
      micBtn.setAttribute("aria-expanded", "false");
    }

    function toggleSpeechLanguageMenu() {
      const willOpen = !speechLangSelect.classList.contains("open");
      speechLangSelect.classList.toggle("open", willOpen);
      micBtn.setAttribute("aria-expanded", String(willOpen));
    }

    speechLangSelect.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-lang]");
      if (!btn) return;
      currentSpeechLang = btn.dataset.lang;
      setActiveSpeechLangButton(currentSpeechLang);
      closeSpeechLanguageMenu();
      startListening();
    });

    document.addEventListener("click", (e) => {
      if (!e.target.closest(".speech-control")) closeSpeechLanguageMenu();
    });

    function setMicListeningState(listening) {
      micBtn.classList.toggle("listening", listening);
      micBtnLabel.textContent = listening ? "Listening…" : "Speak";
      micBtn.setAttribute("aria-label", listening ? "Stop listening" : "Speak");
      micBtn.setAttribute("aria-expanded", "false");
      micBtn.title = listening ? "Listening… Click to stop." : "Speak your research question";
    }

    function showSpeechError(error) {
      const messages = {
        "no-speech": "Didn't catch that — try speaking again.",
        "not-allowed": "Microphone access was blocked. Allow it to use voice input.",
        "service-not-allowed": "Speech recognition is blocked by this browser or device.",
        "audio-capture": "No microphone was found or it is unavailable.",
        network: "Speech recognition could not reach its service. Check your connection and try again.",
        aborted: "Speech recognition was stopped.",
      };
      showError(messages[error] || `Speech recognition error: ${error}`);
    }

    async function requestMicrophonePermission() {
      try {
        const permission = await navigator.permissions?.query({ name: "microphone" });
        if (permission?.state === "denied") {
          showSpeechError("not-allowed");
          return false;
        }
      } catch {
        // The Permissions API is not supported everywhere; getUserMedia and
        // SpeechRecognition below still provide the actual permission prompt.
      }

      if (!navigator.mediaDevices?.getUserMedia) return true;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
        return true;
      } catch (error) {
        if (error.name === "NotAllowedError" || error.name === "SecurityError") {
          showSpeechError("not-allowed");
        } else if (error.name === "NotFoundError" || error.name === "NotReadableError") {
          showSpeechError("audio-capture");
        } else {
          showError("Unable to access the microphone. Check your browser permissions and try again.");
        }
        return false;
      }
    }

    async function startListening() {
      if (isListening || isStartingRecognition) return;
      closeSpeechLanguageMenu();
      isStartingRecognition = true;
      micBtn.disabled = true;

      try {
        if (!(await requestMicrophonePermission())) {
          isStartingRecognition = false;
          micBtn.disabled = false;
          return;
        }

        speechPrefix = promptInput.value.trim();
        recognition = new SpeechRecognitionCtor();
        recognition.lang = currentSpeechLang; // en-US or ta-IN; mixed speech is browser-dependent.
        recognition.interimResults = true;
        recognition.continuous = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
          isStartingRecognition = false;
          isListening = true;
          micBtn.disabled = false;
          setMicListeningState(true);
        };

        recognition.onresult = (event) => {
          let transcript = "";
          for (let i = 0; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
          }
          // Fill the textarea only. The person reviews it and presses Send
          // manually; recognition never submits a research request.
          promptInput.value = [speechPrefix, transcript.trim()].filter(Boolean).join(" ");
        };

        recognition.onerror = (event) => showSpeechError(event.error);

        recognition.onend = () => {
          isStartingRecognition = false;
          isListening = false;
          micBtn.disabled = false;
          setMicListeningState(false);
          recognition = null;
        };

        recognition.start();
      } catch (error) {
        isStartingRecognition = false;
        isListening = false;
        micBtn.disabled = false;
        setMicListeningState(false);
        const detail = error instanceof Error && error.message ? `: ${error.message}` : ".";
        showError(`Unable to start speech recognition${detail}`);
      }
    }

    if (!SpeechRecognitionCtor) {
      micBtn.disabled = true;
      micBtn.title = "Speech input isn't supported in this browser.";
      micBtnLabel.textContent = "Speak (unsupported)";
    } else {
      micBtn.addEventListener("click", () => {
        if (isListening) {
          recognition && recognition.stop();
          return;
        }
        if (isStartingRecognition) return;
        toggleSpeechLanguageMenu();
      });
    }



    // --- Research pipeline / streaming (unchanged) -----------------------

    function buildProgressList() {
      progressList.innerHTML = "";
      document.querySelectorAll(".pipeline-stage").forEach((stage) => {
        stage.classList.remove("active", "done");
      });
      document.getElementById("status-title").textContent = "Ready to research";
      document.getElementById("status-detail").textContent = "Enter your question and click Send to begin.";
    }

    function setProgressState(completedNode) {
      const idx = stageForNode(completedNode);
      if (idx === -1) return;
      const stages = Array.from(document.querySelectorAll(".pipeline-stage"));
      stages.forEach((stage, i) => {
        stage.classList.remove("active", "done");
        if (i < idx) stage.classList.add("done");
        else if (i === idx) stage.classList.add("done");
        else if (i === idx + 1) stage.classList.add("active");
      });
      const label = STEPS[idx]?.label || "Working";
      document.getElementById("status-title").textContent = label;
      document.getElementById("status-detail").textContent = "Research pipeline is processing your request.";
    }

    function resetUI() {
      errorText.style.display = "none";
      errorText.textContent = "";

      buildProgressList();
      progressSection.style.display = "block";
      progressCaption.textContent = "Research in progress";

      reportSection.style.display = "none";
      reportDivider.style.display = "none";
      reportTitle.textContent = "";
      reportSummary.textContent = "";
      keyFindingsEl.innerHTML = "";
      findingsContainer.innerHTML = "";
      sourcesList.innerHTML = "";
      reportTextForClipboard = "";
      clearTimeout(copyFeedbackTimer);
      copyReportBtn.textContent = "Copy";
    }

    function showError(message) {
      errorText.textContent = message;
      errorText.style.display = "block";
    }

    function hostnameOf(url) {
      try {
        return new URL(url).hostname.replace(/^www\./, "");
      } catch {
        return url;
      }
    }

    function buildReportClipboardText(report) {
      const lines = [report.title, "", report.executive_summary, "", "Key findings"];

      for (const point of report.key_findings || []) {
        lines.push(`• ${point}`);
      }

      for (const finding of report.findings || []) {
        lines.push("", finding.topic, finding.summary);
        for (const point of finding.key_points || []) {
          lines.push(`• ${point}`);
        }
      }

      return lines.filter((line) => typeof line === "string").join("\n").trim();
    }

    copyReportBtn.addEventListener("click", async () => {
      if (!reportTextForClipboard) return;

      if (!navigator.clipboard?.writeText) {
        showError("Copying is not supported in this browser.");
        return;
      }

      try {
        await navigator.clipboard.writeText(reportTextForClipboard);
        clearTimeout(copyFeedbackTimer);
        copyReportBtn.textContent = "Copied ✓";
        copyFeedbackTimer = setTimeout(() => {
          copyReportBtn.textContent = "Copy";
        }, 1800);
      } catch {
        showError("Could not copy the report. Please try again.");
      }
    });

    function renderReport(report, errors) {
      progressCaption.textContent = "Research complete";
      document.querySelectorAll(".pipeline-stage").forEach((stage) => {
        stage.classList.remove("active");
        stage.classList.add("done");
      });
      document.getElementById("status-title").textContent = "Research complete";
      document.getElementById("status-detail").textContent = "Your report is ready below.";

      if (!report) {
        showError(
          errors && errors.length
            ? errors[errors.length - 1]
            : "No report could be produced for this request."
        );
        return;
      }

      reportTitle.textContent = report.title;
      reportSummary.textContent = report.executive_summary;
      reportTextForClipboard = buildReportClipboardText(report);

      keyFindingsEl.innerHTML = "";
      for (const point of report.key_findings || []) {
        const li = document.createElement("li");
        li.textContent = point;
        keyFindingsEl.appendChild(li);
      }

      findingsContainer.innerHTML = "";
      for (const finding of report.findings || []) {
        const block = document.createElement("div");
        block.className = "finding";

        const heading = document.createElement("h4");
        heading.textContent = finding.topic;
        block.appendChild(heading);

        const summary = document.createElement("p");
        summary.textContent = finding.summary;
        block.appendChild(summary);

        if (finding.key_points && finding.key_points.length) {
          const ul = document.createElement("ul");
          for (const kp of finding.key_points) {
            const li = document.createElement("li");
            li.textContent = kp;
            ul.appendChild(li);
          }
          block.appendChild(ul);
        }

        if (finding.sources && finding.sources.length) {
          const sourcesLine = document.createElement("p");
          sourcesLine.className = "finding-sources";
          sourcesLine.innerHTML =
            "Sources: " +
            finding.sources
              .map((url) => `<a href="${url}" target="_blank" rel="noopener">${hostnameOf(url)}</a>`)
              .join(", ");
          block.appendChild(sourcesLine);
        }

        findingsContainer.appendChild(block);
      }

      sourcesList.innerHTML = "";
      for (const url of report.sources || []) {
        const li = document.createElement("li");
        li.innerHTML = `<a href="${url}" target="_blank" rel="noopener">${hostnameOf(url)}</a><span class="source-url">${url}</span>`;
        sourcesList.appendChild(li);
      }

      reportDivider.style.display = "block";
      reportSection.style.display = "block";
    }

    function parseSseBlock(block) {
      let eventName = null;
      let dataLine = null;
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine = line.slice(5).trim();
      }
      if (!eventName || dataLine === null) return null;
      try {
        return { event: eventName, data: JSON.parse(dataLine) };
      } catch {
        return null;
      }
    }

    async function runResearch(prompt) {
      resetUI();
      sendBtn.disabled = true;
      sendBtn.textContent = "Researching…";
      promptInput.disabled = true;

      try {
        const response = await fetch(`${API_BASE}/api/research`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            preferred_provider: currentSettings.provider,
            custom_api_keys: buildCustomApiKeysPayload(),
          }),
        });

        if (!response.ok || !response.body) {
          showError(`Request failed (status ${response.status}).`);
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const rawBlock = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);

            const parsed = parseSseBlock(rawBlock);
            if (!parsed) continue;

            if (parsed.event === "progress") {
              if (parsed.data.node && parsed.data.node !== "start") {
                setProgressState(parsed.data.node);
              }
            } else if (parsed.event === "final_report") {
              renderReport(parsed.data.final_report, parsed.data.errors);
            } else if (parsed.event === "error") {
              showError(parsed.data.message || "The research pipeline failed.");
            }
          }
        }
      } catch (err) {
        showError(`Could not reach the research API (${err.message}).`);
      } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = "Send";
        promptInput.disabled = false;
      }
    }

    sendBtn.addEventListener("click", () => {
      const value = promptInput.value.trim();
      if (!value) {
        showError("Enter a research question first.");
        return;
      }
      runResearch(value);
    });

    promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
      }
    });
