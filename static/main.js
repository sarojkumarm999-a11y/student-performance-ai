const $ = (sel) => document.querySelector(sel);

function toNumberOrNull(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function formToStudentPayload(form) {
  const fd = new FormData(form);
  return {
    age: toNumberOrNull(fd.get("age")),
    gender: fd.get("gender") || null,
    attendance_pct: toNumberOrNull(fd.get("attendance_pct")),
    study_hours_per_day: toNumberOrNull(fd.get("study_hours_per_day")),
    prev_gpa: toNumberOrNull(fd.get("prev_gpa")),
    assignments_submitted: toNumberOrNull(fd.get("assignments_submitted")),
    extracurricular_score: toNumberOrNull(fd.get("extracurricular_score")),
    parent_education: fd.get("parent_education") || null,
    internet_access: toNumberOrNull(fd.get("internet_access")),
    part_time_job: toNumberOrNull(fd.get("part_time_job")),
    counseling_sessions: toNumberOrNull(fd.get("counseling_sessions")),
    behavioral_notes: fd.get("behavioral_notes") || "",
  };
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error || `Request failed with status ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function setGradeBadge(grade) {
  const badge = $("#outGradeBadge");
  badge.className = "badge";
  badge.textContent = grade || "—";

  if (!grade || grade === "—") {
    badge.classList.add("status-badge-default");
    return;
  }

  const g = grade.toUpperCase();
  if (g.startsWith("A")) badge.classList.add("grade-badge-a");
  else if (g.startsWith("B")) badge.classList.add("grade-badge-b");
  else if (g.startsWith("C")) badge.classList.add("grade-badge-c");
  else if (g.startsWith("D")) badge.classList.add("grade-badge-d");
  else if (g.startsWith("F")) badge.classList.add("grade-badge-f");
  else badge.classList.add("status-badge-default");
}

function setPassFailBadge(status) {
  const badge = $("#outPassFailBadge");
  badge.className = "badge";

  if (status === 1) {
    badge.textContent = "PASS";
    badge.classList.add("status-badge-pass");
  } else if (status === 2) {
    badge.textContent = "AT RISK";
    badge.classList.add("status-badge-warn");
  } else if (status === 0) {
    badge.textContent = "FAIL";
    badge.classList.add("status-badge-fail");
  } else {
    badge.textContent = "—";
    badge.classList.add("status-badge-default");
  }
}

function renderFormattedInsights(insightsText) {
  const container = $("#outInsightsContainer");
  if (!insightsText || !insightsText.trim()) {
    container.innerHTML = `<p class="placeholder-text">Click <strong>"Predict + AI Insights"</strong> to generate personalized academic advising recommendations.</p>`;
    return;
  }

  // Format line breaks and bullet points into styled HTML
  const lines = insightsText.trim().split("\n");
  let html = "";
  let inList = false;

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      if (!inList) {
        html += "<ul style='margin:4px 0 10px 18px; padding:0;'>";
        inList = true;
      }
      html += `<li style='margin-bottom:4px;'>${trimmed.substring(2)}</li>`;
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      if (trimmed.endsWith(":")) {
        html += `<h4 style="margin:10px 0 4px; color:#A78BFA; font-size:13px;">${trimmed}</h4>`;
      } else {
        html += `<p style="margin:0 0 8px;">${trimmed}</p>`;
      }
    }
  });

  if (inList) html += "</ul>";
  container.innerHTML = html;
}

function renderPrediction(resp) {
  const p = resp?.predictions || {};
  const gpa = typeof p.predicted_gpa === "number" ? p.predicted_gpa : null;

  if (gpa !== null) {
    $("#outGpa").textContent = gpa.toFixed(2);
    const pct = Math.min(Math.max((gpa / 10.0) * 100, 0), 100);
    $("#outGpaBar").style.width = `${pct}%`;
  } else {
    $("#outGpa").textContent = "—";
    $("#outGpaBar").style.width = "0%";
  }

  setGradeBadge(p.grade);
  setPassFailBadge(p.pass_fail);

  const raw = JSON.stringify(resp, null, 2);
  $("#outRaw").textContent = raw;

  const insights = resp?.insights;
  const insightsErr = resp?.insights_error;

  renderFormattedInsights(insights);

  if (insightsErr) {
    $("#outInsightsErr").hidden = false;
    $("#outInsightsErr").textContent = insightsErr;
  } else {
    $("#outInsightsErr").hidden = true;
    $("#outInsightsErr").textContent = "";
  }
}

function fillExample() {
  const ex = {
    age: 19,
    gender: "Female",
    attendance_pct: 88.5,
    study_hours_per_day: 4.5,
    prev_gpa: 8.3,
    assignments_submitted: 90,
    extracurricular_score: 7,
    parent_education: "Bachelor",
    internet_access: 1,
    part_time_job: 0,
    counseling_sessions: 2,
    behavioral_notes: "Highly engaged student, demonstrates active participation, attends tutoring regularly, and shows strong teamwork.",
  };
  for (const [k, v] of Object.entries(ex)) {
    const el = document.querySelector(`[name="${k}"]`);
    if (!el) continue;
    el.value = String(v);
  }
}

function resetForm() {
  $("#predictForm").reset();
  renderPrediction({ predictions: {} });
}

async function showMetrics() {
  const dlg = $("#metricsDialog");
  $("#metricsBody").textContent = "Loading metrics dataset…";
  $("#metricGpaMae").textContent = "…";
  $("#metricGpaR2").textContent = "…";
  $("#metricGradeAcc").textContent = "…";
  $("#metricPassF1").textContent = "…";

  dlg.showModal();
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();
    $("#metricsBody").textContent = JSON.stringify(data, null, 2);

    if (data.gpa) {
      $("#metricGpaMae").textContent = typeof data.gpa.mae === "number" ? data.gpa.mae.toFixed(3) : "—";
      $("#metricGpaR2").textContent = typeof data.gpa.r2 === "number" ? (data.gpa.r2 * 100).toFixed(1) + "%" : "—";
    }
    if (data.grade) {
      $("#metricGradeAcc").textContent = typeof data.grade.accuracy === "number" ? (data.grade.accuracy * 100).toFixed(1) + "%" : "—";
    }
    if (data.pass_fail) {
      $("#metricPassF1").textContent = typeof data.pass_fail.f1 === "number" ? (data.pass_fail.f1 * 100).toFixed(1) + "%" : "—";
    }
  } catch (e) {
    $("#metricsBody").textContent = String(e);
  }
}

function setBusy(isBusy, buttonId, spinnerId) {
  const btn = $(buttonId);
  const spin = $(spinnerId);
  if (isBusy) {
    btn.disabled = true;
    spin.hidden = false;
  } else {
    btn.disabled = false;
    spin.hidden = true;
  }
}

function wire() {
  const form = $("#predictForm");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setBusy(true, "#btnPredict", "#spinPredict");
    try {
      const student = formToStudentPayload(form);
      const resp = await postJson("/api/predict", { student });
      renderPrediction(resp);
    } catch (err) {
      renderPrediction({ predictions: {}, insights_error: String(err) });
    } finally {
      setBusy(false, "#btnPredict", "#spinPredict");
    }
  });

  $("#btnPredictInsights").addEventListener("click", async () => {
    setBusy(true, "#btnPredictInsights", "#spinInsights");
    try {
      const student = formToStudentPayload(form);
      const resp = await postJson("/api/predict-with-insights", { student });
      renderPrediction(resp);
    } catch (err) {
      renderPrediction({ predictions: {}, insights_error: String(err) });
    } finally {
      setBusy(false, "#btnPredictInsights", "#spinInsights");
    }
  });

  $("#btnExample").addEventListener("click", fillExample);
  $("#btnReset").addEventListener("click", resetForm);
  $("#btnMetrics").addEventListener("click", showMetrics);
  $("#btnCloseMetrics").addEventListener("click", () => $("#metricsDialog").close());

  $("#btnToggleRaw").addEventListener("click", () => {
    $("#outRaw").classList.toggle("hidden");
  });

  $("#btnToggleMetricsJson").addEventListener("click", () => {
    $("#metricsBody").classList.toggle("hidden");
  });
}

document.addEventListener("DOMContentLoaded", wire);
