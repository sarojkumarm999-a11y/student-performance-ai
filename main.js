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
    const msg = data?.error || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

function renderPrediction(resp) {
  const p = resp?.predictions || {};
  const gpa = typeof p.predicted_gpa === "number" ? p.predicted_gpa.toFixed(2) : "—";
  $("#outGpa").textContent = gpa;
  $("#outGrade").textContent = p.grade ?? "—";
  $("#outPassFail").textContent = p.pass_fail === 1 ? "PASS" : p.pass_fail === 0 ? "FAIL" : "—";

  const raw = JSON.stringify(resp, null, 2);
  $("#outRaw").textContent = raw;

  const insights = resp?.insights;
  const insightsErr = resp?.insights_error;
  $("#outInsights").textContent = insights ? insights : "—";

  if (insightsErr) {
    $("#outInsightsErr").hidden = false;
    $("#outInsightsErr").textContent = `Claude insights unavailable: ${insightsErr}`;
  } else {
    $("#outInsightsErr").hidden = true;
    $("#outInsightsErr").textContent = "";
  }
}

function fillExample() {
  const ex = {
    age: 19,
    gender: "Female",
    attendance_pct: 88,
    study_hours_per_day: 4.5,
    prev_gpa: 3.3,
    assignments_submitted: 90,
    extracurricular_score: 7,
    parent_education: "Bachelor",
    internet_access: 1,
    part_time_job: 0,
    counseling_sessions: 2,
    behavioral_notes: "Highly engaged, consistent improvement, strong teamwork and leadership.",
  };
  for (const [k, v] of Object.entries(ex)) {
    const el = document.querySelector(`[name="${k}"]`);
    if (!el) continue;
    el.value = String(v);
  }
}

async function showMetrics() {
  const dlg = $("#metricsDialog");
  $("#metricsBody").textContent = "Loading…";
  dlg.showModal();
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();
    $("#metricsBody").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("#metricsBody").textContent = String(e);
  }
}

function wire() {
  const form = $("#predictForm");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#btnPredict").disabled = true;
    try {
      const student = formToStudentPayload(form);
      const resp = await postJson("/api/predict", { student });
      renderPrediction(resp);
    } catch (err) {
      renderPrediction({ predictions: {}, insights_error: String(err) });
    } finally {
      $("#btnPredict").disabled = false;
    }
  });

  $("#btnPredictInsights").addEventListener("click", async () => {
    $("#btnPredictInsights").disabled = true;
    try {
      const student = formToStudentPayload(form);
      const resp = await postJson("/api/predict-with-insights", { student });
      renderPrediction(resp);
    } catch (err) {
      renderPrediction({ predictions: {}, insights_error: String(err) });
    } finally {
      $("#btnPredictInsights").disabled = false;
    }
  });

  $("#btnExample").addEventListener("click", fillExample);
  $("#btnMetrics").addEventListener("click", showMetrics);
  $("#btnCloseMetrics").addEventListener("click", () => $("#metricsDialog").close());
}

document.addEventListener("DOMContentLoaded", wire);

