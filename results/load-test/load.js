// k6 load test for the library API. One run holds a constant number of virtual
// users (VUS) for DURATION; run.sh steps VUS up to find where it breaks.
//
//   MODE=ask     each user asks a question about the ingested video and reads
//                the whole event stream, then pauses THINK seconds
//   MODE=browse  each user lists the library and opens the video, then pauses
//
// Questions go to the extractive answerer, so the run measures Gist's own
// stored-evidence scoring and selection, not a hosted model's latency or
// rate limit.

import http from "k6/http";
import { check, sleep } from "k6";

const API = __ENV.API || "http://127.0.0.1:8100";
const VIDEO = __ENV.VIDEO;
const MODE = __ENV.MODE || "ask";
const THINK = Number(__ENV.THINK || 1);

// Video-MME questions about the ingested video (eZ5lFBPpRkM, items 868-1..3).
const QUESTIONS = [
  "What is the relationship of the woman standing next to the main character in the video when he visits Jatilui?",
  "When the people in the video saw the turtles for the first time, how many days into their journey was it?",
  "What does the protagonist of the video record?",
];

export const options = {
  scenarios: {
    load: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS || 1),
      duration: __ENV.DURATION || "60s",
      gracefulStop: "120s",
    },
  },
  summaryTrendStats: ["min", "med", "p(90)", "p(95)", "p(99)", "max", "avg", "count"],
  // Empty thresholds make k6 report each endpoint as its own submetric.
  thresholds: {
    "http_req_duration{name:query}": [],
    "http_req_duration{name:list}": [],
    "http_req_duration{name:detail}": [],
  },
};

export function setup() {
  if (!VIDEO) throw new Error("set VIDEO to the id of a ready video");
}

export default function () {
  if (MODE === "browse") {
    const list = http.get(`${API}/v1/library/videos`, { tags: { name: "list" } });
    check(list, { "list 200": (r) => r.status === 200 });
    const detail = http.get(`${API}/v1/library/videos/${VIDEO}`, { tags: { name: "detail" } });
    check(detail, { "detail 200": (r) => r.status === 200 });
  } else {
    const query = QUESTIONS[(__VU + __ITER) % QUESTIONS.length];
    const res = http.post(
      `${API}/v1/library/videos/${VIDEO}/query`,
      JSON.stringify({ query, answerer: "extractive" }),
      { headers: { "Content-Type": "application/json" }, tags: { name: "query" }, timeout: "180s" },
    );
    // A stream can return 200 and still fail inside; only a `done` event counts.
    check(res, {
      "query 200": (r) => r.status === 200,
      "query completed": (r) => typeof r.body === "string" && r.body.includes("event: done"),
    });
  }
  sleep(THINK);
}
