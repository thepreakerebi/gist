// k6 load test for the library API. One run holds a constant number of virtual
// users (VUS) for DURATION; run.sh steps VUS up to find where it breaks.
//
//   MODE=ask     each user asks a question about the ingested video and reads
//                the whole event stream, then pauses THINK seconds
//   MODE=varied  as ask, but each iteration draws from 40 different questions,
//                so most requests select new evidence and cut new clips
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

// Forty questions a viewer of a Bali travel guide might ask. Distinct wording
// leads to distinct selections, so clips are cut instead of found on disk.
const VARIED = [
  "Which beach does he recommend first?",
  "What does he say about the rice terraces?",
  "How do they get to the Gili islands?",
  "What animal do they see on Komodo?",
  "Where do they go snorkelling?",
  "What does he eat at the market?",
  "How much does the boat trip cost?",
  "What temple do they visit?",
  "When do they watch the sunrise?",
  "What does he warn about scooters?",
  "Which waterfall do they visit?",
  "What is the weather like during the trip?",
  "Where do they stay in Ubud?",
  "What does he say about the monkeys?",
  "Which island has no cars?",
  "What do they do at Nusa Penida?",
  "How long is the ferry ride?",
  "What does the guide say about the dragons?",
  "What drink does he try?",
  "Where is the swing they ride?",
  "What does he say about tourists?",
  "What is the best time of year to visit?",
  "Do they go diving or only snorkelling?",
  "What does he film from the drone?",
  "What colour is the sand on the pink beach?",
  "Who joins him on the hike?",
  "What does he say about the traffic in Canggu?",
  "Which viewpoint do they climb to?",
  "What souvenir does he buy?",
  "How does he describe the food?",
  "What happens on the boat in rough water?",
  "Where do they see manta rays?",
  "What advice does he give about money?",
  "What does he say at the end of the video?",
  "Which place does he call his favourite?",
  "How many days does the whole trip take?",
  "What does the woman next to him say?",
  "What does he record with his camera?",
  "What is shown at the very beginning?",
  "Where do they watch the sunset?",
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
    const pool = MODE === "varied" ? VARIED : QUESTIONS;
    const query = pool[(__VU * 7 + __ITER) % pool.length];
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
