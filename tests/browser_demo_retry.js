// Evaluate this expression on browser_demo_server.py's page in a real browser.
// Each request reaches the real HTTP app and disposable Postgres before the
// wrapper delays or loses its response. Providers use deterministic HTTP mocks.
(async () => {
  const results = [];
  const originalFetch = window.fetch.bind(window);
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const wait = async (predicate) => {
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if (await predicate()) return;
      await new Promise(resolve => setTimeout(resolve, 30));
    }
    throw new Error("Timed out waiting for browser/worker");
  };
  const state = async () => (await originalFetch("/api/state")).json();
  const settled = async () => {
    let current;
    await wait(async () => {
      current = await state();
      const messages = [...current.patient, ...current.asha];
      check(!messages.some(m => m.status === "failed"), "Worker failed a browser request");
      return messages.every(m => !m.status || m.status === "done");
    });
    return current;
  };
  const run = async (name, test) => {
    try {
      const evidence = await test();
      results.push({ name, status: "PASS", ...evidence });
    } catch (error) {
      results.push({ name, status: "FAIL", error: error.message });
    } finally {
      window.fetch = originalFetch;
    }
  };
  const form = document.getElementById("patient-form");
  const box = document.getElementById("patient-text");
  const edit = value => {
    box.value = value;
    box.dispatchEvent(new Event("input", { bubbles: true }));
  };
  const idle = () => wait(() => !form.dataset.sending);
  const inbound = s => s.patient.filter(m => m.direction === "inbound");

  await run("empty input and double submit with edits during success", async () => {
    const before = inbound(await settled()).length;
    edit(" \n ");
    form.requestSubmit();
    check(inbound(await state()).length === before, "Blank message was queued");
    let release;
    let accepted;
    const committed = new Promise(resolve => { accepted = resolve; });
    const requests = [];
    window.fetch = async (url, options) => {
      const response = await originalFetch(url, options);
      if (url === "/api/messages" && options?.method === "POST") {
        requests.push(JSON.parse(options.body));
        accepted();
        await new Promise(resolve => { release = resolve; });
      }
      return response;
    };
    edit("namaste first message");
    form.requestSubmit();
    form.requestSubmit();
    await committed;
    edit("namaste next draft");
    release();
    await idle();
    check(box.value === "namaste next draft", "Successful send erased the next draft");
    window.fetch = originalFetch;
    form.requestSubmit();
    await idle();
    const current = inbound(await settled());
    check(requests.length === 1, "Double submit made multiple POSTs");
    check(current.length === before + 2, "Both distinct drafts must be stored once");
    check(current.slice(-2).map(m => m.body).join("|") ===
      "namaste first message|namaste next draft", "Wrong draft reached Postgres");
    return { queued: 2, duplicatePosts: 0, nextDraftPreserved: true };
  });

  await run("lost text response retries once and edited text gets a new ID", async () => {
    await idle();
    const before = inbound(await settled()).length;
    const requests = [];
    let lose = true;
    window.fetch = async (url, options) => {
      const response = await originalFetch(url, options);
      if (url === "/api/messages" && options?.method === "POST") {
        requests.push(JSON.parse(options.body));
        if (lose) { lose = false; throw new TypeError("Response lost after commit"); }
      }
      return response;
    };
    edit("namaste retry original");
    form.requestSubmit();
    await idle();
    check(box.value === "namaste retry original", "Failed send erased draft");
    form.requestSubmit();
    await idle();
    edit("namaste edited after retry");
    form.requestSubmit();
    await idle();
    const current = inbound(await settled());
    check(requests[0].client_id === requests[1].client_id, "Retry changed text ID");
    check(requests[1].client_id !== requests[2].client_id, "Edited message reused text ID");
    check(current.length === before + 2, "Text retry duplicated or lost a message");
    return { posts: requests.length, queued: 2 };
  });

  await run("editing and restoring an in-flight draft retains its uncertain ID", async () => {
    const before = inbound(await settled()).length;
    const requests = [];
    let release;
    let accepted;
    const committed = new Promise(resolve => { accepted = resolve; });
    window.fetch = async (url, options) => {
      const response = await originalFetch(url, options);
      if (url === "/api/messages" && options?.method === "POST") {
        requests.push(JSON.parse(options.body));
        if (requests.length === 1) {
          accepted();
          await new Promise(resolve => { release = resolve; });
          throw new TypeError("Response lost after commit");
        }
      }
      return response;
    };
    edit("namaste restore this draft");
    form.requestSubmit();
    await committed;
    edit("namaste another draft");
    edit("namaste restore this draft");
    release();
    await idle();
    form.requestSubmit();
    await idle();
    check(requests[0].client_id === requests[1].client_id, "Editing in flight lost retry identity");
    check(inbound(await settled()).length === before + 1, "Restored draft retry queued twice");
    return { posts: 2, queued: 1 };
  });

  await run("committed photo with lost response is one message and prescription case", async () => {
    await idle();
    const before = await settled();
    // A synthetic committed fixture is fetched through the local server only.
    const bytes = await (await originalFetch("/test-photo")).arrayBuffer();
    const upload = async (name, data = bytes) => {
      const transfer = new DataTransfer();
      transfer.items.add(new File([data], name, { type: "image/jpeg", lastModified: 1234 }));
      const input = document.getElementById("photo");
      input.files = transfer.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
      await idle();
    };
    const requests = [];
    let lose = true;
    window.fetch = async (url, options) => {
      const response = await originalFetch(url, options);
      if (String(url).startsWith("/api/photos?") && options?.method === "POST") {
        requests.push({ id: new URL(url, location.href).searchParams.get("client_id"),
          response: await response.clone().json() });
        if (lose) { lose = false; throw new TypeError("Response lost after commit"); }
      }
      return response;
    };
    await upload("prescription.jpg");
    await settled();
    await upload("prescription.jpg");
    const retried = await settled();
    const images = s => inbound(s).filter(m => m.kind === "image").length;
    const prescriptions = s => s.cases.filter(c => c.cause === "prescription to confirm").length;
    check(images(retried) === images(before) + 1, "Photo retry queued another message");
    check(prescriptions(retried) === prescriptions(before) + 1, "Photo retry created another case");
    check(requests[0].id === requests[1].id && requests[1].response.duplicate,
      "Unresolved photo did not reuse its ID");
    // An acknowledged photo can be deliberately sent again.
    await upload("prescription.jpg");
    await settled();
    check(requests[2].id !== requests[1].id, "Intentional resend reused acknowledged ID");
    // A different file while one is unresolved must have a separate identity.
    lose = true;
    await upload("prescription.jpg");
    await settled();
    const other = await (await originalFetch("/test-photo?different=true")).arrayBuffer();
    await upload("other.jpg", other);
    const final = await settled();
    check(requests[3].id !== requests[4].id, "Different photo reused unresolved ID");
    check(images(final) === images(before) + 4, "Wrong final photo queue count");
    check(prescriptions(final) === prescriptions(before) + 4, "Wrong final prescription case count");
    return { retryQueued: 1, retryCases: 1, intentionalResendAndDifferentPhoto: true, requests };
  });
  window.fetch = originalFetch;
  return { results, passed: results.every(r => r.status === "PASS"), state: await settled() };
})()
