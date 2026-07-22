# Deploying to Render (no credit card required)

Render's free tier builds directly from a Dockerfile — no changes needed to
how the app itself works, just a different deploy target than AWS ECS.

## Important free-tier caveat, specific to our 2-service setup

**Render's free web services spin down after 15 minutes of no incoming
traffic, and take ~30-60 seconds to wake up on the next request.** This
affects us in a specific way:

- If the **frontend** sleeps, the first person to open the site just waits
  ~30-60s for it to wake — mildly annoying, not broken.
- If the **API** sleeps while the frontend is awake, the first chat/upload
  after a quiet period will hit the same delay — and because our
  `REQUEST_TIMEOUT` in `frontend/app.py` is 120s, it should still succeed,
  just feel slow on that first request.
- Both services sleep independently, so it's possible for the frontend to be
  awake (someone's looking at the page) while the API is asleep (nobody's
  chatted in >15 min) — the very next chat message will have that cold-start
  delay.

None of this breaks anything, it's just the tradeoff for zero-cost hosting.
If this matters for a demo, open both URLs a minute before you need them so
they're warm.

## One-time setup

### 1. Create a Render account
Go to https://render.com → sign up with GitHub (recommended — it also
simplifies connecting your repo) or email. **No card required** for free
web services.

### 2. Generate your PORTAL_API_KEY (if you haven't already)
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Save this value — you'll enter it in the Render dashboard for both services,
and it must be identical in both places.

### 3. Deploy via Blueprint (recommended — provisions both services at once)

This repo includes `render.yaml` at the root, which describes both services.

1. In the Render dashboard: **New** → **Blueprint**
2. Connect your GitHub account if prompted, then select the
   `rag-document-portal` repo
3. Render reads `render.yaml` and shows you both services it's about to
   create (`rag-portal-api`, `rag-portal-frontend`)
4. Before clicking Apply, you'll be prompted to fill in the `sync: false`
   environment variables (Render deliberately doesn't let secrets live in
   the YAML file/git history):
   - **rag-portal-api**: `GROQ_API_KEY`, `PORTAL_API_KEY`
   - **rag-portal-frontend**: `PORTAL_API_KEY` (same value as the API's)
5. Click **Apply** — Render builds both Docker images and deploys them.
   First build takes a while (5-10+ minutes) since it's installing FAISS,
   sentence-transformers, PyMuPDF, etc. from scratch.

### 4. Fix the cross-service URLs (one-time, after first deploy)

`render.yaml` has placeholder URLs for `ALLOWED_ORIGINS` and `API_BASE`
because Render assigns the real `*.onrender.com` URLs only after the
services are created. Once both are live:

1. Note the real URLs shown in the Render dashboard for each service
   (e.g. `https://rag-portal-api-xxxx.onrender.com`)
2. Go to **rag-portal-api** → Environment → update `ALLOWED_ORIGINS` to the
   frontend's real URL
3. Go to **rag-portal-frontend** → Environment → update `API_BASE` to the
   API's real URL
4. Both services redeploy automatically when you save an environment variable

### 5. Test it

Open the frontend's `https://rag-portal-frontend-xxxx.onrender.com` URL.
Expect the first load to be slow (cold start) — give it up to a minute.
Upload a document, ask a question, same as your local testing.

## Alternative: manual setup (if you'd rather not use the Blueprint)

Create each service individually in the Render dashboard:
**New → Web Service → connect repo → Runtime: Docker → Dockerfile Path:**
`docker/api.Dockerfile` (or `docker/frontend.Dockerfile` for the second one)
**→ Plan: Free**, then set the same environment variables listed in step 3
above manually in each service's Environment tab.

## Ongoing deploys

Render auto-deploys on every push to your connected branch (`main`) by
default — no GitHub Actions needed for this path, Render handles the
build+deploy itself by watching the repo directly. If you'd rather keep
using GitHub Actions for lint/SonarQube gating before Render deploys, that's
still possible (CI as a required check, Render still watching `main`), but
isn't required to get this working.

## What's different from the AWS/ECS plan

- No IAM roles, no ECR, no ECS task definitions needed — Render handles all
  of that internally.
- No ALB/ACM setup needed for HTTPS — Render provisions a free TLS
  certificate and HTTPS URL automatically for every service, out of the box.
- Encryption at rest: Render encrypts data at rest on their infrastructure
  by default, similar to the Fargate default noted in `infra/encryption-at-rest.md`.
- The `infra/ecs/`, `infra/iam/`, and `infra/alb/` folders in this repo are
  still here, unused for now — kept in case you move to AWS later once card
  access isn't a blocker. Nothing about the Render deploy depends on them.
