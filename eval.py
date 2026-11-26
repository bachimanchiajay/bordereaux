# MASTER TASK: Complete Phase 4 & Phase 5 — SNOW Docker E2E Integration and Microsoft 365 Agents Playground Testing

Act as a Principal Python Backend Engineer, Senior AI Architect, Docker Engineer, Microsoft Teams Bot Framework Specialist and QA Automation Engineer.

I have three interconnected services:

1. snow-knowledge-service
2. chat-orchestrator
3. snow-research-teams-bot

My objective is to complete the entire application flow using Docker images and validate it through Microsoft 365 Agents Playground.

IMPORTANT: All three application services must run inside Docker containers. Do not run Python application services directly on my Mac.

I have already completed:

- Phase 1: Repository and configuration audit.
- Phase 2: Docker integration setup.
- Phase 3: Docker image builds.

Phase 4 and Phase 5 are partially completed.

Do not generate any Markdown or PDF technical documentation in this execution. Documentation will be generated separately after complete E2E validation.

# 1. STRICT ENTERPRISE NETWORKING AND SECURITY REQUIREMENTS

This is a financial-services company environment.

The following tools and services are STRICTLY PROHIBITED:

- ngrok
- Microsoft Dev Tunnels
- LocalTunnel
- Cloudflare Tunnel
- Any other public tunnelling service.
- Any unapproved external tunnelling or reverse-proxy service.

Do not:
- Install these tools.
- Start these tools.
- Add them to Docker Compose.
- Add their configuration to environment files.
- Create public tunnel URLs.
- Expose internal application services through an unapproved public endpoint.

Use:
- Docker Compose networking.
- Internal Docker service-name resolution.
- Localhost port mapping where appropriate.
- Existing Docker network configuration.
- Existing approved corporate networking infrastructure.

The preferred local testing flow is:

Microsoft 365 Agents Playground
→ Locally accessible bot endpoint
→ Teams Bot Docker container
→ Chat Orchestrator Docker container
→ Knowledge Service Docker container
→ PostgreSQL/pgvector
→ Real Tarvos LLM Gateway

First inspect whether the installed Playground can directly reach the locally exposed Docker endpoint.

Do not assume that local connectivity is supported in every Playground configuration.

If external HTTPS connectivity is required, STOP and explain the approved enterprise alternatives. Do not introduce a tunnel as a workaround.

Preserve production authentication and security controls.

# 2. CURRENT VERIFIED IMPLEMENTATION

Use these previous execution results as the baseline.

## Docker infrastructure

- Docker Compose E2E environment exists.
- Shared network: snow-e2e-net.
- PostgreSQL with pgvector is healthy.
- Database initialization passed on rerun.
- Persistent database and vector data must be preserved.

## Application images

- snow-knowledge-service:e2e
- snow-orchestrator:e2e
- snow-teams-bot:e2e

## Application ports

- Knowledge service: 8012.
- Chat orchestrator: 8000.
- Teams bot: 3978.

## Knowledge service

- Approximately 56,266 vector records.
- Embedding dimension: 768.
- Embedding model: nomic-embed-text-v1.5.
- Reranker: ms-marco-MiniLM-L-6-v2.
- Dense retrieval tested.
- Hybrid retrieval tested.
- Cross-encoder reranking tested.
- Ticket exclusion tested.

## Chat orchestrator

- Container healthy.
- Successfully communicates with knowledge service.
- Real Tarvos LLM integration tested.
- /snow/chat returned HTTP 200.
- Actual generated response and five source tickets received.

## Existing test results

- Orchestrator smoke test: 12/12 passed.
- Bot unit tests: 155 passed.
- Orchestrator tests: 54 passed.
- Knowledge tests: 162 passed.
- Knowledge tests: 5 skipped due to Helm dependency.
- Knowledge tests: 1 failed due to .gitignore environmental check.
- Knowledge tests: 97 skipped due to missing database configuration.

Do not unnecessarily rebuild or modify the working RAG and LLM integration.

# 3. PHASE 4 — COMPLETE SERVICE STARTUP

## Issue A: Teams bot startup

The bot has not successfully started.

The previously identified missing configuration:

1. ORCHESTRATOR_SCOPE
2. bot-ca.pem

Inspect the actual source code and identify:

- Where ORCHESTRATOR_SCOPE is consumed.
- Expected value and format.
- Required Entra application registration.
- Required API permissions.
- Whether the scope represents the orchestrator API.
- How the bot uses bot-ca.pem.
- Expected certificate format.
- Certificate mount path.
- Whether the CA bundle is mandatory for the current connection.

Do not invent values.

Do not create a dummy certificate.

Do not disable TLS verification.

If the real configuration is missing, ask me to provide it.

Continue independent tests wherever possible.

## Issue B: Start the complete Docker stack

Once prerequisites are available, start in dependency order:

1. PostgreSQL.
2. Database initialization.
3. snow-knowledge-service.
4. chat-orchestrator.
5. snow-research-teams-bot.

Validate:

- Container health.
- Application startup.
- Correct ports.
- Docker DNS.
- Internal service communication.
- Database connectivity.
- Bot health endpoint.

Use existing Compose service names and configuration.

Preserve all persistent volumes and existing vector records.

# 4. PHASE 5 — COMPLETE REMAINING INTEGRATION TESTS

Previously reported scenario status:

1. All three services start: Partial.
2. Bot endpoint responds: Not run.
3. Bot calls orchestrator: Partial.
4. Orchestrator communicates with knowledge service: Passed.
5. Knowledge retrieval works: Passed.
6. LLM generates a response: Passed.
7. Response returns through orchestrator: Passed.
8. Bot sends final response through Bot Framework: Not run.

Complete only the outstanding scenarios and run regression tests when necessary.

## Scenario 1: All three services start

Start and validate all three application containers.

Verify actual health status and logs.

## Scenario 2: Bot endpoint responds

- Test the actual health endpoint.
- Verify port 3978.
- Inspect bot logs.
- Confirm the messaging route exists.
- Verify local endpoint connectivity.

Distinguish health endpoint success from authenticated messaging success.

## Scenario 3: Bot calls orchestrator

The previous bot client test received a real answer and five source tickets from the orchestrator.

However, the actual Entra authentication and Bot Framework flow have not been validated.

Verify:

- ORCHESTRATOR_SCOPE.
- Bot-to-orchestrator authentication.
- Internal service URL.
- Request payload.
- Response payload.
- Error handling.

Use the actual orchestrator, not a mock.

## Scenario 8: Bot Framework response

Once bot configuration is ready:

1. Submit a real Bot Framework activity.
2. Verify bot activity handling.
3. Verify orchestrator invocation.
4. Verify knowledge retrieval.
5. Verify real Tarvos LLM response.
6. Verify final bot response delivery.

Do not claim E2E success using direct orchestrator API calls alone.

# 5. MICROSOFT 365 AGENTS PLAYGROUND AUTHENTICATION ISSUE

Microsoft 365 Agents Playground is already installed.

The Playground can reach the Docker bot endpoint, but the bot is rejecting incoming requests.

Observed logs:

GET /health HTTP/1.1 200 OK

POST /api/messages HTTP/1.1 401 Unauthorized

JWT token validation failed:
Entra inbound token caller app is not allowed.

The installed SDK version reported in the previous execution is 2.1.1.

Investigate the exact SDK package and authentication implementation.

## Required investigation

1. Inspect requirements.txt.
2. Identify the exact Teams SDK package and installed version inside the bot container.
3. Inspect bot application initialization.
4. Inspect the incoming JWT token validation logic.
5. Identify why the Playground caller application is rejected.
6. Determine whether Playground sends unauthenticated requests or authenticated Entra tokens in the current testing mode.
7. Determine whether SDK 2.1.1 supports DANGEROUSLY_ALLOW_UNAUTHENTICATED_REQUESTS.
8. Inspect how local Playground testing is configured.
9. Identify whether the authentication failure is due to local unauthenticated testing mode or incorrect Entra configuration.

Do not assume the root cause without inspecting the implementation.

## Local Playground testing configuration

If the installed SDK supports it, evaluate the documented local-testing option:

DANGEROUSLY_ALLOW_UNAUTHENTICATED_REQUESTS=true

Or its supported Python initialization equivalent.

Only enable this in a separate local Playground testing configuration.

Do not:
- Enable unauthenticated requests globally.
- Disable authentication in the production configuration.
- Change production Entra credentials.
- Remove token validation from the application.
- Disable TLS verification.
- Bypass security checks in the normal deployment.

Keep separate configurations for:

1. Local Playground testing.
2. Normal authenticated bot operation.
3. Production deployment.

Use a Docker Compose override or another isolated configuration approach, after inspecting the existing setup.

Verify that the normal authenticated configuration continues enforcing token validation.

## Playground connectivity validation

Verify whether the Playground can access the bot through the local Docker port mapping.

Do not configure ngrok or Dev Tunnels.

If the Playground cannot reach the local endpoint directly:

- Explain the technical limitation.
- Identify whether an approved internal HTTPS endpoint is required.
- Stop and ask for confirmation before introducing any networking changes.

# 6. KNOWLEDGE SERVICE DATABASE-DEPENDENT TESTS

Previously:

- 162 passed.
- 97 skipped because database configuration was unavailable.
- 5 skipped because Helm was not installed.
- 1 failed due to a .gitignore environmental check.

Investigate the 97 skipped tests.

1. Inspect test fixtures.
2. Identify database requirements.
3. Create a separate disposable PostgreSQL/pgvector test database.
4. Never reset the existing E2E database.
5. Execute the skipped tests against the isolated test database.
6. Report actual results.

# 7. RESOLVE THE .gitignore TEST FAILURE

Inspect the failing test.

The previous execution suggested that Git was unavailable in the test environment.

1. Confirm whether the test requires Git.
2. If required, install Git in the relevant test image using its existing base-image package manager.
3. Rebuild only the affected image.
4. Rerun the test.
5. Confirm whether the failure is resolved.

Do not bypass the test.

# 8. HELM TESTS

Five tests were previously skipped because Helm was unavailable.

Inspect whether Helm is actually necessary.

If appropriate, install the required test dependency in the test environment and execute the tests.

Do not modify the production application image unnecessarily.

Do not mark skipped tests as passed.

# 9. INVESTIGATE EXISTING NEGATIVE TEST RESULTS

Previously observed:

| Scenario | Result |
|---|---|
| Search service unreachable | HTTP 503 |
| Invalid payload | HTTP 422 |
| Empty retrieval | HTTP 200 with no relevant tickets |
| Missing configuration | Configuration error |
| Invalid LLM key / unreachable gateway | HTTP 200 with fallback response |
| Unknown session ID | HTTP 200 |
| LLM timeout | Not tested |
| Bad Entra credentials | Not tested |

Inspect these exact behaviours.

For each:
- Find the implementation.
- Identify exception handling.
- Determine whether the behaviour is intentional.
- Explain operational implications.
- Recommend changes only where justified.

Investigate whether fallback responses could be mistaken for actual LLM-generated responses.

Test an actual LLM timeout separately from gateway unavailability.

Do not modify API contracts without explaining the impact.

# 10. SECURITY REQUIREMENTS

Never print or expose:

- LLM_API_KEY.
- LLM_USE_CASE_ID values.
- CLIENT_SECRET.
- Database passwords.
- Microsoft application credentials.
- Private certificate contents.

Use SET or EMPTY checks only.

A previous diagnostic accidentally printed an LLM credential.

Treat that key as exposed and ensure rotation is completed according to company policy.

Do not include secrets in command output, logs or test reports.

# 11. FINAL ACCEPTANCE CRITERIA

The goal is to achieve:

1. All three Docker application containers healthy.
2. Knowledge service operational.
3. Chat orchestrator operational.
4. Real Tarvos LLM integration operational.
5. Bot authentication working.
6. Bot-to-orchestrator communication working.
7. Orchestrator-to-knowledge communication working.
8. Complete Bot Framework request-response working.
9. Microsoft 365 Agents Playground conversation successfully tested.
10. Negative tests and remaining skipped tests accurately reported.

# 12. EXECUTION RULES

Follow this sequence:

Step 1: Inspect existing configuration and current container status.

Step 2: Identify the exact missing bot authentication and certificate prerequisites.

Step 3: Investigate Playground SDK authentication behaviour.

Step 4: Prepare isolated local Playground testing configuration, if supported.

Step 5: Start the bot using Docker.

Step 6: Complete the outstanding integration tests.

Step 7: Execute the isolated database-dependent tests.

Step 8: Resolve the .gitignore test failure.

Step 9: Investigate the five Helm test skips.

Step 10: Perform complete end-to-end testing.

At each stage:

- Show actual commands.
- Show actual results.
- Explain failures.
- Request missing credentials rather than inventing them.
- Preserve existing working functionality.
- Do not expose secrets.
- Do not use any tunnelling service.

Do not generate technical documentation or PDFs in this task.

First inspect the existing environment and report the exact issues before making changes.
