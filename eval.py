# TASK: Complete Phase 4 and Phase 5 — SNOW Services Docker E2E Testing

Act as a Senior Python Backend Engineer, Docker Engineer, QA Automation Engineer and Microsoft Bot Framework Integration Specialist.

I have completed:

- Phase 1: Repository and configuration audit.
- Phase 2: Docker integration setup.
- Phase 3: Docker image builds.

Phase 4 and Phase 5 are partially completed.

Your responsibility is to resolve the outstanding issues and complete the remaining tests.

IMPORTANT:
- Do not generate Markdown or PDF documentation.
- Do not create technical guides or architecture documents.
- Documentation will be handled separately after successful end-to-end testing.
- All three application services must run inside Docker.
- Do not run Python application services directly on the host.
- Preserve existing working configurations and persistent data.
- Do not unnecessarily modify application logic.
- Do not invent credentials, secrets or certificates.
- Never print actual secret values.

# CURRENT VERIFIED STATUS

## Existing infrastructure

- PostgreSQL with pgvector is healthy.
- Database initialization passed on rerun.
- Knowledge service is healthy.
- Chat orchestrator is healthy.
- Real Tarvos LLM integration works.
- Knowledge service reports approximately 56,266 vectors.
- Embedding dimension: 768.
- Embedding model: nomic-embed-text-v1.5.
- Reranker: ms-marco-MiniLM-L-6-v2.

## Existing Docker images

- snow-knowledge-service:e2e
- snow-orchestrator:e2e
- snow-teams-bot:e2e

## Existing application ports

- Knowledge service: 8012.
- Chat orchestrator: 8000.
- Teams bot: 3978.

## Existing successful tests

- Orchestrator smoke test: 12/12 passed.
- Bot unit tests: 155 passed.
- Orchestrator unit tests: 54 passed.
- Knowledge service tests: 162 passed.

Previously observed additional results:

- Knowledge service: 5 Helm-related tests skipped.
- Knowledge service: 1 .gitignore-related test failed.
- Knowledge service: 97 database-dependent tests skipped.

The real LLM chat flow has already returned HTTP 200, an actual answer and five source tickets.

Do not unnecessarily repeat successful tests unless changes require regression testing.

# PHASE 4 — COMPLETE SERVICE STARTUP AND VALIDATION

## ISSUE 1: Teams bot has not started

The following configuration is currently missing:

1. ORCHESTRATOR_SCOPE
2. bot-ca.pem

Inspect the actual source code and configuration.

Determine:

- Where ORCHESTRATOR_SCOPE is consumed.
- What value and format are expected.
- Which Entra application registration it relates to.
- What authentication permissions are required.
- How bot-ca.pem is loaded.
- Whether the CA bundle is mandatory.
- Which certificate format and mount path are expected.

Do not invent the scope.

Do not generate a dummy certificate.

Do not disable TLS verification.

If real configuration is missing, request it from me and continue independent tests that do not require it.

## ISSUE 2: Start the complete Docker stack

Once the missing bot configuration is supplied:

Start services in dependency order:

1. PostgreSQL.
2. Database initialization.
3. snow-knowledge-service.
4. chat-orchestrator.
5. snow-research-teams-bot.

Validate:

- Container status.
- Health checks.
- Startup logs.
- Container ports.
- Docker DNS resolution.
- Internal service connectivity.
- Database connectivity.
- Bot health endpoint.

Use the existing Docker Compose configuration and service names.

Do not delete existing database volumes or vector records.

# PHASE 5 — COMPLETE REMAINING INTEGRATION TESTS

## ISSUE 3: Scenario 1 — All three services start successfully

Previous result: Partial. The bot was not started.

Complete startup and validate all three application containers.

Report actual health-check results.

## ISSUE 4: Scenario 2 — Bot endpoint responds correctly

Previous result: Not run.

After the bot starts:

- Test its actual health endpoint.
- Verify its listening port.
- Inspect startup logs.
- Verify that the endpoint is accessible from the intended Docker network.

Do not confuse a health endpoint response with successful Bot Framework authentication.

## ISSUE 5: Scenario 3 — Bot calls chat-orchestrator

Previous result: Partially tested.

The existing bot client test executed inside the bot image and received a real orchestrator answer with five source tickets.

However, the actual Entra authentication and Bot Framework messaging flow have not been validated.

Now verify:

- Bot-to-orchestrator authentication.
- ORCHESTRATOR_SCOPE.
- Correct internal service URL.
- Request payload.
- Response payload.
- Error handling.

Use the real orchestrator.

## ISSUE 6: Scenario 8 — Bot returns answer through Bot Framework

Previous result: Not run.

Once the required configuration is available:

1. Submit a real Bot Framework activity.
2. Verify the bot receives the activity.
3. Verify the message handler processes the request.
4. Verify the bot calls the orchestrator.
5. Verify the real LLM response returns.
6. Verify the bot sends the final answer through the Bot Framework adapter.

Do not use mocked responses to claim successful integration.

## ISSUE 7: Knowledge service — 97 skipped database tests

Previous test result:

- 162 passed.
- 97 skipped because a test database was not configured.
- 5 skipped because Helm was unavailable.
- 1 failed due to a .gitignore environmental check.

Investigate the database-dependent tests.

Requirements:

1. Inspect existing test fixtures and database configuration.
2. Identify the exact PostgreSQL/pgvector requirements.
3. Create a separate disposable test database.
4. Never reset or delete the existing E2E database.
5. Run the 97 skipped tests against the isolated database.
6. Record actual passed, failed and skipped counts.

## ISSUE 8: Resolve the .gitignore test failure

Investigate the existing failure.

The previous execution suggested that Git was unavailable in the test environment.

1. Inspect the exact failing test.
2. Confirm whether Git is required.
3. If required, install Git in the relevant test image using the appropriate package manager.
4. Rebuild only the affected image.
5. Rerun the failing test.
6. Confirm the result.

Do not bypass or remove the test.

## ISSUE 9: Helm-related tests

Five tests were previously skipped because Helm was unavailable.

Inspect whether Helm is actually required to run them.

If required, explain the dependency and execute them if feasible.

Do not mark skipped tests as passed.

# ISSUE 10: Investigate the observed negative-test behaviours

Use the existing test results as the baseline.

Investigate these exact observations:

| Scenario | Previously observed behaviour |
|---|---|
| Search service unreachable | HTTP 503 |
| Invalid payload | HTTP 422 |
| Empty retrieval | HTTP 200 with no relevant tickets |
| Missing configuration | Clear missing-configuration error |
| Invalid LLM key / unreachable gateway | HTTP 200 with fallback response |
| Unknown session ID | HTTP 200 |
| LLM gateway timeout | Not tested |
| Bad Entra credentials | Not tested |

For each behaviour:

- Inspect the actual implementation.
- Identify the relevant exception handler.
- Determine whether the behaviour is intentional.
- Check whether fallback responses can be mistaken for genuine LLM answers.
- Investigate unknown session handling.
- Test an actual LLM timeout separately from gateway unavailability.

Do not change API contracts or application logic without explaining the reason and impact.

# FINAL ACCEPTANCE CRITERIA

The objective is to achieve:

1. All three application containers healthy.
2. Knowledge service working.
3. Chat orchestrator working.
4. Real Tarvos LLM integration working.
5. Bot authentication working.
6. Bot-to-orchestrator communication working.
7. Orchestrator-to-knowledge communication working.
8. Complete Bot Framework request-response working.
9. Microsoft 365 Agents Playground test readiness.

# EXECUTION RULES

- Inspect before modifying.
- Preserve the existing working RAG and LLM pipeline.
- Use Docker Compose.
- Run application services only in Docker.
- Never print credentials.
- Do not reset persistent data.
- Do not use mock responses for real E2E acceptance.
- Do not report unexecuted tests as passed.
- Do not mark partially completed phases as completed.

Start by inspecting the current configuration and identifying the exact missing bot prerequisites.

If any required secrets or certificate files are missing, request them from me.

Continue independent tests wherever possible while waiting for those prerequisites.

At the end, provide a concise Phase 4 and Phase 5 execution summary with actual commands, test results, failures and remaining blockers.
