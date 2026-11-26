This project is a Python backend for a Microsoft Teams app (Teams Python package). 
It uses a database, the Blue Prism API, and an LLM client. The code is messy: DB 
connections, classes, loose functions, SQL queries, and external API calls are mixed 
together. My lead asked me to restructure the ENTIRE codebase to proper coding standards.

CORE RULE (highest priority)
Logic and functionality must NOT change. Same inputs must give the same outputs, same 
DB results, same API calls and payloads, same prompts sent to the LLM, same responses 
to Teams. This is a structural refactor only. If you think something is a bug, do not 
fix it silently. List it for me and keep the current behavior.

STEP 1 - ANALYZE (do not edit anything yet)
Read every file in the project and give me:
- A map of the current structure: every file, class, loose function, DB connection, 
  SQL query, Blue Prism API call, and LLM call, and where each lives.
- The main flows end to end (for example: Teams message -> handler -> LLM -> DB -> 
  Blue Prism -> reply), so we agree on the behavior that must stay identical.
- A list of problems found (duplicate connections, hardcoded credentials, SQL outside 
  the DB layer, print statements, unnecessary loops, duplicated code, bare excepts).
- The proposed new folder structure.
Wait for my approval before changing code.

STEP 2 - REFACTOR (after I approve)

Architecture
1. Layered structure: handlers/bot -> services (business logic) -> repository/clients -> 
   db / external APIs. Handlers never touch SQL or call external APIs directly.
2. ALL SQL queries live in one module (db/queries.py). No SQL anywhere else.
3. DB connection in db/connection.py as a Singleton (one shared pool, created once, 
   reused everywhere) with a context manager for acquire/release and safe cleanup.
4. Blue Prism API in clients/blueprism_client.py and the LLM in clients/llm_client.py. 
   Both are Singletons, reuse a single HTTP session/client, and have timeouts, 
   retries where the current code already retries, and error handling. 
   Keep the existing prompts, model names, parameters, and request/response 
   formats exactly as they are. Move prompt templates into a prompts module without 
   altering their text.
5. Move all loose functions into the right class or a utils module, grouped by 
   responsibility. Refactor the entire project, not just these modules.

Code quality
6. PEP 8, type hints, docstrings, meaningful names.
7. Remove unnecessary loops and redundant code (comprehensions, built-ins, dict/set 
   lookups, or SQL WHERE/JOIN/IN instead of fetch-and-loop), only where the result 
   stays identical.
8. Parameterized queries only.
9. Config in config.py from environment variables / .env. No hardcoded credentials, 
   URLs, API keys, or model names scattered around. Add .env.example.

Logging and errors
10. Central app logger (core/logger.py): consistent format (timestamp, level, module, 
    message), console + rotating file handler, configurable level. Each module uses 
    logging.getLogger(__name__). Replace every print(). Never log secrets, tokens, or 
    full user messages containing sensitive data.
11. Custom exception classes, specific exceptions instead of bare except, log with 
    context. Preserve the current error behavior toward the user (same fallback 
    messages and same replies in Teams).

Safety process
12. Work in small steps: config/logger -> DB + queries -> Blue Prism client -> 
    LLM client -> services -> handlers -> remaining modules. After each step, run 
    existing tests or at least confirm the app imports and starts, and tell me.
13. Before refactoring, if there are no tests, write a few basic characterization tests 
    for the key flows (mock DB, Blue Prism, and LLM) that capture current behavior, 
    then use them to prove nothing changed.
14. No new heavy dependencies without asking. Update requirements.txt if needed.
15. Keep the public entry points and Teams message handling behavior the same so 
    deployment does not change.

FINAL OUTPUT
- Final folder structure.
- A table of what moved where (old file/function -> new file/function).
- Bugs or suspicious logic you noticed but intentionally left unchanged.
- Anything uncertain, and a manual test checklist for me.




Act as a Senior Python Architect and Software Engineer. Inspect my entire existing Python Microsoft Teams backend project and directly fix all code quality issues before Git push.

- Follow PEP 8, SOLID, DRY, KISS, and clean architecture principles.
- Refactor poorly structured code into appropriate modules with clear separation of Teams handlers, business logic, database connections, and SQL queries.
- Centralize SQL queries in a dedicated module and implement safe, reusable database connection management with proper pooling, cleanup, and transaction handling.
- Simplify deeply nested loops, eliminate duplicate logic, unnecessary database calls, unused imports, and overly complex functions.
- Add appropriate type hints, meaningful exception handling, structured logging, and secure configuration.
- Fix SQL injection risks, hardcoded credentials, resource leaks, and concurrency issues.
- Preserve all existing business logic, API contracts, Teams functionality, and database behavior.
- Add or update tests and run available tests, linting, formatting, and type checks. Fix failures caused by your changes.
- Make changes directly in the existing files. Do not rewrite everything unnecessarily, introduce needless design patterns, or generate reports instead of fixing code.
- Do not commit, push, or deploy. Do not overwrite unrelated changes. Review the final Git diff and ensure no secrets are included.

**Start fixing the code immediately. Work incrementally, validate every change, and continue until all safely fixable issues are resolved. Clearly identify any remaining failures or blockers.**

# Task: Build and Perform End-to-End Testing of Three SNOW Services Using Docker

Act as a Senior DevOps Engineer, Python Backend Architect, Docker/Kubernetes Engineer and Microsoft Teams Bot Integration Specialist.

I have three Python services that must be built and tested together using Docker images and Docker Compose.

## Services

1. snow-knowledge-service
2. chat-orchestrator
3. snow-research-teams-bot

My objective is to execute the complete application flow and test it through Microsoft 365 Agents Playground.

**STRICT REQUIREMENT: All three application services must run inside Docker containers. Do not run Python services directly on my Mac.**

Do not assume the architecture, API endpoints, environment variables or dependencies. Inspect the actual source code.

# PHASE 1 — Repository and Configuration Audit (READ ONLY)

First inspect all three service directories.

For each service, inspect:

- Dockerfile
- docker-compose.yml and any override Compose files
- .env.example and other environment templates
- requirements.txt and development dependencies
- app.py / main.py / entry points
- config.py and runtime_env.py
- API routers and endpoint definitions
- Health check endpoints
- Database connections and initialization scripts
- External API integrations
- Inter-service HTTP communication
- Authentication and authorization
- Logging configuration
- Existing unit, integration and smoke tests
- README and deployment documentation

Inspect the existing Docker configuration before proposing changes.

## 1. Build the dependency map

Identify:

- Which service calls which service.
- Actual API endpoints and HTTP methods.
- Required request and response payloads.
- Required database services.
- Required vector database services.
- LLM and embedding model dependencies.
- Any external AI Gateway dependencies.
- File storage and persistent volume requirements.
- Authentication between services.

Create an actual service dependency diagram based on the code.

## 2. Environment variable audit

Create a consolidated environment variable inventory.

For each variable provide:

- Variable name
- Service that requires it
- Whether it is mandatory or optional
- Whether it is a secret
- What it is used for
- Whether a default value exists
- Whether it is already defined in an example file
- Whether it must be supplied by me

Inspect actual code references. Do not invent variables.

Pay particular attention to:

- MICROSOFT_APP_ID
- MICROSOFT_APP_PASSWORD
- MICROSOFT_APP_TENANT_ID
- Bot messaging endpoint
- LLM gateway URL
- LLM authentication credentials
- Chat model identifier
- Embedding model identifier
- Database credentials and connection strings
- Vector database configuration
- Service-to-service URLs
- Storage paths

These are candidate variables only. Confirm whether each is actually used.

## 3. Docker readiness audit

Check whether the current Dockerfiles and Compose files can support the complete application.

Identify:

- Missing Dockerfiles
- Incorrect build contexts
- Missing dependencies
- Incorrect container ports
- Port conflicts
- Incorrect localhost references
- Missing health checks
- Missing persistent volumes
- Incorrect startup ordering
- Database initialization requirements
- Environment file references
- Container networking issues

Inside Docker, service-to-service communication must use Docker Compose service names rather than localhost.

## 4. Produce these deliverables

Create an audit report containing:

1. Service architecture diagram.
2. Service dependency table.
3. API endpoint inventory.
4. Complete environment variable inventory.
5. List of mandatory secrets I must provide.
6. List of environment files required for each service.
7. Database and external dependency inventory.
8. Docker readiness assessment.
9. Proposed Docker Compose architecture.
10. Exact commands required for the next phase.

Create a consolidated template named `.env.e2e.example` if useful, but do not put real secrets in it.

## STOP CONDITION — DO NOT PROCEED YET

After completing Phase 1:

- Do not build Docker images.
- Do not start containers.
- Do not modify application code.
- Do not modify existing environment files.
- Do not overwrite Docker Compose files.
- Do not delete volumes or data.
- Do not expose credentials in logs or reports.

Show me the exact list of environment files and secrets required.

STOP and wait for me to provide the required configuration.

# PHASE 2 — Docker Integration Setup

Only proceed after I confirm and provide the required environment configuration.

1. Review the supplied environment files.
2. Validate mandatory variables without printing their secret values.
3. Prepare a unified Docker Compose configuration if required.
4. Ensure all three services have separate containers.
5. Configure a shared Docker network.
6. Configure correct internal service URLs.
7. Configure health checks and dependency readiness.
8. Configure persistent volumes for databases, vector storage and uploaded documents.
9. Ensure containers restart appropriately.
10. Ensure configuration and secrets are injected securely.

Use existing Dockerfiles where possible. Avoid unnecessary application code changes.

Do not use mock implementations to conceal broken service integrations.

# PHASE 3 — Build Docker Images

Build all three images using Docker Compose.

Validate:

- Image build succeeds.
- Python dependencies install correctly.
- Application entry points work.
- Container ports are correct.
- No missing environment variables.
- No dependency conflicts.

Use actual service names and existing configuration.

Do not execute Python application services outside Docker.

# PHASE 4 — Start and Validate Services

Start the complete application stack.

Validate in dependency order:

1. Database and infrastructure dependencies.
2. snow-knowledge-service.
3. chat-orchestrator.
4. snow-research-teams-bot.

Check:

- Container health.
- Startup logs.
- Health endpoints.
- Database connectivity.
- Vector database connectivity.
- Inter-service HTTP connectivity.
- LLM gateway connectivity.
- Embedding generation.
- Document ingestion and retrieval.

Do not consider a running container sufficient evidence of a healthy service.

# PHASE 5 — End-to-End Integration Tests

Run tests inside Docker containers.

Test the following scenarios:

1. All three services start successfully.
2. Bot endpoint responds correctly.
3. Bot successfully calls chat-orchestrator.
4. Chat orchestrator successfully communicates with knowledge service.
5. Knowledge service retrieves relevant documents.
6. LLM gateway generates a response.
7. Response returns through orchestrator.
8. Bot sends the final response back through the Bot Framework integration.

Use existing test suites wherever possible.

Add integration or smoke tests only where necessary.

Validate both successful and failure scenarios:

- Invalid credentials.
- Missing configuration.
- Unavailable downstream service.
- Invalid request payload.
- LLM gateway timeout.
- Empty retrieval results.

Report actual test results. Never mark an unexecuted test as passed.

# PHASE 6 — Microsoft 365 Agents Playground Testing

Prepare the Teams bot for testing through Microsoft 365 Agents Playground.

Inspect:

- Bot Framework adapter configuration.
- Messaging endpoint and route.
- Bot authentication configuration.
- Microsoft App ID and tenant configuration.
- Existing Teams app manifest.
- Bot ID in the manifest.
- Supported scopes and app configuration.
- Required local development or externally accessible endpoint.

Determine whether Agents Playground can connect to the Docker-hosted bot endpoint directly in the current setup.

If external HTTPS access is required, explain the approved secure approach. Do not introduce ngrok or any unapproved public tunnel.

Do not claim Playground connectivity until an actual message exchange succeeds.

Provide the exact steps for connecting and testing in Microsoft 365 Agents Playground.

# PHASE 7 — Final Validation Report

Produce a final report containing:

- Docker images built.
- Container names and statuses.
- Ports and endpoints.
- Service dependency diagram.
- Database status.
- Health-check results.
- Integration test results.
- End-to-end test results.
- Playground connectivity status.
- Failed tests and root causes.
- Remaining configuration requirements.
- Commands to restart, stop and inspect the stack.

## Important execution rules

- Docker images and containers only for all three application services.
- Do not run application services locally using uvicorn or python.
- Do not invent API endpoints or configuration.
- Do not hardcode secrets.
- Do not expose secrets in logs.
- Do not overwrite existing configurations without reviewing them.
- Preserve existing application functionality.
- Avoid unnecessary code refactoring.
- Do not delete existing databases, vector indexes or uploaded documents.
- Use Docker Compose for repeatable builds and execution.


  # TASK: Complete Phase 4 and Phase 5 — SNOW Docker E2E Testing

Act as a Senior Python Backend Engineer, Docker Engineer, QA Automation Engineer and Microsoft Bot Framework Integration Specialist.

I have already completed:
- Phase 1: Repository audit.
- Phase 2: Docker integration setup.
- Phase 3: Docker image builds.

Phase 4 and Phase 5 are partially completed.

Your task is to resolve ONLY the outstanding issues identified in the previous execution reports and complete the remaining tests.

IMPORTANT:
- All three application services must run inside Docker.
- Do not run Python application services directly on the host.
- Preserve the existing Docker configuration.
- Do not delete or reset existing databases or vector data.
- Do not modify working application logic unnecessarily.
- Do not invent credentials or configuration.
- Do not print secret values.
- Use actual source code and existing configuration.

# PHASE 4 — COMPLETE SERVICE STARTUP AND VALIDATION

## Issue 1: Teams bot has not started

Current situation:

- Knowledge service is healthy.
- Chat orchestrator is healthy.
- PostgreSQL and pgvector are healthy.
- Teams bot image has been built.
- Teams bot container has not been successfully started.

The following requirements are missing:

1. ORCHESTRATOR_SCOPE
2. bot-ca.pem

Actions:

1. Inspect the bot configuration and identify exactly where ORCHESTRATOR_SCOPE is consumed.
2. Determine the expected value and format from the actual authentication implementation.
3. Identify how the bot uses bot-ca.pem.
4. Determine the expected certificate format and mount path.
5. Verify whether the certificate is required for the current connection configuration.
6. Identify the required Entra application registration settings and permissions.
7. Check whether the configured credentials match the intended application registration.

Do not invent ORCHESTRATOR_SCOPE.

Do not generate a dummy CA certificate to bypass TLS validation.

Do not disable certificate validation.

Wait for me to supply the missing legitimate configuration where required.

## Issue 2: Validate the complete Docker stack

Once the required bot configuration is supplied:

Start and validate in dependency order:

1. PostgreSQL.
2. Database initialization.
3. snow-knowledge-service.
4. chat-orchestrator.
5. snow-research-teams-bot.

Verify:

- Container status.
- Health checks.
- Application startup logs.
- Internal Docker networking.
- Service-to-service connectivity.
- Bot health endpoint.
- Bot authentication configuration.

Use the existing Compose configuration and actual service names.

# PHASE 5 — COMPLETE OUTSTANDING INTEGRATION TESTS

The previous report shows these results:

| Scenario | Existing result |
|---|---|
| 1. All three services start | Partial: bot not started |
| 2. Bot endpoint responds | Not run |
| 3. Bot calls orchestrator | Partially run |
| 4–7. Orchestrator → Knowledge → Retrieval → LLM | Passed |
| 8. Bot returns answer via Bot Framework | Not run |

Do not unnecessarily repeat already-passed tests unless a dependency change requires regression testing.

## Issue 3: Complete Scenario 2 — Bot endpoint

After the bot starts:

- Test the actual bot health endpoint.
- Verify the expected HTTP status.
- Check the bot container logs.
- Verify the endpoint is listening on the expected container port.
- Distinguish health endpoint success from authenticated Bot Framework messaging success.

## Issue 4: Complete Scenario 3 — Bot calls orchestrator

The previous report says the bot's own client code, executed inside the bot image, received a real answer and five source tickets from the live orchestrator.

However, Entra authentication and the actual bot messaging flow were not exercised.

Now:

1. Verify the actual bot-to-orchestrator authentication.
2. Verify ORCHESTRATOR_SCOPE.
3. Execute the existing bot client integration test.
4. Verify the real orchestrator response.
5. Verify source ticket handling.
6. Capture actual request and response status without exposing credentials.

## Issue 5: Complete Scenario 8 — Bot Framework response

Once bot authentication and messaging prerequisites are available:

1. Submit an actual Bot Framework activity.
2. Verify activity handling.
3. Verify the bot invokes the orchestrator.
4. Verify the orchestrator response reaches the bot.
5. Verify the final response is sent through the Bot Framework adapter.

Do not mark this scenario as passed using a mocked response or direct orchestrator API call.

# ISSUE 6 — KNOWLEDGE SERVICE TEST SUITE

Previous test results:

- 162 passed.
- 5 skipped because Helm is not installed.
- 1 failed due to the .gitignore environmental check.
- 97 skipped because no database was configured for those tests.

The previous execution proposed using a separate disposable PostgreSQL/pgvector database rather than the real E2E database.

Implement this carefully.

### Database-dependent tests

1. Inspect the existing test fixtures and database configuration.
2. Create a separate disposable pgvector test database/container.
3. Do not connect these tests to the existing production-like E2E database.
4. Do not reset the existing database.
5. Run the 97 database-dependent tests against the isolated test database.
6. Report passed, failed and skipped counts.

### .gitignore test failure

The previous report identified the missing Git executable as an environmental cause.

1. Inspect the failing test.
2. Confirm whether it requires Git.
3. If required, add Git to the relevant test image using the appropriate base-image package manager.
4. Rebuild only the affected image.
5. Rerun the test.
6. Confirm the exact failure is resolved.

Do not modify the test merely to bypass the check.

### Helm tests

The five Helm-related tests were skipped because Helm was not installed.

Inspect their requirements and report whether Helm installation is necessary to execute them.

Do not mark skipped tests as passed.

# ISSUE 7 — INVESTIGATE THE EXISTING NEGATIVE TEST RESULTS

The previous execution reported the following actual behaviours.

| Scenario | Observed behaviour |
|---|---|
| Search service unreachable | HTTP 503 RAG Service Unavailable |
| Invalid payloads | HTTP 422 |
| Empty retrieval | HTTP 200 with no relevant tickets |
| Missing configuration | Service reports missing configuration |
| Invalid LLM key / unreachable LLM / missing key and use-case ID | HTTP 200 with fallback response |
| Unknown session_id | HTTP 200 instead of an error |
| LLM gateway timeout | Not run; unreachable gateway was tested instead |
| Bad Entra credentials | Not run because bot was not started |

Investigate these exact behaviours.

For each:

1. Locate the relevant implementation.
2. Identify the exception-handling logic.
3. Determine whether the response is intentional.
4. Identify whether the behaviour creates an operational or debugging issue.
5. Recommend a correction only when justified by the implementation.

Pay special attention to:

- Fallback responses being returned with HTTP 200.
- Whether fallback responses can be mistaken for genuine LLM-generated answers.
- Unknown session ID handling.
- Actual timeout handling versus an unreachable gateway.

Do not change API contracts or production behaviour without explaining the impact and obtaining approval.

# FINAL VALIDATION REPORT

Produce a consolidated Phase 4 and Phase 5 report.

Include:

1. Issues identified.
2. Root cause of each issue.
3. Files modified.
4. Docker images rebuilt.
5. Containers started.
6. Exact commands executed.
7. Tests passed.
8. Tests failed.
9. Tests skipped.
10. Remaining dependencies.
11. Actual end-to-end status.

Use this execution matrix:

| Scenario | Previous status | Current status | Evidence |
|---|---|---|---|

Only mark a test passed when it actually executes successfully.

# DOCUMENTATION UPDATE

Update the existing files:

SNOW_Research_AI_Agent_Technical_Guide.md

SNOW_Research_AI_Agent_Technical_Guide.pdf

Record the actual Phase 4 and Phase 5 implementation, testing commands, results and outstanding issues.

Do not generate a new generic document.

Preserve the previous phase history.

# EXECUTION RULE

First inspect the current configuration and identify the exact missing bot prerequisites.

If credentials or certificate files are missing, stop and request them from me.

Continue with independent knowledge service and orchestrator tests wherever possible.

Do not block unrelated tests unnecessarily.

Do not use mocks to claim that the real Bot Framework integration passed.

Proceed step by step and provide evidence for every completed item.
- Report actual results, not assumptions.

Work phase by phase and respect the STOP condition after Phase 1.
