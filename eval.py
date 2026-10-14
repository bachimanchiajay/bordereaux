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
