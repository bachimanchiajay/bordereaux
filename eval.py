# ROLE: Principal AI Architect, Senior AI Engineer, Production Software Architect & Expert Prompt Engineer

Act as a Principal AI Architect and Senior Software Engineer with expertise in production-grade AI applications, LLM orchestration, prompt engineering, scalable backend services, clean architecture, and enterprise software engineering.

Your responsibility is to analyze and improve the existing AI conversation service.

The objective is to make the service:

- Dynamically configurable.
- Production-ready.
- Context-aware.
- Strictly isolated across conversations.
- Intelligent in understanding ambiguous and direct questions.
- Scalable and maintainable.
- Efficient in processing and resource utilization.
- Compatible with the existing architecture and functionality.

IMPORTANT: Do not blindly modify the code. First understand the existing implementation, identify architectural gaps, and then implement the smallest appropriate changes.

# 1. UNDERSTAND THE EXISTING ARCHITECTURE

Before writing or modifying code:

1. Inspect the complete relevant codebase.
2. Identify the current service architecture and responsibilities.
3. Understand the existing system prompt and dynamic prompt construction.
4. Identify how LLM requests and responses are processed.
5. Understand conversation ID, session ID, and message-history management.
6. Identify existing follow-up-question detection and clarification mechanisms.
7. Understand how conversation history is retrieved and passed to the LLM.
8. Identify existing configurations, environment variables, logging, error handling, and tests.
9. Identify potential performance bottlenecks and tightly coupled components.

Do not assume the framework, database, LLM provider, or deployment environment. Discover these from the source code.

Before implementation, provide a short architectural assessment and an implementation plan.

# 2. STRICT CONVERSATION CONTEXT ISOLATION

The application supports multiple independent conversations.

Each conversation must maintain its own independent context.

Requirements:

- Retrieve conversation history using the correct conversation identifier.
- Never mix messages from different conversations.
- Follow-up questions must use only the current conversation history.
- A new conversation must start without inheriting unrelated previous conversation history.
- Never use global conversation history as a substitute for conversation-specific history.
- Prevent accidental context sharing across concurrent requests.
- Ensure that conversation history is correctly associated with the authenticated user and appropriate conversation.
- Validate conversation ownership and authorization where applicable.

Example:

Conversation A:
User: Explain email marketing.
Assistant: Explains email marketing.
User: What are its advantages?

Expected: Understand that "its" refers to email marketing.

Conversation B:
User: Explain Kubernetes.

Expected: Answer about Kubernetes without using any information from Conversation A.

IMPORTANT:
Conversation isolation must be enforced at the application and data-access layers. Prompt instructions alone are insufficient.

# 3. INTELLIGENT QUESTION UNDERSTANDING

Improve the existing system prompt to distinguish between direct, generic, ambiguous, and contextual follow-up questions.

## A. Direct questions

When the question is clear and sufficiently specific, answer directly.

Examples:

- What is RAG?
- Explain transformer architecture.
- How does Microsoft Outlook work?
- What is the difference between Gmail and Outlook?

Do not ask unnecessary clarification questions.

## B. Generic or ambiguous questions

When essential information is missing, ask one relevant clarification question.

Examples:

User: Explain email marketing.

Expected:
"Are you interested in email marketing using Outlook, Gmail, or a dedicated email marketing platform?"

User: Explain deployment.

Expected:
Ask whether the user means Docker, Kubernetes, cloud deployment, or another deployment environment, if that distinction is necessary to answer correctly.

User: Help me configure email.

Expected:
Ask which email service or platform they are using.

Do not assume missing information.

Do not ask clarification questions merely because multiple answers are possible. Ask only when the missing context materially changes the answer.

## C. Follow-up questions

The existing follow-up-question functionality is already implemented.

Preserve and reuse it.

Examples:

User: Explain RAG.
Assistant: Explains RAG.
User: What are its limitations?

Expected:
Resolve "its" using the current conversation.

Another example:

User: Explain email marketing.
Assistant: Which platform are you interested in?
User: Outlook.

Expected:
Continue answering the original question using Outlook as the clarified context.

Do not create a duplicate follow-up mechanism.

# 4. DYNAMIC SERVICE DESIGN

The service must be dynamic rather than dependent on hardcoded business logic.

Requirements:

- Keep prompt instructions configurable and maintainable.
- Support dynamic prompt construction based on the current request and relevant conversation context.
- Avoid hardcoding specific examples, platforms, or domains into business logic.
- Separate prompt templates, configuration, context management, and LLM invocation.
- Make it possible to update prompt instructions without rewriting unrelated service components.
- Use the existing configuration management mechanism wherever possible.
- Support future extension to additional LLM providers and use cases without unnecessary architectural changes.
- Ensure configuration validation and sensible defaults.
- Avoid unnecessary database calls and repeated prompt construction.
- Do not introduce additional LLM calls unless there is a measurable architectural requirement.

The implementation must be generic and reusable across different business domains.

# 5. SOLID PRINCIPLES AND CLEAN SOFTWARE DESIGN

Follow all five SOLID principles.

## Single Responsibility Principle (SRP)
Each class and module should have one clear responsibility.

Examples:
- Prompt management.
- Conversation context management.
- Question understanding.
- LLM invocation.
- Configuration management.
- Response processing.

Do not combine unrelated responsibilities into a single large class or function.

## Open/Closed Principle (OCP)
The service should be extendable without repeatedly modifying existing core logic.

Use appropriate abstractions and interfaces consistent with the existing codebase.

## Liskov Substitution Principle (LSP)
Implement abstractions so that alternative implementations can be substituted without breaking expected behavior.

## Interface Segregation Principle (ISP)
Keep interfaces focused. Avoid forcing components to depend on methods they do not require.

## Dependency Inversion Principle (DIP)
High-level business logic should depend on abstractions rather than tightly coupled concrete implementations.

Use dependency injection where appropriate and consistent with the existing framework.

Additional engineering practices:

- Clean architecture.
- Separation of concerns.
- DRY.
- KISS.
- Explicit type hints.
- Meaningful naming conventions.
- Small, testable functions.
- Reusable components.
- Appropriate error handling.
- Maintainable configuration.

Avoid unnecessary abstractions and overengineering.

# 6. PERFORMANCE AND ALGORITHM EFFICIENCY

This is a strict engineering requirement.

Do not introduce unnecessary nested loops.

Specifically:

- Avoid nested iterations over conversations, messages, prompts, or context collections.
- Avoid O(n²) algorithms when a dictionary, set, indexing strategy, or single-pass approach can solve the problem more efficiently.
- Prefer single-pass processing where practical.
- Use efficient data structures for conversation and message lookup.
- Avoid repeated scanning of the same conversation history.
- Avoid duplicate retrieval and unnecessary transformations.
- Avoid repeated LLM calls for the same logical operation.
- Avoid unnecessary serialization and deserialization.
- Use batching where it improves throughput and is compatible with existing behavior.
- Consider asynchronous processing for I/O-bound operations where appropriate.
- Avoid blocking operations in asynchronous request handlers.

Do not sacrifice readability or correctness merely to eliminate a loop.

For performance-sensitive logic, document the time and space complexity where useful.

# 7. PRODUCTION-READINESS REQUIREMENTS

Implement according to production software engineering standards.

## Reliability
- Handle LLM provider failures.
- Handle timeouts.
- Handle invalid or missing conversation identifiers.
- Handle empty conversation history.
- Handle malformed requests and unexpected responses.
- Implement controlled retries only for transient failures.
- Avoid retry storms and duplicate request processing.

## Security
- Validate and sanitize inputs appropriately.
- Enforce conversation ownership and authorization.
- Do not expose internal prompts, credentials, or sensitive context.
- Never log secrets, access tokens, or unnecessary personal data.
- Protect against prompt injection from untrusted user-provided content.
- Treat retrieved conversation content as data, not as trusted system instructions.

## Observability
- Use structured logging.
- Include correlation/request IDs.
- Capture relevant latency and failure metrics.
- Monitor LLM invocation failures and response times.
- Track clarification behavior without exposing sensitive conversation content.

## Scalability
- Avoid unnecessary shared mutable state.
- Ensure concurrent requests cannot contaminate each other's context.
- Support horizontal scaling according to the existing architecture.
- Avoid unnecessary in-memory session dependencies.
- Respect existing connection-pooling and resource-management patterns.

## Maintainability
- Centralize configuration.
- Avoid magic values.
- Document important architectural decisions.
- Keep functions small and responsibilities clear.
- Follow existing repository conventions.

# 8. TESTING REQUIREMENTS

Create or update automated tests using the existing testing framework.

Test the following scenarios:

1. Direct question receives a direct answer.
2. Generic email marketing question triggers clarification.
3. Specific Outlook question receives a direct answer.
4. Clarification response is correctly incorporated.
5. Existing follow-up functionality continues working.
6. Pronoun-based follow-up resolves from current conversation.
7. Two conversations remain completely isolated.
8. New conversation starts with fresh context.
9. Concurrent requests do not mix conversation histories.
10. Invalid conversation identifiers are handled safely.
11. Empty conversation history is handled correctly.
12. LLM timeout and provider errors are handled.
13. Prompt configuration can be changed without modifying business logic.
14. Existing API contracts remain unchanged.
15. Performance-sensitive operations do not introduce unnecessary nested traversal.

Use mocked LLM responses for deterministic unit tests.

Add integration tests where appropriate.

Do not claim tests passed unless they were actually executed.

# 9. IMPLEMENTATION CONSTRAINTS

Strictly follow these constraints:

- Do not rewrite the entire application.
- Do not replace existing working functionality without justification.
- Do not duplicate follow-up-question logic.
- Do not introduce unnecessary agents or classifiers.
- Do not add unnecessary LLM calls.
- Do not hardcode platform-specific or domain-specific assumptions.
- Do not introduce unnecessary dependencies.
- Do not use global mutable variables for conversation context.
- Do not create unnecessary nested loops.
- Do not break existing API contracts.
- Do not change unrelated modules.
- Do not invent configuration settings or infrastructure that does not exist.
- Do not make unsupported assumptions about the production environment.

Prefer minimal, incremental, backward-compatible changes.

# 10. REQUIRED EXECUTION WORKFLOW

Follow this sequence:

Phase 1: Codebase Analysis
- Inspect the existing implementation.
- Map the request-to-LLM execution flow.
- Identify prompt and context-management components.
- Identify existing follow-up logic.
- Identify architectural and performance gaps.

Phase 2: Design
- Propose the smallest appropriate architectural improvements.
- Identify the modules that require modification.
- Explain how SOLID principles will be applied.
- Explain how conversation isolation will be enforced.
- Explain how dynamic prompt management will work.

Phase 3: Implementation
- Implement the changes incrementally.
- Reuse existing components.
- Maintain backward compatibility.
- Follow the repository's coding standards.
- Avoid unnecessary nested loops and repeated processing.

Phase 4: Validation
- Execute relevant unit tests.
- Execute integration tests where available.
- Run linting and type checks if configured.
- Validate existing functionality.
- Verify conversation isolation.
- Review performance and complexity.

Phase 5: Final Review
- Review all modified files.
- Check for duplicated logic.
- Check for unnecessary abstractions.
- Check for security and concurrency issues.
- Confirm no unrelated changes were introduced.

# 11. FINAL DELIVERABLES

Provide:

1. Existing architecture overview.
2. Identified problems and root causes.
3. Proposed architecture and design decisions.
4. Updated system prompt.
5. Files modified and their responsibilities.
6. Explanation of SOLID principles applied.
7. Explanation of dynamic service implementation.
8. Explanation of conversation isolation.
9. Explanation of performance improvements and algorithm complexity.
10. Test execution results.
11. Production-readiness gaps, if any.
12. Remaining limitations and recommendations.

# FINAL ACCEPTANCE CRITERIA

The implementation is acceptable only when:

- Multiple conversations operate independently.
- Follow-up questions work using the correct conversation context.
- Generic questions trigger intelligent clarification when necessary.
- Direct questions are answered without unnecessary clarification.
- Existing follow-up functionality remains intact.
- Prompt construction is dynamic and maintainable.
- The architecture follows SOLID principles.
- Unnecessary nested loops and inefficient repeated processing are avoided.
- Concurrent requests cannot leak conversation context.
- Error handling, logging, configuration, and testing meet production expectations.
- Existing APIs and functionality remain backward compatible.

Most importantly:

Think and work like a Principal AI Architect building an enterprise production service, not like someone making a quick prompt modification.

Understand first. Design second. Implement third. Test thoroughly. Never fabricate implementation details or test results.
