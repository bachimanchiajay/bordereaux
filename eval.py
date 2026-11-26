
Create a detailed enterprise-level component architecture diagram for the SNOW Research AI Agent, inspired by the component architecture reference image provided by me.

The purpose of this diagram is to help:

- Deployment engineers understand what components need to be deployed.
- Infrastructure teams understand service dependencies.
- Architects understand who calls whom.
- Developers understand the request and response flow.
- Support engineers identify where failures occur.
- Security teams understand service communication boundaries.
- Operations teams understand which databases, volumes and external services are required.

IMPORTANT:

Use the actual implementation from all three repositories.

Do not copy component names from the reference diagram unless those components actually exist in my source code.

Do not invent workers, schedulers, queues, registries, storage services, APIs or databases.

If a component does not exist, do not include it as an implemented component.

## A. Create a detailed four-column component architecture diagram

Organize the main architecture into four logical columns, following the reference image.

### COLUMN 1 — INGESTION AND DOCUMENT PROCESSING PIPELINE

Inspect snow-knowledge-service and identify every implemented component involved in document ingestion.

Investigate:

- Document upload endpoint.
- Ingestion API.
- Ingestion orchestrator.
- Background processing.
- Document extraction.
- PDF processing.
- OCR, if implemented.
- Text normalization.
- Document parsing.
- Chunking.
- Metadata extraction.
- Embedding generation.
- Vector insertion.
- Database persistence.
- Ingestion status updates.
- Error handling.

Include actual components, classes, functions and workers only when they exist.

For every connection, show:

- Calling component.
- Receiving component.
- API endpoint or function.
- Data passed.
- Processing performed.
- Output produced.

Example flow to investigate, not assume:

Document Upload
→ Ingestion API
→ Document Processing
→ Text Extraction
→ Chunking
→ Embedding Generation
→ Vector Database.

If ingestion is asynchronous, explicitly show the actual queue, background task or worker implementation.

If ingestion is synchronous, represent it as synchronous processing.

### COLUMN 2 — DATA SOURCES AND SOURCE OF TRUTH

Identify the actual data sources used by the application.

Inspect:

- Uploaded documents.
- ServiceNow integration.
- External data sources.
- Existing relational databases.
- Document metadata.
- Configuration and client information, if implemented.
- Document storage.
- Any source-of-truth systems.

Identify which component owns each data source.

Explain:

- Who writes data.
- Who reads data.
- Which service owns the connection.
- Whether data is stored locally, in a database or in an external system.
- Whether synchronization is implemented.
- Whether source data is copied, transformed or referenced.

Clearly distinguish actual integrated systems from proposed future integrations.

Do not assume a client portal, document registry or synchronization scheduler exists.

### COLUMN 3 — SHARED STORES AND SERVICES

Identify all shared data stores, infrastructure dependencies and AI services used by the three applications.

Investigate:

- PostgreSQL.
- pgvector extension.
- Vector tables.
- Document/chunk tables.
- Metadata tables.
- Conversation/session storage.
- Uploaded document storage.
- Persistent Docker volumes.
- Embedding model.
- Embedding generation service.
- LLM gateway.
- AI model endpoints.
- SQL Server, if used.
- Other external services actually configured.

For each storage component document:

1. Name.
2. Purpose.
3. Owning service.
4. Container or external resource.
5. Database/schema/table, where applicable.
6. Data stored.
7. Read operations.
8. Write operations.
9. Persistence mechanism.
10. Backup or retention mechanism, if implemented.

Show which services access each shared resource.

For example, if supported by actual code:

Knowledge Service → PostgreSQL/pgvector → Vector Records.

Orchestrator → Knowledge Service → Retrieval Results.

Make it clear whether access occurs through HTTP APIs or direct database connections.

Do not portray a resource as shared when the applications actually access it independently.

### COLUMN 4 — QUERY AND CHAT PIPELINE

Inspect the complete request and response flow across:

- Microsoft 365 Agents Playground.
- Microsoft Teams.
- Bot Framework integration.
- snow-research-teams-bot.
- chat-orchestrator.
- snow-knowledge-service.
- Embedding generation.
- Dense retrieval.
- BM25 retrieval, if implemented.
- Hybrid retrieval.
- Reciprocal Rank Fusion, if implemented.
- Cross-encoder reranking.
- Context construction.
- LLM gateway.
- Final response generation.
- Bot response delivery.

Show exactly which component calls each API.

Include the actual route names and HTTP methods.

Example structure to verify against the code:

Microsoft 365 Agents Playground
→ Teams Bot Messaging Endpoint
→ Bot Message Handler
→ Orchestrator Client
→ Chat API
→ Knowledge Search API
→ Retrieval Pipeline
→ Vector Database
→ Context Construction
→ Tarvos LLM Gateway
→ Response Generation
→ Orchestrator Response
→ Teams Bot
→ User.

Do not assume that the example request ordering or endpoints are correct. Trace the actual implementation.

## B. Show exactly WHO CALLS WHOM

This is the most important requirement.

Every connection between two components must contain a meaningful label.

Examples:

- POST /api/messages
- POST /api/sessions/create
- POST /snow/chat
- POST /v1/search
- HTTP GET health check
- Upload document
- Extract text
- Generate embedding
- Insert vector
- Search vectors
- Retrieve document metadata
- Send prompt + context
- Receive LLM response
- Return bot activity

Use the actual API paths and methods from the source code.

Where communication occurs through internal Python function calls rather than HTTP, label it accordingly.

Where a database operation occurs, identify the actual table or operation if available.

Avoid generic arrow labels such as "connects to" when the exact operation can be determined.

## C. Add component ownership and deployment mapping

For every component in the architecture, identify:

| Field | Description |
|---|---|
| Component | Actual component name |
| Repository | Which of the three services owns it |
| Module/File | Actual source file |
| Class/Function | Main implementation |
| Docker service | Compose service name |
| Container | Actual container name, if confirmed |
| Image | Docker image |
| Port | Actual exposed/listening port |
| Protocol | HTTP, HTTPS, database or internal call |
| Dependencies | Required upstream components |
| Storage | Data resources used |
| Configuration | Required environment variables |
| Health check | Actual health endpoint or check |

Do not guess container names or ports.

This mapping must help the deployment team understand exactly what needs to be deployed.

## D. Create an inter-service communication matrix

Produce a detailed table:

| Caller | Receiver | Protocol | API/Function | Purpose | Authentication | Expected result |

Include every relevant communication relationship.

Distinguish:

- External to internal communication.
- Bot to orchestrator.
- Orchestrator to knowledge service.
- Knowledge service to PostgreSQL.
- Knowledge service to embedding model.
- Orchestrator to LLM gateway.
- Bot/orchestrator to feedback storage, if implemented.
- Ingestion service to external sources, if implemented.

Use actual authentication and authorization details from the implementation.

Do not expose credentials.

## E. Create a detailed runtime sequence diagram

Create a Mermaid sequence diagram for a real user question.

Show the execution sequence:

1. User submits a question in Agents Playground.
2. Bot receives the activity.
3. Bot processes the incoming message.
4. Bot invokes the orchestrator.
5. Orchestrator processes the chat request.
6. Orchestrator invokes knowledge retrieval.
7. Knowledge service processes the search.
8. Relevant vectors and document content are retrieved.
9. Retrieval results return to the orchestrator.
10. Orchestrator constructs the LLM prompt.
11. Orchestrator calls Tarvos LLM Gateway.
12. LLM produces an answer.
13. Orchestrator returns the answer.
14. Bot prepares the response.
15. Bot sends the response through Bot Framework.
16. User receives the answer.

Include actual APIs, function calls and response transitions.

Only include steps supported by source code.

## F. Create a separate document ingestion sequence diagram

Show:

1. Document upload.
2. API validation.
3. Extraction.
4. Text processing.
5. Chunking.
6. Metadata generation.
7. Embedding generation.
8. Database insertion.
9. Persistence.
10. Ingestion status response.

Identify which service executes each stage.

Identify whether operations are synchronous or asynchronous.

Include actual API endpoints and relevant database writes.

## G. Create an actual Docker deployment topology diagram

Create a separate infrastructure-level diagram showing:

- Docker Compose project.
- Shared Docker network.
- PostgreSQL/pgvector container.
- Database initialization container.
- Knowledge service container.
- Orchestrator container.
- Teams bot container.
- Persistent Docker volumes.
- Host-to-container port mappings.
- Internal container-to-container connections.
- External Tarvos LLM Gateway connectivity.
- Other actual external dependencies.

Show:

- Service name.
- Image name.
- Container port.
- Host port.
- Health check.
- Startup dependencies.
- Persistent storage.

Use actual Docker Compose configuration.

Do not include any tunnel infrastructure.

Clearly distinguish internal Docker networking from externally accessible endpoints.

## H. Add deployment dependency ordering

Create an explicit dependency diagram or table showing the safe deployment sequence.

Investigate:

- Which infrastructure containers must start first.
- Which database initialization must finish first.
- When the knowledge service can become healthy.
- When the orchestrator can start.
- When the bot can start.
- Which external dependencies are required at runtime.

Use actual Compose health checks and dependency conditions.

Do not assume `depends_on` alone guarantees application readiness.

Explain the actual readiness mechanism.

## I. Add failure propagation and troubleshooting paths

Identify the impact when each component is unavailable.

Examples to investigate:

- PostgreSQL unavailable.
- Knowledge service unavailable.
- Embedding service unavailable.
- Vector retrieval fails.
- Orchestrator unavailable.
- Tarvos gateway timeout.
- Bot authentication failure.
- Bot-to-orchestrator communication failure.

Create a troubleshooting matrix:

| Failed component | Observable symptom | Affected downstream components | Logs to inspect | Diagnostic command |

Use actual container names and logs.

## J. Diagram formatting

All architecture diagrams must follow the reference style:

- Horizontal, left-to-right logical grouping.
- Clearly defined component boxes.
- Consistent colors per logical layer.
- Labelled directional arrows.
- Actual component names.
- Actual API and data operation labels.
- Clear separation of services and shared resources.
- Deployment boundaries.
- Readable text.
- Minimal crossing arrows.

Use these logical color groups:

- Blue: ingestion and processing.
- Orange: source systems.
- Green: databases, shared resources and infrastructure.
- Purple: query, chat and LLM processing.
- Grey: Docker runtime and deployment boundaries.

The diagrams must be readable in both Markdown and PDF.

If a single diagram becomes overcrowded, create multiple related diagrams instead of reducing text size.

## K. Required architecture deliverables

Add these diagrams to the final technical guide:

1. High-level system architecture.
2. Detailed four-column component interaction architecture.
3. Document ingestion pipeline.
4. Data storage and ownership architecture.
5. Query and chat component architecture.
6. User question sequence diagram.
7. Document ingestion sequence diagram.
8. Docker deployment topology.
9. Inter-service dependency architecture.
10. Failure propagation architecture.

Retain the Mermaid source code for every diagram in the Markdown document.

## L. Final architecture quality validation

Before generating the final PDF:

- Validate every component against the source code.
- Validate all API paths.
- Validate caller and receiver direction.
- Validate communication protocols.
- Validate database ownership.
- Validate Docker service names.
- Validate port mappings.
- Validate volumes.
- Validate environment dependencies.
- Validate component dependencies.
- Ensure there are no invented components.
- Ensure there are no invented API calls.
- Ensure actual versus recommended components are clearly distinguished.

The primary outcome must be an architecture document that an infrastructure/deployment engineer can use to understand exactly what to deploy, where each component runs, what it communicates with, what it depends on, and how the complete request and data flows through the application.
