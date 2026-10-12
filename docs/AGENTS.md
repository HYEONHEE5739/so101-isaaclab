# AGENTS.md

## Purpose

Keep the project understandable, maintainable, and easy to extend while allowing AI-assisted development.

The goal is not to minimize the number of files. Keep responsibilities clear, avoid unnecessary duplication, and integrate new features into the existing architecture.

## 1. Before Changing Code

Before implementing a non-trivial feature:

1. Inspect the relevant project structure.
2. Find existing classes, functions, scripts, and configs related to the feature.
3. Understand the current execution flow and data flow.
4. Prefer extending or reusing existing code over creating parallel implementations.
5. Identify the smallest set of files that should change.

For substantial changes, explain the implementation plan before modifying files:
- Files to modify
- Files to create, if any
- Classes/functions to add or change
- How the new code connects to existing code
- Changes to execution/data flow
- Whether a new launcher, script, or config is actually necessary

Do not modify code merely to demonstrate an idea before the implementation direction is clear.

## 2. Existing Code First

- Reuse existing classes, functions, utilities, and configuration structures whenever appropriate.
- If a feature belongs to an existing class's responsibility, add it there rather than creating a parallel class.
- Do not create a new class simply because it is convenient for the immediate task.
- Do not duplicate existing logic under a new name.
- Do not create `*_new`, `*_v2`, `*_final`, `*_test2`, or similarly duplicated files unless there is a clear architectural reason.
- Preserve existing public interfaces unless changing them is necessary.

Do not assume that "new file = cleaner code." First determine whether the existing architecture has an appropriate extension point.

## 3. File and Folder Structure

Follow the existing project structure.

Typical responsibilities:
- `src/`: core application/library code
- `scripts/`: intentional executable entry points
- `configs/`: reusable configuration
- `tests/`: automated tests
- `docs/`: architecture and development documentation
- `experiments/`: isolated experimental work when appropriate

Do not create a new directory merely to hold one temporary file.

If the project already has an established structure, follow it instead of imposing a new one.

## 4. Launcher / Script Rules

Create a new launcher or script only when:
- A genuinely separate execution entry point is required, or
- The project already uses that pattern for the relevant workflow.

Do not create a launcher merely to:
- import several existing modules and call them once,
- work around understanding the existing execution flow,
- expose a temporary experiment,
- duplicate an existing execution path.

Prefer extending an existing entry point when the new functionality belongs to the same workflow.

Temporary experiments should be isolated from the main architecture and should not become permanent production entry points without justification.

## 5. Configuration Rules

Reuse existing configuration structures when possible.

Before creating a config:
1. Check whether the existing config can represent the required values.
2. Extend the existing config if that keeps the structure coherent.
3. Create a separate config only when it represents a genuinely different workflow, environment, or configuration boundary.

Avoid duplicated configuration values across multiple files.

## 6. Class and Module Responsibilities

Each class/module should have a clear responsibility.

When adding functionality, ask:
- Does this logically belong to an existing class?
- Does adding it make that class excessively broad?
- Does the functionality represent a separate reusable responsibility?
- Is a new abstraction actually justified?

Avoid both unnecessary fragmentation and large "God classes" containing unrelated responsibilities.

## 7. Implementation

- Make the smallest coherent change that satisfies the requirement.
- Follow the project's existing coding style.
- Do not rewrite unrelated code.
- Do not perform broad refactors unless requested or clearly necessary.
- Do not silently change behavior outside the requested scope.
- Keep naming consistent with the existing project.
- Prefer simple, readable implementations over unnecessary abstraction.

If implementation reveals that the original plan is structurally wrong, stop and explain the issue before making a large architectural change.

## 8. Testing and Verification

After implementation:
1. Run the most relevant existing tests.
2. Add tests when the new behavior should be automatically verified.
3. Run the relevant application/script when practical.
4. Check that existing functionality has not been unintentionally broken.

Report:
- What was tested
- What passed
- What failed
- What could not be verified

Do not claim something works if it was not actually tested.

## 9. Post-Implementation Review

After completing a non-trivial feature, review the resulting structure:
1. Did we create unnecessary files?
2. Did we duplicate an existing class/function?
3. Could any new code have been integrated into an existing class/module?
4. Did we create an unnecessary launcher or config?
5. Are responsibilities still clear?
6. Is the new feature connected naturally to the existing execution flow?
7. Did we introduce unnecessary abstractions?
8. Did we modify unrelated code?

If cleanup is clearly needed, explain the issue and recommend the smallest appropriate cleanup. Do not perform unrelated refactoring automatically.

## 10. Communication During Coding

For non-trivial work, keep explanations focused and concrete.

Before implementation, explain the architecture-level plan rather than narrating every line of code.

After implementation, summarize:
- Modified files
- Newly created files
- What changed in each
- How the components connect
- Test/verification results

Do not overwhelm the user with unnecessary implementation narration.

## 11. Priority of Decisions

When deciding how to implement a feature, use this priority:
1. Correctness
2. Integration with the existing architecture
3. Reuse of existing code
4. Clear separation of responsibilities
5. Minimal unnecessary changes
6. Maintainability
7. Convenience of implementation

Do not choose a new file, wrapper, launcher, or abstraction merely because it is easier to implement.

## 12. AI-Assisted Development

The AI is responsible for implementation, but the project structure should remain understandable to the human developer.

When multiple reasonable approaches exist:
- Prefer the one easiest to understand and maintain.
- Explain meaningful architectural choices.
- Do not hide complexity behind unnecessary wrappers or generated files.
- Keep the relationship between entry points, core logic, and supporting modules explicit.

The final project should make it possible to answer:
- Where does execution start?
- Where is the main logic?
- Where does data enter and leave?
- Which class owns each responsibility?
- Where should the next related feature be added?
