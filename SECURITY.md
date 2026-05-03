# Security Policy

OpenKoASR is an evaluation and leaderboard project. Please do not publish credentials, private dataset artifacts, raw audio, raw transcripts, or private per-sample predictions in issues or pull requests.

## Reporting a Vulnerability

If GitHub private vulnerability reporting is enabled for this repository, use it for security-sensitive reports. Otherwise, open an issue with a minimal description and do not include secrets, private files, tokens, or exploit details that could harm users.

Useful reports include:

- Accidental exposure of credentials or private paths
- Unsafe handling of downloaded model or dataset artifacts
- Public artifacts that may contain restricted transcripts or personal data
- Dependency or workflow issues that could expose repository secrets

## Maintainer Response

Security reports should be triaged before normal feature work. Confirmed issues should be fixed in code or documentation, followed by a note in the relevant release or pull request.
