# Security Policy

## Supported versions

nano-agent is an experimental research project without formal releases yet. Security
fixes are applied to the `main` branch only.

| Version | Supported |
| ------- | --------- |
| `main`  | Yes       |
| Older tags | No     |

## Reporting a vulnerability

Please do not open a public issue for security problems. Instead, report privately
through GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository, or contact the maintainer directly.

When reporting, include the affected component, steps to reproduce, and the impact
you observed. We aim to acknowledge a report within a few days.

## Automated checks

This repository runs CodeQL code scanning and uses Dependabot for dependency alerts.
These checks are configured under `.github/`.

## Scope note

This project automates interaction with a third-party game client. Contributors and
users are responsible for complying with the terms of service of any game or platform
they run the agent against.
