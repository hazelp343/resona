# Security Policy

## Supported versions

resona is pre-1.0; security fixes are applied to the latest released `0.x`
series only.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/hazelp343/resona/security/advisories/new)
rather than opening a public issue.

Include a description, a minimal reproduction, and the affected version. I aim to
acknowledge reports within a few days and to ship a fix or mitigation as quickly
as is practical.

resona has no runtime network access and touches the filesystem only for the
audio and event files you explicitly pass it, so the most likely surface is
malformed input (a crafted WAV header, for instance). Reports in that area are
especially welcome.
