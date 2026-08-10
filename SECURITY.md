# Security Policy

## Supported Versions

Security fixes are handled for the latest released version of `tscv-vision`.
Older versions may receive fixes at the maintainer's discretion when the patch
is low risk.

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Report security concerns by emailing:

```text
dfr@esmad.ipp.pt
```

Include:

- A description of the vulnerability.
- Steps to reproduce or a minimal proof of concept.
- Affected versions or commits, if known.
- Any suggested mitigation.

The maintainer will acknowledge valid reports as soon as practical and will
coordinate disclosure after a fix is available.

## Security Scope

`tscv-vision` primarily loads NumPy `.npy` inputs and performs local numerical
processing. Treat input files as untrusted data and avoid running workflows on
files from unknown sources in privileged environments.
