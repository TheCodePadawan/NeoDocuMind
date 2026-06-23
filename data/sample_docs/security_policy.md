# Northwind Analytics — Information Security Policy

## 1. Password Requirements
All accounts must use a passphrase of at least 14 characters. Passwords must be
rotated every 180 days and may not reuse any of the previous five passwords.
Multi-factor authentication (MFA) is mandatory for all systems that handle
customer data.

## 2. Data Classification
Northwind classifies data into three tiers:
- **Public**: marketing material and published reports. No restrictions.
- **Internal**: project plans and internal wikis. Restricted to employees.
- **Confidential**: customer records, financials, and credentials. Access is
  granted on a strict need-to-know basis and all access is logged.

Confidential data must never be stored on personal devices or sent over
unencrypted channels.

## 3. Acceptable Use
Company devices are for business use. Installing unapproved software is
prohibited. Employees must lock their screens when away from their desk.

## 4. Incident Response
Any suspected security incident — including phishing, lost devices, or data
leaks — must be reported to security@northwind.example within one hour of
discovery. The security team aims to acknowledge reports within 30 minutes and
to contain confirmed incidents within four hours. A post-incident review is
completed within five business days.

## 5. Third-Party Vendors
Vendors that process Confidential data must sign a Data Processing Agreement and
pass an annual security review. Vendor access is revoked automatically when a
contract ends.

## 6. Encryption
Data at rest must be encrypted using AES-256. Data in transit must use TLS 1.2 or
higher. Encryption keys are rotated annually and stored in the company key vault.
