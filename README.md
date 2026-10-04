# BurnSight AI — Predictive & Explainable Burn-In Screening

**Smart India Hackathon 2026 | Problem Statement: SIH26170**  
**Organization:** ISRO  
**Theme:** Smart Automation  
**Category:** Software

BurnSight AI is an AI-driven burn-in screening and predictive risk monitoring system designed to identify abnormal component behaviour early, predict future degradation, and provide an explainable QA decision before the complete 168-hour burn-in cycle is finished.

The system combines **dynamic anomaly detection, early degradation prediction, risk assessment, explainability, and what-if analysis** into a unified decision-support pipeline for QA inspectors.

> **Prototype Notice:** The current prototype uses a synthetic burn-in dataset created for demonstration and model development. The results shown in this repository are not validation results on real ISRO hardware or proprietary ISRO data.

---

## Problem Statement

Traditional burn-in and Environmental Stress Screening (ESS) processes often rely on fixed parametric limits.

A component may remain within an absolute specification limit while still showing an abnormal degradation trend compared with other components in the same lot.

For example:

```text
Lot average leakage  → 10 µA
Component leakage    → 45 µA
Datasheet maximum    → 50 µA
