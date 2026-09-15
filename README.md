# ForgeSight

## AI-Powered Predictive Maintenance & Machine Health Intelligence

ForgeSight is a machine health monitoring and predictive maintenance platform that uses machine learning to estimate machine failure risk and provide actionable maintenance recommendations.

## Problem

Unexpected machine failures can cause downtime, increased maintenance costs, and operational disruption. Traditional maintenance approaches may identify problems only after machine performance has already degraded.

ForgeSight provides an early indication of machine failure risk using machine operating conditions.

## Solution

ForgeSight analyzes machine parameters and uses a machine learning model to:

- Predict machine failure probability
- Calculate a Machine Health Index
- Classify machine risk
- Identify important machine factors
- Provide maintenance recommendations
- Analyze fleet-level machine health

## Dataset

ForgeSight was developed and evaluated using the **UCI AI4I 2020 Predictive Maintenance Dataset**, a synthetic dataset designed to model industrial machine behavior.

The dataset contains machine operating conditions and failure information used to develop and evaluate the prediction system.

## Machine Learning Pipeline

```text
Machine Data
     ↓
Feature Engineering
     ↓
Data Preprocessing
     ↓
Random Forest Model
     ↓
Failure Probability
     ↓
Health Index
     ↓
Risk Classification
     ↓
Maintenance Recommendation