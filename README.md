# Fleet Electrification Analysis

## Description
This project provides a framework for analysing HGV (Heavy Goods Vehicle) fleet data to evaluate the feasibility of transitioning from ICE (Internal Combustion Engine) vehicles to electric vehicles (EVs).
This assumes telematics data is available for the ICE vehicles as well as battery energy + SOC (state of charge) simulated data

## Features
- Data cleaning for missing values, dupes, speed jumps and GPS anomalies
- Route mapping
- Trip and downtime analysis
- Fuel vs energy cost comparison
- SOC (State of Charge) tracking and flat battery prediction for current route
- Geolocation clustering and hotspot detection

## Requirements
Install required libraries:
```bash
pip install -r requirements.txt
