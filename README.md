# Hydraulic Conductivity Estimation using Ryd's Formula

This script estimates hydraulic conductivity (K) from SGU well data using Ryd's empirical formula. It visualizes results as CDFs for different depth intervals.

## Features
- Downloads SGU's well data
- Clips data using a polygon shapefile
- Computes hydraulic conductivity K
- Computes hydraulic conductivity K3D according to Matherons formula
- Plots cumulative distribution functions (CDFs)
- Outputs Excel and PNG results

## References
"SGU rekommenderar att man i fortsättningen använder formeln framtagen av Ryd för att beräkna K baserat på kapacitetsdata i SGU:s brunnsarkiv från närliggande bergbrunnar.

K = 0,0756 × Q1,0255/Lw

där Lw (m) är längden av den öppna, vattenfyllda borrhålslängden i berg."

More information is provided at SGUs website https://www.sgu.se/anvandarstod-for-geologiska-fragor/bedomning-av-influensomrade-avseende-grundvatten/utgangslage-och-utredningsstrategi/berakningsexempel-hydraulisk-konduktivitet-for-vattenforande-sprickigt-berg/
![kumulativ fördelningsfunktion av K för samtliga brunnar](https://github.com/user-attachments/assets/3f64f6d8-6f77-4204-bfb6-55c96a4aa3dd)
![kumulativ fördelningsfunktion av K för djupintervall](https://github.com/user-attachments/assets/5b76618e-c68b-4ae7-881b-69fc9b3e7a02)

## Setup

```bash
pip install -r requirements.txt
python main.py
