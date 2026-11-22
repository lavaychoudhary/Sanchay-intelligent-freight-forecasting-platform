\# SANCHAY — Maritime Freight Forecasting \& Chartering Optimizer



\*\*Smart India Hackathon 2026 | Problem Statement ID:\*\* `SIH26006` | \*\*Team:\*\* `Code Alphabet`



SANCHAY is an end-to-end decision-support pipeline and interactive Streamlit dashboard for \*\*8 major Indian ports\*\* (\*Paradip, Visakhapatnam, Haldia, Mundra, Kandla, Krishnapatnam, Mormugao, New Mangalore\*). It combines \*\*YOLOv8s\*\* ship detection, gradient-boosted time-series forecasting (`HistGBM`, with `LightGBM`/`XGBoost` support), and \*\*Mixed-Integer Linear Programming (MILP)\*\* (`scipy.optimize.milp`) to optimize vessel chartering and bulk cargo allocation.



\---



\## Presentation Overview

\- See `SIH\_Presentation\_updated.pptx` in the repository for the problem framing, architecture, feasibility, and strategic alignment with India's National Logistics Policy and PM Gati Shakti.



\---



\## What SANCHAY Predicts \& Optimizes

1\. \*\*Anchorage \& Congestion Signals:\*\* Tracks `anchorage\_vessel\_count`, `port\_congestion\_index`, and live uploaded-image ship detections via `best.pt` (`YOLOv8s`).

2\. \*\*Port Delay \& 7-Day Freight Forecasts:\*\*

&#x20;  - \*\*Port Delay (`expected\_delay\_days`):\*\* Adding CV/congestion features on chronological test data (`>= 2025-07-01`) reduced MAE from \*\*0.56658 to 0.45074 days\*\* (\*\*-20.4%\*\*) and improved $R^2$ from \*\*0.1722 to 0.4733\*\*.

&#x20;  - \*\*7-Day Freight Benchmark (`target\_logret\_7d` $\\rightarrow$ `pred\_benchmark\_7d\_usd\_mt`):\*\* Improves 7-day benchmark price MAE slightly (`$0.6844` to `$0.6766`/MT) and serves as a directional scenario signal alongside delay/demurrage risk.

3\. \*\*MILP Chartering Plan:\*\* Solves for cost-optimal vessel class selection (`Handysize`, `Supramax`, `Panamax`, `Capesize`), \*\*Spot Now\*\* vs. \*\*7-Day Forward\*\* voyage timing, and cargo allocation subject to port draft limits, berth capacities, and cargo demand.



\---



\## Project Structure

```text

├── app.py                              # Interactive Streamlit dashboard

├── train\_and\_optimize.py               # Forecasting + MILP chartering optimizer

├── requirements.txt                    # Python dependencies

├── sanchay\_ml\_features\_v3.csv          # v3 ML feature dataset

├── Sanchay\_Numeric\_Dataset\_v3.xlsx     # v3 multi-sheet workbook \& MILP parameters

├── best.pt                             # Trained YOLOv8s ship detection weights

├── metrics\_comparison.csv              # Without\_CV vs. With\_CV test metrics

├── CVIP_project.ipynb                   # Computer vision and ml model training 

├── results.csv                         # YOLOv8s training metrics log

├── SIH\_Presentation\_updated.pptx       # Project presentation deck

├── files/

│   ├── sanchay\_v3\_pipeline.py          # v3 dataset \& CV-alignment pipeline

│   └── run\_cv\_on\_scenes.py             # Scene-level YOLOv8s inference utility

└── models/                             # Saved forecasting model artifacts

