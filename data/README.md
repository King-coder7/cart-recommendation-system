# Data folder

Place the project CSVs here before running the model:
- cart_export_11-15.csv
- answer_2016-01.csv
- cart_export_16_11.csv
- cart_export_17_10.csv
- cart_export_19_05.csv

The scripts in `models/` expect these files to be present in this directory.
cart-recommendation-system
Production-style cart-based recommendation system designed to forecast the products a customer is likely to add next based on historical purchase patterns, temporal context, and evaluation of recommendation quality.

The project uses shopping cart histories to build co-purchase relationships between products and evaluates recommendation performance using historical answer sets. A baseline recommender is implemented using conditional probabilities derived from repeated product co-occurrence within cart sessions. This provides a strong benchmark for understanding product affinity and temporal drift across different time windows.

The repository is structured to support experimentation with different training periods and evaluation dates, including validation against 2016 datasets. It is intended as a foundation for more advanced recommendation approaches, including temporal modeling, cold-start handling, and deployment as a production-ready API.

This project focuses on:

cart/session-based product recommendation
co-purchase signal extraction from historical transaction data
evaluation using HitRate@K
temporal drift analysis across training windows
extensibility toward production deployment and API serving
