"""
Locust load test for the churn prediction endpoint - this is what
verifies the "1000+ requests/minute" resume claim with real numbers,
not a guess.

Deliberately targets POST /customers/{id}/predict, NOT
/predictions/{id}/recommend. /predict is pure CPU (XGBoost inference) +
a fast DB write - genuinely our own system's throughput ceiling.
/recommend makes a real network call to Groq's API, which has its own
rate limits and latency we don't control; load-testing it against
1000/min would measure Groq's free tier, not our system.

Run (see docs/notes.md Phase 8 for the full command and results):
    locust -f scripts/load_test.py --host http://127.0.0.1:8000
"""

import random

from locust import HttpUser, between, task

# Real customer_ids that exist in the seeded database (1 to 200,000).
MIN_CUSTOMER_ID = 1
MAX_CUSTOMER_ID = 200_000


class ChurnPredictionUser(HttpUser):
    # Near-zero wait between requests - we want to find the ceiling, not
    # simulate realistic human pacing.
    wait_time = between(0, 0.1)

    @task
    def predict_churn(self):
        customer_id = random.randint(MIN_CUSTOMER_ID, MAX_CUSTOMER_ID)
        # name= groups every request under one logical endpoint for
        # reporting - without it, Locust treats each distinct customer_id
        # URL as its own separate endpoint, scattering stats across
        # thousands of single-request rows instead of one aggregate.
        self.client.post(f"/customers/{customer_id}/predict", name="/customers/[id]/predict")
