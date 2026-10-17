"""Quick test of the GenAI layer: IBM Granite (big-endian GGUF) via llama.cpp on LinuxONE.

    LOCAL_GGUF=/home/ibm-user/granite-3.3-2b-instruct-be.Q4_K_M.gguf python3 scripts/genai_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.advisory import genai_status, llm_explain  # noqa: E402

facts = {"traveller": "Asthma / respiratory", "mode": "walk", "aqi": 232, "category": "Poor",
         "rule_based_advice": "Expected AQI 232 (Poor) - worst likely case. Postpone non-essential trips; if you must go, "
                              "use the cleanest route and a well-fitted mask. Lowest-exposure departure in the next 24 h: Sun 19:00.",
         "fastest_route": {"minutes": 32, "km": 2.56, "dose_ug": 65.4},
         "cleanest_route": {"minutes": 34, "km": 2.71, "dose_ug": 57.1},
         "cleanest_saves_percent": 12.7, "best_time_to_leave": "Sun 19:00"}
print("status:", genai_status())
for i in range(2):                                   # 1st call includes loading the model
    t = time.time()
    out = llm_explain(facts, "en")
    print(f"\ncall {i + 1}: {time.time() - t:.1f} s")
    print(out["text"] if out else "no output (check LOCAL_GGUF)")
