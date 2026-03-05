import os
import pandas as pd
from time import time

from dataset_pipeline import identities as identity_script
from configs import experiment_config
from db_handler import write_identities, write_scenarios, write_stories, get_connection


def create_dataset():
    input_dir = experiment_config.input_dir

    # Ensure schema exists
    conn = get_connection(input_dir)
    conn.close()

    # 1. Generate and write identities
    print("Generating identities...")
    identity_rows = identity_script.identity_pipeline()
    write_identities(identity_rows, input_dir)
    print(f"Wrote {len(identity_rows)} identities")

    # 2. Read and write scenarios from CSV
    print("Loading scenarios...")
    scenarios_path = os.path.join(input_dir, "scenarios.csv")
    df_scenario = pd.read_csv(scenarios_path)
    scenario_rows = [
        {"id": int(row["id"]), "template": row["scenario"]}
        for _, row in df_scenario.iterrows()
    ]
    write_scenarios(scenario_rows, input_dir)
    print(f"Wrote {len(scenario_rows)} scenarios")

    # 3. Generate stories (Cartesian product)
    print("Generating stories...")
    identity_ids = [r["id"] for r in identity_rows]
    scenario_ids = [r["id"] for r in scenario_rows]

    story_id = 0
    time_start = time()
    rows = []
    total = len(identity_ids) ** 2 * len(scenario_ids)

    for sys_identity in identity_ids:
        for scen_identity in identity_ids:
            for scenario in scenario_ids:
                rows.append({
                    "id": story_id,
                    "system_identity_id": sys_identity,
                    "subject_identity_id": scen_identity,
                    "scenario_id": scenario,
                })

                if len(rows) >= 1000:
                    write_stories(rows, input_dir)
                    rows = []

                if story_id % 1000 == 0:
                    progress = (10 * story_id) // total
                    t = time() - time_start
                    print(f"[{'=' * progress}>{' ' * (10 - progress)}] {progress * 10}% {int(t // 60)}min {int(t % 60)}s", end="\r")

                story_id += 1

    if rows:
        write_stories(rows, input_dir)

    print(f"\nWrote {story_id} stories")


def pipeline():
    create_dataset()
